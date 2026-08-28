"""Tenant-scoped cancellation-request coverage for the test-only v2 route."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.api import register_tracer_routes
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.run_events import EventInput, RunEventRepository


def _cancellation_app(database_url: str, *, cancellation_enabled: bool) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=database_url),
        ):
            yield

    app = FastAPI(lifespan=lifespan)
    register_tracer_routes(
        app,
        enabled=True,
        cancellation_enabled=cancellation_enabled,
    )
    return app


async def _seed_run(
    database_url: str,
    *,
    tenant_id: str = "tenant-a",
    terminal: bool = False,
) -> UUID:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        repository = RunEventRepository(pool)
        run = await repository.create_run(
            tenant_id=tenant_id,
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="seed-instance",
        )
        if terminal:
            await repository.complete_run(
                tenant_id=tenant_id,
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                event=EventInput(
                    event_key="lifecycle:completed:0",
                    type="done",
                    data={"status": "completed"},
                ),
            )
        return run.run_id


async def _persisted_state(
    database_url: str,
    *,
    tenant_id: str,
    run_id: UUID,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        async with pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT status, owner_instance_id, execution_epoch
                    FROM langgraph_v2.runs
                    WHERE tenant_id = %s AND run_id = %s
                    """,
                    (tenant_id, run_id),
                )
                run = await cursor.fetchone()
                await cursor.execute(
                    """
                    SELECT requested_at
                    FROM langgraph_v2.cancellation_intents
                    WHERE tenant_id = %s AND run_id = %s
                    """,
                    (tenant_id, run_id),
                )
                intents = await cursor.fetchall()
                await cursor.execute(
                    """
                    SELECT sequence, event_key
                    FROM langgraph_v2.events
                    WHERE tenant_id = %s AND run_id = %s
                    ORDER BY sequence
                    """,
                    (tenant_id, run_id),
                )
                events = await cursor.fetchall()
    return run, intents, events


def test_cancellation_route_is_default_off() -> None:
    app = FastAPI()
    register_tracer_routes(app, enabled=True)

    assert "/v2/runs/{run_id}/cancel" not in {
        getattr(route, "path", None) for route in app.routes
    }


def test_running_run_cancellation_is_durable_and_idempotent(
    langgraph_v2_migrated_database_url: str,
) -> None:
    run_id = asyncio.run(_seed_run(langgraph_v2_migrated_database_url))
    app = _cancellation_app(
        langgraph_v2_migrated_database_url,
        cancellation_enabled=True,
    )

    with TestClient(app) as client:
        first = client.post(
            f"/v2/runs/{run_id}/cancel",
            headers={"X-Application-Id": "tenant-a"},
        )
        repeated = client.post(
            f"/v2/runs/{run_id}/cancel",
            headers={"X-Application-Id": "tenant-a"},
        )

    assert first.status_code == repeated.status_code == 202
    assert (
        first.json()
        == repeated.json()
        == {
            "status": "accepted",
            "runId": str(run_id),
            "runStatus": "running",
        }
    )
    run, intents, events = asyncio.run(
        _persisted_state(
            langgraph_v2_migrated_database_url,
            tenant_id="tenant-a",
            run_id=run_id,
        )
    )
    assert run == {
        "status": "running",
        "owner_instance_id": "seed-instance",
        "execution_epoch": 1,
    }
    assert len(intents) == 1
    assert events == []


def test_cancellation_hides_missing_and_cross_tenant_runs(
    langgraph_v2_migrated_database_url: str,
) -> None:
    run_id = asyncio.run(_seed_run(langgraph_v2_migrated_database_url))
    app = _cancellation_app(
        langgraph_v2_migrated_database_url,
        cancellation_enabled=True,
    )

    with TestClient(app) as client:
        missing = client.post(
            f"/v2/runs/{uuid4()}/cancel",
            headers={"X-Application-Id": "tenant-a"},
        )
        cross_tenant = client.post(
            f"/v2/runs/{run_id}/cancel",
            headers={"X-Application-Id": "tenant-b"},
        )

    assert missing.status_code == 404
    assert cross_tenant.status_code == 404


def test_terminal_cancellation_is_a_non_mutating_idempotent_response(
    langgraph_v2_migrated_database_url: str,
) -> None:
    run_id = asyncio.run(_seed_run(langgraph_v2_migrated_database_url, terminal=True))
    app = _cancellation_app(
        langgraph_v2_migrated_database_url,
        cancellation_enabled=True,
    )

    with TestClient(app) as client:
        first = client.post(
            f"/v2/runs/{run_id}/cancel",
            headers={"X-Application-Id": "tenant-a"},
        )
        repeated = client.post(
            f"/v2/runs/{run_id}/cancel",
            headers={"X-Application-Id": "tenant-a"},
        )

    assert first.status_code == repeated.status_code == 200
    assert (
        first.json()
        == repeated.json()
        == {
            "status": "already_terminal",
            "runId": str(run_id),
            "runStatus": "completed",
        }
    )
    run, intents, events = asyncio.run(
        _persisted_state(
            langgraph_v2_migrated_database_url,
            tenant_id="tenant-a",
            run_id=run_id,
        )
    )
    assert run["status"] == "completed"
    assert intents == []
    assert events == [{"sequence": 1, "event_key": "lifecycle:completed:0"}]
