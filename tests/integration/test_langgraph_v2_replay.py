"""Persisted snapshot replay coverage for the default-off v2 control route."""

from __future__ import annotations

import asyncio
import json
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


def _replay_app(database_url: str, *, replay_enabled: bool) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=database_url),
        ):
            yield

    app = FastAPI(lifespan=lifespan)
    register_tracer_routes(app, enabled=True, replay_enabled=replay_enabled)
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
        for ordinal in (1, 2):
            await repository.append_event(
                tenant_id=tenant_id,
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                event=EventInput(
                    event_key=f"phase:query:step_completed:{ordinal}",
                    type="step_completed",
                    step="query",
                    data={"ordinal": ordinal, "payload": ["unchanged"]},
                ),
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


async def _durable_counts(database_url: str) -> tuple[int, int, int]:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        async with pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM langgraph_v2.runs) AS runs,
                        (SELECT count(*) FROM langgraph_v2.messages) AS messages,
                        (SELECT count(*) FROM langgraph_v2.events) AS events
                    """
                )
                row = await cursor.fetchone()
    return row["runs"], row["messages"], row["events"]


def _events(response_text: str) -> list[dict[str, object]]:
    return [
        json.loads(frame.removeprefix("data: "))
        for frame in response_text.strip().split("\n\n")
        if frame
    ]


def test_replay_route_is_default_off() -> None:
    app = FastAPI()
    register_tracer_routes(app, enabled=True)

    assert "/v2/runs/{run_id}/stream" not in {
        getattr(route, "path", None) for route in app.routes
    }


def test_replay_returns_only_snapshot_events_after_sequence_and_closes_while_running(
    langgraph_v2_migrated_database_url: str,
) -> None:
    run_id = asyncio.run(_seed_run(langgraph_v2_migrated_database_url))
    app = _replay_app(langgraph_v2_migrated_database_url, replay_enabled=True)

    with TestClient(app) as client:
        response = client.get(
            f"/v2/runs/{run_id}/stream?afterSequence=1",
            headers={"X-Application-Id": "tenant-a"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-run-id"] == str(run_id)
    assert _events(response.text) == [
        {
            "type": "step_completed",
            "sequence": 2,
            "step": "query",
            "data": {"ordinal": 2, "payload": ["unchanged"]},
        }
    ]


def test_replay_of_terminal_run_closes_and_sequence_beyond_latest_is_empty(
    langgraph_v2_migrated_database_url: str,
) -> None:
    run_id = asyncio.run(
        _seed_run(langgraph_v2_migrated_database_url, terminal=True)
    )
    app = _replay_app(langgraph_v2_migrated_database_url, replay_enabled=True)

    with TestClient(app) as client:
        replay = client.get(
            f"/v2/runs/{run_id}/stream",
            headers={"X-Application-Id": "tenant-a"},
        )
        beyond_latest = client.get(
            f"/v2/runs/{run_id}/stream?afterSequence=99",
            headers={"X-Application-Id": "tenant-a"},
        )

    assert [event["sequence"] for event in _events(replay.text)] == [1, 2, 3]
    assert beyond_latest.status_code == 200
    assert beyond_latest.text == ""


def test_replay_validates_cursor_and_hides_missing_or_cross_tenant_runs(
    langgraph_v2_migrated_database_url: str,
) -> None:
    run_id = asyncio.run(_seed_run(langgraph_v2_migrated_database_url))
    app = _replay_app(langgraph_v2_migrated_database_url, replay_enabled=True)

    with TestClient(app) as client:
        negative = client.get(
            f"/v2/runs/{run_id}/stream?afterSequence=-1",
            headers={"X-Application-Id": "tenant-a"},
        )
        malformed = client.get(
            f"/v2/runs/{run_id}/stream?afterSequence=not-a-number",
            headers={"X-Application-Id": "tenant-a"},
        )
        missing = client.get(
            f"/v2/runs/{uuid4()}/stream",
            headers={"X-Application-Id": "tenant-a"},
        )
        cross_tenant = client.get(
            f"/v2/runs/{run_id}/stream",
            headers={"X-Application-Id": "tenant-b"},
        )

    assert negative.status_code == 422
    assert malformed.status_code == 422
    assert missing.status_code == 404
    assert cross_tenant.status_code == 404


def test_replay_from_another_fastapi_instance_does_not_write_durable_records(
    langgraph_v2_migrated_database_url: str,
) -> None:
    run_id = asyncio.run(
        _seed_run(langgraph_v2_migrated_database_url, terminal=True)
    )
    first_instance = _replay_app(langgraph_v2_migrated_database_url, replay_enabled=True)
    second_instance = _replay_app(langgraph_v2_migrated_database_url, replay_enabled=True)

    with TestClient(first_instance):
        before = asyncio.run(_durable_counts(langgraph_v2_migrated_database_url))
        with TestClient(second_instance) as client:
            response = client.get(
                f"/v2/runs/{run_id}/stream",
                headers={"X-Application-Id": "tenant-a"},
            )
        after = asyncio.run(_durable_counts(langgraph_v2_migrated_database_url))

    assert response.status_code == 200
    assert [event["sequence"] for event in _events(response.text)] == [1, 2, 3]
    assert after == before
