"""Replay/live following coverage for the default-off v2 control tracer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.api import register_tracer_routes
from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.replay import PersistedEventFollower
from app.langgraph_v2.run_events import EventInput, RunEventRepository


async def _seed(repository: RunEventRepository) -> tuple[UUID, str, int]:
    run = await repository.create_run(
        tenant_id="tenant-a",
        run_id=uuid4(),
        conversation_id="conversation-1",
        owner_instance_id="writer",
    )
    event = await repository.append_event(
        tenant_id="tenant-a",
        run_id=run.run_id,
        owner_instance_id=run.owner_instance_id,
        execution_epoch=run.execution_epoch,
        event=EventInput(
            event_key="phase:query:step_completed:1",
            type="step_completed",
            step="query",
            data={"ordinal": 1},
        ),
    )
    return run.run_id, run.owner_instance_id, event.sequence


def _live_endpoint(app: FastAPI):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/v2/runs/{run_id}/stream"
    )


def _request(app: FastAPI) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v2/runs/run/stream",
            "headers": [],
            "app": app,
        }
    )


@pytest.mark.asyncio
async def test_public_replay_then_live_is_gapless_and_closes_at_terminal(
    langgraph_v2_migrated_database_url: str,
) -> None:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=langgraph_v2_migrated_database_url),
        ):
            yield

    app = FastAPI(lifespan=lifespan)
    register_tracer_routes(app, enabled=True, replay_enabled=True)
    async with app.router.lifespan_context(app):
        pool = app.state.langgraph_v2_postgres_pool
        repository = RunEventRepository(
            pool, live_events=app.state.langgraph_v2_live_events
        )
        run_id, owner, _ = await _seed(repository)
        response = await _live_endpoint(app)(
            run_id,
            _request(app),
            "tenant-a",
            after_sequence=0,
        )
        frames = response.body_iterator

        first = json.loads((await anext(frames)).removeprefix("data: "))
        assert first["sequence"] == 1

        second_event = await repository.append_event(
            tenant_id="tenant-a",
            run_id=run_id,
            owner_instance_id=owner,
            execution_epoch=1,
            event=EventInput(
                event_key="phase:retrieval:step_completed:1",
                type="step_completed",
                step="retrieval",
                data={"ordinal": 2},
            ),
        )
        second = json.loads((await anext(frames)).removeprefix("data: "))
        assert second["sequence"] == second_event.sequence == 2

        terminal = await repository.complete_run(
            tenant_id="tenant-a",
            run_id=run_id,
            owner_instance_id=owner,
            execution_epoch=1,
            event=EventInput(
                event_key="lifecycle:completed:0",
                type="done",
                data={"status": "completed"},
            ),
        )
        done = json.loads((await anext(frames)).removeprefix("data: "))
        assert done["sequence"] == terminal.sequence == 3
        with pytest.raises(StopAsyncIteration):
            await anext(frames)


@pytest.mark.asyncio
async def test_remote_wakeup_loss_uses_polling_without_sequence_loss(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        writer = RunEventRepository(pool, live_events=LiveEventWakeups())
        run_id, owner, _ = await _seed(writer)
        follower = PersistedEventFollower(
            RunEventRepository(pool),
            LiveEventWakeups(),
            poll_interval_seconds=0.001,
        ).follow(tenant_id="tenant-a", run_id=run_id, after_sequence=1)
        waiting = asyncio.create_task(anext(follower))
        await asyncio.sleep(0)
        appended = await writer.append_event(
            tenant_id="tenant-a",
            run_id=run_id,
            owner_instance_id=owner,
            execution_epoch=1,
            event=EventInput(
                event_key="phase:retrieval:step_completed:1",
                type="step_completed",
                step="retrieval",
                data={"ordinal": 2},
            ),
        )

        assert (await waiting).sequence == appended.sequence == 2


@pytest.mark.asyncio
async def test_cursor_beyond_latest_waits_for_running_run_then_closes_terminal(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunEventRepository(pool)
        run_id, owner, _ = await _seed(repository)
        follower = PersistedEventFollower(
            repository,
            LiveEventWakeups(),
            poll_interval_seconds=0.001,
        ).follow(tenant_id="tenant-a", run_id=run_id, after_sequence=99)
        waiting = asyncio.create_task(anext(follower))
        await asyncio.sleep(0.01)
        assert not waiting.done()
        await repository.complete_run(
            tenant_id="tenant-a",
            run_id=run_id,
            owner_instance_id=owner,
            execution_epoch=1,
            event=EventInput(
                event_key="lifecycle:completed:0",
                type="done",
                data={"status": "completed"},
            ),
        )
        with pytest.raises(StopAsyncIteration):
            await waiting


@pytest.mark.asyncio
async def test_follower_converts_one_expired_claim_to_interrupted_event(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunEventRepository(pool)
        run_id, _, _ = await _seed(repository)
        async with pool.connection() as connection:
            await connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = 'tenant-a' AND run_id = %s
                """,
                (run_id,),
            )
        follower = PersistedEventFollower(
            repository,
            LiveEventWakeups(),
            poll_interval_seconds=0.001,
        ).follow(tenant_id="tenant-a", run_id=run_id, after_sequence=1)

        interrupted = await anext(follower)
        assert interrupted.event_key == "lifecycle:interrupted:1"
        assert interrupted.data == {"status": "interrupted", "reason": "claim_expired"}
        with pytest.raises(StopAsyncIteration):
            await anext(follower)
        run = await repository.get_run("tenant-a", run_id)
        assert (run.status, run.execution_epoch, run.owner_instance_id) == (
            "interrupted",
            2,
            "",
        )


@pytest.mark.asyncio
async def test_only_one_expiry_cas_wins_and_losers_replay_its_event(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunEventRepository(pool)
        run_id, _, _ = await _seed(repository)
        async with pool.connection() as connection:
            await connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = 'tenant-a' AND run_id = %s
                """,
                (run_id,),
            )

        first, second = await asyncio.gather(
            repository.interrupt_expired_run(
                tenant_id="tenant-a", run_id=run_id, observed_execution_epoch=1
            ),
            RunEventRepository(pool).interrupt_expired_run(
                tenant_id="tenant-a", run_id=run_id, observed_execution_epoch=1
            ),
        )

        assert sum(event is not None for event in (first, second)) == 1
        assert [event.event_key for event in await repository.list_events("tenant-a", run_id)] == [
            "phase:query:step_completed:1",
            "lifecycle:interrupted:1",
        ]
