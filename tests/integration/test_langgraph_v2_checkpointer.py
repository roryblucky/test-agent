from __future__ import annotations

from typing import Any
from uuid import uuid4

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.checkpointing import (
    FencedAsyncPostgresSaver,
    checkpoint_namespace_for,
    exact_checkpoint_config,
    initial_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.graph import build_tracer_graph
from app.langgraph_v2.run_events import ClaimFenced, RunEventRepository


async def _setup_saver(pool: Any) -> AsyncPostgresSaver:
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver


def _state(query: str, conversation_id: str) -> dict:
    return {
        "query": query,
        "conversation_id": conversation_id,
        "client_request_id": None,
        "events": [],
    }


def test_identifier_encoding_keeps_tenant_boundaries_and_tuple_shapes_distinct() -> (
    None
):
    assert thread_id_for("tenant/a", "conversation|1") != thread_id_for(
        "tenant", "a/conversation|1"
    )
    assert checkpoint_namespace_for("tenant-a", "run-1", 1) != checkpoint_namespace_for(
        "tenant-a", "run-1", 2
    )


@pytest.mark.asyncio
async def test_committed_checkpoint_is_read_by_a_fresh_saver(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await _setup_saver(pool)

        async def write_pointer(checkpoint_id: str, checkpoint_ns: str) -> None:
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
            )

        checkpoint_ns = checkpoint_namespace_for(
            "tenant-a", str(run_id), run.execution_epoch
        )
        graph = build_tracer_graph(
            FencedAsyncPostgresSaver(
                pool,
                checkpoint_namespace=checkpoint_ns,
                pointer_writer=write_pointer,
            )
        )
        await graph.ainvoke(
            _state("hello", "conversation-1"),
            config=initial_checkpoint_config(
                thread_id=thread_id_for("tenant-a", "conversation-1"),
                checkpoint_ns=checkpoint_ns,
            ),
        )

        persisted_run = await repository.get_run("tenant-a", run_id)
        assert persisted_run.checkpoint_id is not None
        assert persisted_run.checkpoint_ns == checkpoint_ns

        fresh_saver = await _setup_saver(pool)
        checkpoint = await fresh_saver.aget(
            exact_checkpoint_config(
                thread_id=thread_id_for("tenant-a", "conversation-1"),
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=persisted_run.checkpoint_id,
            )
        )
        assert checkpoint is not None


@pytest.mark.asyncio
async def test_checkpoint_namespace_prevents_cross_tenant_lookup(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await _setup_saver(pool)

        async def write_pointer(checkpoint_id: str, checkpoint_ns: str) -> None:
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
            )

        checkpoint_ns = checkpoint_namespace_for(
            "tenant-a", str(run_id), run.execution_epoch
        )
        graph = build_tracer_graph(
            FencedAsyncPostgresSaver(
                pool,
                checkpoint_namespace=checkpoint_ns,
                pointer_writer=write_pointer,
            )
        )
        await graph.ainvoke(
            _state("hello", "conversation-1"),
            config=initial_checkpoint_config(
                thread_id=thread_id_for("tenant-a", "conversation-1"),
                checkpoint_ns=checkpoint_ns,
            ),
        )
        persisted_run = await repository.get_run("tenant-a", run_id)
        assert persisted_run.checkpoint_id is not None

        fresh_saver = await _setup_saver(pool)
        assert (
            await fresh_saver.aget(
                exact_checkpoint_config(
                    thread_id=thread_id_for("tenant-b", "conversation-1"),
                    checkpoint_ns=checkpoint_namespace_for(
                        "tenant-b", str(run_id), run.execution_epoch
                    ),
                    checkpoint_id=persisted_run.checkpoint_id,
                )
            )
            is None
        )


@pytest.mark.asyncio
async def test_stale_saver_checkpoint_is_orphaned_and_cannot_move_run_pointer(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await _setup_saver(pool)
        stale_checkpoint_ids: list[str] = []

        async def stale_pointer(checkpoint_id: str, checkpoint_ns: str) -> None:
            stale_checkpoint_ids.append(checkpoint_id)
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id="instance-a",
                execution_epoch=1,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
            )

        stale_namespace = checkpoint_namespace_for(
            "tenant-a", str(run_id), run.execution_epoch
        )
        stale_graph = build_tracer_graph(
            FencedAsyncPostgresSaver(
                pool,
                checkpoint_namespace=stale_namespace,
                pointer_writer=stale_pointer,
            )
        )
        await stale_graph.ainvoke(
            _state("hello", "conversation-1"),
            config=initial_checkpoint_config(
                thread_id=thread_id_for("tenant-a", "conversation-1"),
                checkpoint_ns=checkpoint_namespace_for(
                    "tenant-a", str(run_id), run.execution_epoch
                ),
            ),
        )
        assert stale_checkpoint_ids

        with psycopg.connect(
            langgraph_v2_migrated_database_url,
            autocommit=True,
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET owner_instance_id = 'instance-b',
                    execution_epoch = 2,
                    heartbeat_at = clock_timestamp(),
                    expires_at = clock_timestamp() + interval '30 seconds'
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run_id),
            )

        stale_checkpoint_ids.clear()
        with pytest.raises(ClaimFenced):
            await stale_graph.ainvoke(
                _state("retry", "conversation-1"),
                config=initial_checkpoint_config(
                    thread_id=thread_id_for("tenant-a", "conversation-1"),
                    checkpoint_ns=stale_namespace,
                ),
            )
        assert stale_checkpoint_ids
        orphan_id = stale_checkpoint_ids[-1]

        async def fresh_pointer(checkpoint_id: str, checkpoint_ns: str) -> None:
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id="instance-b",
                execution_epoch=2,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
            )

        new_namespace = checkpoint_namespace_for("tenant-a", str(run_id), 2)
        fresh_graph = build_tracer_graph(
            FencedAsyncPostgresSaver(
                pool,
                checkpoint_namespace=new_namespace,
                pointer_writer=fresh_pointer,
            )
        )
        await fresh_graph.ainvoke(
            _state("resume", "conversation-1"),
            config=initial_checkpoint_config(
                thread_id=thread_id_for("tenant-a", "conversation-1"),
                checkpoint_ns=new_namespace,
            ),
        )
        current_run = await repository.get_run("tenant-a", run_id)
        assert current_run.checkpoint_ns == new_namespace
        assert current_run.checkpoint_id != orphan_id
        fresh_saver = await _setup_saver(pool)
        assert (
            await fresh_saver.aget(
                exact_checkpoint_config(
                    thread_id=thread_id_for("tenant-a", "conversation-1"),
                    checkpoint_ns=new_namespace,
                    checkpoint_id=orphan_id,
                )
            )
            is None
        )


@pytest.mark.asyncio
async def test_pointer_failure_leaves_committed_checkpoint_without_advancing_run(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await _setup_saver(pool)
        checkpoint_ns = checkpoint_namespace_for(
            "tenant-a", str(run_id), run.execution_epoch
        )
        committed_ids: list[str] = []

        async def failing_pointer(checkpoint_id: str, _: str) -> None:
            committed_ids.append(checkpoint_id)
            raise RuntimeError("simulated pointer transaction failure")

        graph = build_tracer_graph(
            FencedAsyncPostgresSaver(
                pool,
                checkpoint_namespace=checkpoint_ns,
                pointer_writer=failing_pointer,
            )
        )
        with pytest.raises(RuntimeError, match="simulated pointer"):
            await graph.ainvoke(
                _state("hello", "conversation-1"),
                config=initial_checkpoint_config(
                    thread_id=thread_id_for("tenant-a", "conversation-1"),
                    checkpoint_ns=checkpoint_ns,
                ),
            )

        assert committed_ids
        persisted_run = await repository.get_run("tenant-a", run_id)
        assert persisted_run.checkpoint_id is None
        assert persisted_run.checkpoint_ns is None
        fresh_saver = await _setup_saver(pool)
        assert (
            await fresh_saver.aget(
                exact_checkpoint_config(
                    thread_id=thread_id_for("tenant-a", "conversation-1"),
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=committed_ids[0],
                )
            )
            is not None
        )
