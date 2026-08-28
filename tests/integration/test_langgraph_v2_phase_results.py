from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from app.langgraph_v2.phase_results import (
    ALLOWED_PHASE_NAMES,
    PhaseName,
    PhaseResultConflict,
    PhaseResultInput,
    PhaseResultRepository,
)
from app.langgraph_v2.run_events import ClaimFenced, EventInput, RunEventRepository


def _phase_input(
    phase_name: PhaseName = "query",
    normalized_result: dict[str, Any] | None = None,
) -> PhaseResultInput:
    return PhaseResultInput(
        phase_name=phase_name,
        normalized_result=normalized_result
        or {"query": "hello", "history_snapshot": ["prior"]},
        events=(
            EventInput(
                event_key=f"phase:{phase_name}:step_completed:1",
                type="step_completed",
                step=phase_name,
                data={"status": "completed"},
            ),
        ),
    )


@pytest.mark.asyncio
async def test_phase_result_commit_is_atomic_and_idempotent(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        phases = PhaseResultRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        phase = _phase_input()

        first = await phases.commit(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            phase=phase,
        )
        second = await phases.commit(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            phase=phase,
        )

        assert first == second
        assert await phases.get_completed("tenant-a", run.run_id, "query") == first
        events = await runs.list_events("tenant-a", run.run_id)
        assert len(events) == 1
        assert events[0].sequence == 1
        assert "owner_instance_id" not in first.canonical_result
        assert "attempt" not in first.canonical_result


@pytest.mark.asyncio
async def test_phase_replay_reuses_result_without_invocation(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        phases = PhaseResultRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        phase = _phase_input("retrieval", {"artifacts": [{"artifact_id": "a-1"}]})
        await phases.commit(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            phase=phase,
        )

        invoked = False

        async def invoke() -> PhaseResultInput:
            nonlocal invoked
            invoked = True
            return _phase_input("retrieval", {"artifacts": [{"artifact_id": "wrong"}]})

        replayed = await phases.get_or_invoke(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            phase_name="retrieval",
            invoke=invoke,
        )

        assert replayed.normalized_result == {"artifacts": [{"artifact_id": "a-1"}]}
        assert invoked is False


@pytest.mark.asyncio
async def test_conflict_and_stale_epoch_cannot_replace_phase_result(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        phases = PhaseResultRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        phase = _phase_input("answer", {"answer": "first", "citations": []})
        await phases.commit(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            phase=phase,
        )

        with pytest.raises(PhaseResultConflict):
            await phases.commit(
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                phase=_phase_input("answer", {"answer": "different", "citations": []}),
            )

        with psycopg.connect(
            langgraph_v2_migrated_database_url, autocommit=True
        ) as connection:
            connection.execute(
                "UPDATE langgraph_v2.runs SET expires_at = clock_timestamp() - interval '1 second' WHERE tenant_id = %s AND run_id = %s",
                ("tenant-a", run.run_id),
            )
        with pytest.raises(ClaimFenced):
            await phases.commit(
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                phase=_phase_input(
                    "answer", {"answer": "replacement", "citations": []}
                ),
            )

        persisted = await phases.get_completed("tenant-a", run.run_id, "answer")
        assert persisted is not None
        assert persisted.normalized_result == {"answer": "first", "citations": []}


def test_phase_names_are_exactly_the_nine_linear_phases() -> None:
    assert ALLOWED_PHASE_NAMES == {
        "query",
        "pre_moderation",
        "question_refinement",
        "retrieval",
        "reranking",
        "answer",
        "groundedness",
        "post_moderation",
        "finalization",
    }
    with pytest.raises(ValidationError):
        PhaseResultInput(
            phase_name=cast(Any, "citations"),
            normalized_result={"citation": "not-a-phase"},
            events=(),
        )
