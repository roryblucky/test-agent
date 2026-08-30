from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.graph import build_tracer_graph, canonical_query
from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.phase_results import (
    ALLOWED_PHASE_NAMES,
    PhaseExecutionContext,
    PhaseName,
    PhaseResultConflict,
    PhaseResultInput,
    PhaseResultRepository,
)
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import (
    ClaimFenced,
    EventInput,
    EventInvariantConflict,
    RunEventRepository,
)
from app.models.domain import Document


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


class RecordingPhaseResultRepository(PhaseResultRepository):
    def __init__(self, pool: AsyncConnectionPool) -> None:
        super().__init__(pool)
        self.phase_reads: list[PhaseName] = []

    async def get_or_invoke(self, **kwargs):  # type: ignore[no-untyped-def]
        self.phase_reads.append(kwargs["phase_name"])
        return await super().get_or_invoke(**kwargs)


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
async def test_phase_result_commit_wakes_local_event_follower(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        wakeups = LiveEventWakeups()
        phases = PhaseResultRepository(pool, live_events=wakeups)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )

        async with wakeups.subscribe("tenant-a", run.run_id) as subscription:
            generation = subscription.generation
            await phases.commit(
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                phase=_phase_input(),
            )

            assert subscription.generation == generation + 1


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
async def test_input_nodes_bypass_phase_and_event_journals_while_retrieval_uses_them(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class Retriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(
                documents=[Document(id="d1", content=query)],
                raw_payload={"query": query},
            )

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        phases = RecordingPhaseResultRepository(pool)
        graph = build_tracer_graph(
            phase_context=PhaseExecutionContext(
                repository=phases,
                artifact_repository=ArtifactRepository(pool),
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            ),
            retriever=Retriever(),
        )
        state = {
            "query": "hello",
            "conversation_id": "conversation-1",
            "client_request_id": None,
            "events": [],
        }

        await graph.ainvoke(state)

        assert phases.phase_reads == ["retrieval", "reranking", "finalization"]
        events = await runs.list_events("tenant-a", run.run_id)
        assert [event.event_key for event in events] == [
            "phase:retrieval:step_start:1",
            "phase:retrieval:step_completed:1",
            "phase:reranking:step_start:1",
            "phase:reranking:step_completed:1",
            "phase:finalization:step_start:1",
            "phase:finalization:step_completed:1",
        ]


@pytest.mark.asyncio
async def test_flagged_input_stops_before_refinement_and_provider_access(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class UnexpectedRefinementActor:
        async def refine(self, query, history):  # type: ignore[no-untyped-def]
            raise AssertionError(f"refinement received flagged input: {query!r}")

    class UnexpectedRetriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            raise AssertionError(f"retriever received flagged input: {query!r}")

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        phases = RecordingPhaseResultRepository(pool)
        result = await build_tracer_graph(
            phase_context=PhaseExecutionContext(
                repository=phases,
                artifact_repository=ArtifactRepository(pool),
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            ),
            refinement_actor=UnexpectedRefinementActor(),
            retriever=UnexpectedRetriever(),
        ).ainvoke(
            {
                "query": "blocked input",
                "conversation_id": "conversation-1",
                "client_request_id": None,
                "events": [],
            }
        )

        assert result["halted"] is True
        assert result["events"][-1]["type"] == "error"
        assert phases.phase_reads == []
        assert await RunEventRepository(pool).list_events("tenant-a", run.run_id) == []


@pytest.mark.asyncio
async def test_conflict_marks_run_failed_without_replacing_phase_result(
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

        persisted = await phases.get_completed("tenant-a", run.run_id, "answer")
        assert persisted is not None
        assert persisted.normalized_result == {"answer": "first", "citations": []}
        assert (await runs.get_run("tenant-a", run.run_id)).status == "failed"


@pytest.mark.asyncio
async def test_event_conflict_rolls_back_the_phase_and_prior_events(
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
        await runs.append_event(
            tenant_id="tenant-a",
            run_id=run.run_id,
            event=EventInput(
                event_key="phase:retrieval:step_completed:2",
                type="step_completed",
                step="retrieval",
                data={"value": "authoritative"},
            ),
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        phase = PhaseResultInput(
            phase_name="retrieval",
            normalized_result={"documents": []},
            events=(
                EventInput(
                    event_key="phase:retrieval:step_completed:1",
                    type="step_completed",
                    step="retrieval",
                    data={"value": "new"},
                ),
                EventInput(
                    event_key="phase:retrieval:step_completed:2",
                    type="step_completed",
                    step="retrieval",
                    data={"value": "conflict"},
                ),
            ),
        )

        with pytest.raises(EventInvariantConflict):
            await phases.commit(
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                phase=phase,
            )

        assert await phases.get_completed("tenant-a", run.run_id, "retrieval") is None
        events = await runs.list_events("tenant-a", run.run_id)
        assert [event.event_key for event in events] == [
            "phase:retrieval:step_completed:2"
        ]


@pytest.mark.asyncio
async def test_duplicate_candidate_event_conflict_rolls_back_everything(
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
        phase = PhaseResultInput(
            phase_name="retrieval",
            normalized_result={"documents": []},
            events=(
                EventInput(
                    event_key="phase:retrieval:step_completed:1",
                    type="step_completed",
                    step="retrieval",
                    data={"value": "first"},
                ),
                EventInput(
                    event_key="phase:retrieval:step_completed:1",
                    type="step_completed",
                    step="retrieval",
                    data={"value": "second"},
                ),
            ),
        )
        with pytest.raises(EventInvariantConflict):
            await phases.commit(
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                phase=phase,
            )
        assert await phases.get_completed("tenant-a", run.run_id, "retrieval") is None
        assert await runs.list_events("tenant-a", run.run_id) == []
        assert (await runs.get_run("tenant-a", run.run_id)).status == "failed"


@pytest.mark.asyncio
async def test_stale_epoch_cannot_commit_a_replacement(
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


def test_phase_result_rejects_volatile_or_unstructured_content() -> None:
    with pytest.raises(ValidationError):
        PhaseResultInput(
            phase_name="query",
            normalized_result={"query": "hello", "timestamp": "volatile"},
            events=(),
        )
    with pytest.raises(ValidationError):
        PhaseResultInput(
            phase_name="query",
            normalized_result="raw provider response",
            events=(),
        )
    with pytest.raises(ValidationError):
        PhaseResultInput(
            phase_name="query",
            normalized_result={"query": "hello"},
            artifact_refs=[{"artifact_id": "a-1", "timestamp_ms": 10}],
            events=(),
        )
    with pytest.raises(ValidationError):
        PhaseResultInput(
            phase_name="query",
            normalized_result={"query": "hello"},
            events=(
                EventInput(
                    event_key="phase:query:step_completed:1",
                    type="step_completed",
                    data={"timestamp": "volatile"},
                ),
            ),
        )


def test_query_canonicalization_is_unicode_and_line_ending_stable() -> None:
    assert canonical_query("  e\u0301\r\n  x  ") == "é\n  x"


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
