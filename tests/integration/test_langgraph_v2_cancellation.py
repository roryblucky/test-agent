"""Tenant-scoped cancellation-request coverage for the test-only v2 runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult, AnswerStreamChunk
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.cancellation import CancellationRepository
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
)
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.pre_moderation import ModerationDecision
from app.langgraph_v2.question_refinement import V2ResolvedQuery
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import ClaimFenced, EventInput, RunEventRepository
from app.models.domain import Document
from app.services.events import EventEmitter
from tests.integration.test_langgraph_v2_tracer import parse_sse, persistent_tracer_app

STOPPED_FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "langgraph_v2" / "v1_stopped_wire.json"
)


async def _seed_run(
    database_url: str,
    *,
    tenant_id: str = "tenant-a",
    terminal: bool = False,
) -> UUID:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        await ConversationMessageRepository(pool).resolve_conversation(
            context=TrustedRequestContext(tenant_id=tenant_id, subject_id="subject-a"),
            conversation_id="conversation-1",
        )
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


async def _persisted_message_roles(database_url: str, run_id: UUID) -> list[str]:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        async with pool.connection() as connection:
            result = await connection.execute(
                """
                SELECT role FROM langgraph_v2.messages
                WHERE tenant_id = 'tenant-a' AND run_id = %s
                ORDER BY created_at
                """,
                (run_id,),
            )
            return [row[0] for row in await result.fetchall()]


@pytest.mark.asyncio
async def test_owner_atomically_applies_cancellation_once_and_fences_stale_epoch(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        cancellations = CancellationRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="owner-a",
        )
        await cancellations.request(tenant_id="tenant-a", run_id=run.run_id)

        stopped = await cancellations.apply_if_requested(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        repeated = await cancellations.apply_if_requested(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        persisted_run = await runs.get_run("tenant-a", run.run_id)
        events = await runs.list_events("tenant-a", run.run_id)

        assert stopped == repeated == events[-1]
        assert stopped is not None
        assert stopped.event_key == "lifecycle:cancelled:1"
        assert stopped.type == "stopped"
        assert stopped.data == {"partial": None}
        assert persisted_run.status == "cancelled"
        assert persisted_run.owner_instance_id == ""
        assert persisted_run.completed_at is None

        with pytest.raises(ClaimFenced):
            await cancellations.apply_if_requested(
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id="stale-owner",
                execution_epoch=0,
            )


@pytest.mark.asyncio
async def test_failed_run_rejects_later_cancellation_transition(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        cancellations = CancellationRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="owner-a",
        )
        failed = await runs.fail_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            event=EventInput(
                event_key="lifecycle:failed:1",
                type="error",
                data="provider failed",
            ),
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )

        request = await cancellations.request(tenant_id="tenant-a", run_id=run.run_id)
        with pytest.raises(ClaimFenced):
            await cancellations.apply_if_requested(
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            )

        persisted = await runs.get_run("tenant-a", run.run_id)
        assert request.accepted is False
        assert persisted.status == "failed"
        assert await runs.list_events("tenant-a", run.run_id) == [failed]


@pytest.mark.asyncio
async def test_interrupted_run_with_pending_intent_cancels_after_explicit_resume(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        cancellations = CancellationRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="owner-a",
        )
        await runs.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        await runs.interrupt_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        request = await cancellations.request(tenant_id="tenant-a", run_id=run.run_id)

        resumed = await runs.resume_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="owner-b",
        )
        stopped = await cancellations.apply_if_requested(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=resumed.owner_instance_id,
            execution_epoch=resumed.execution_epoch,
        )

        assert request.accepted is True
        assert stopped is not None
        assert stopped.event_key == "lifecycle:cancelled:2"
        assert (await runs.get_run("tenant-a", run.run_id)).status == "cancelled"
        with pytest.raises(ClaimFenced):
            await cancellations.apply_if_requested(
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            )


@pytest.mark.asyncio
async def test_expired_owner_cannot_apply_pending_intent_after_resume_epoch(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        cancellations = CancellationRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="owner-a",
        )
        await runs.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        with psycopg.connect(
            langgraph_v2_migrated_database_url, autocommit=True
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run.run_id),
            )
        await cancellations.request(tenant_id="tenant-a", run_id=run.run_id)
        interrupted = await runs.interrupt_expired_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            observed_execution_epoch=run.execution_epoch,
        )
        resumed = await runs.resume_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="owner-b",
        )

        with pytest.raises(ClaimFenced):
            await cancellations.apply_if_requested(
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            )
        stopped = await cancellations.apply_if_requested(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=resumed.owner_instance_id,
            execution_epoch=resumed.execution_epoch,
        )

        assert interrupted is not None
        assert interrupted.event_key == "lifecycle:interrupted:1"
        assert stopped is not None
        assert stopped.event_key == "lifecycle:cancelled:3"
        assert [
            event.event_key for event in await runs.list_events("tenant-a", run.run_id)
        ] == ["lifecycle:interrupted:1", "lifecycle:cancelled:3"]


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id="d1", content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        del query
        return RerankingResult(documents=documents)


class _CancelBeforeAnswer:
    def __init__(self) -> None:
        self.app: FastAPI | None = None

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        del query, documents, history
        assert self.app is not None
        await _request_current_run_cancellation(self.app)
        return AnswerResult(answer="must never be published")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationTurn]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


async def _request_current_run_cancellation(app: FastAPI) -> None:
    pool = app.state.langgraph_v2_postgres_pool
    async with pool.connection() as connection:
        result = await connection.execute(
            """
            SELECT run_id FROM langgraph_v2.runs
            WHERE tenant_id = 'tenant-a' AND status = 'running'
            ORDER BY created_at DESC LIMIT 1
            """
        )
        row = await result.fetchone()
    await CancellationRepository(pool).request(tenant_id="tenant-a", run_id=row[0])


class _CancelDuringRefinement:
    def __init__(self) -> None:
        self.app: FastAPI | None = None

    async def refine(
        self, query: str, history: Sequence[ConversationTurn]
    ) -> V2ResolvedQuery:
        del history
        assert self.app is not None
        await _request_current_run_cancellation(self.app)
        return V2ResolvedQuery(original_query=query, standalone_query=query)


class _Answer:
    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        del query, documents, history
        return AnswerResult(answer="One. Two.")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationTurn]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _CancelDuringPostModeration:
    def __init__(self) -> None:
        self.app: FastAPI | None = None
        self.calls = 0

    async def check(self, text: str) -> ModerationDecision:
        del text
        self.calls += 1
        if self.calls == 2:
            assert self.app is not None
            await _request_current_run_cancellation(self.app)
        return ModerationDecision(is_flagged=False)


def test_public_executor_observes_cancellation_before_answer_batch(
    langgraph_v2_migrated_database_url: str,
) -> None:
    actor = _CancelBeforeAnswer()
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=actor,
    )
    actor.app = app

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    events = parse_sse(response.text)
    assert events[-1]["type"] == "stopped"
    assert events[-1]["data"] == {"partial": None}
    public_stopped = {
        key: value for key, value in events[-1].items() if key != "sequence"
    }
    assert public_stopped == json.loads(STOPPED_FIXTURE_PATH.read_text())["events"][0]
    assert not any(
        event.get("step") == "llm:answer" or event["type"] == "token"
        for event in events
    )
    run_id = UUID(response.headers["x-run-id"])
    run, _, persisted_events = asyncio.run(
        _persisted_state(
            langgraph_v2_migrated_database_url,
            tenant_id="tenant-a",
            run_id=run_id,
        )
    )
    assert run["status"] == "cancelled"
    assert run["owner_instance_id"] == ""
    assert persisted_events[-1]["event_key"] == "lifecycle:cancelled:1"


def test_public_executor_checks_cancellation_at_next_graph_boundary(
    langgraph_v2_migrated_database_url: str,
) -> None:
    actor = _CancelDuringRefinement()
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=actor,
        retriever=_Retriever(),
    )
    actor.app = app

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    events = parse_sse(response.text)
    assert events[-1]["type"] == "stopped"
    assert any(event.get("step") == "llm:refine_question" for event in events)
    assert not any(event.get("step") == "retriever" for event in events)


def test_committed_answer_batch_is_fully_delivered_before_stopped(
    langgraph_v2_migrated_database_url: str,
) -> None:
    moderation = _CancelDuringPostModeration()
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        retriever=_Retriever(),
        ranker=_Ranker(),
        moderation_provider=moderation,
        answer_actor=_Answer(),
    )
    moderation.app = app

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    events = parse_sse(response.text)
    token_events = [event for event in events if event["type"] == "token"]
    assert "".join(event["data"] for event in token_events) == "One. Two."
    assert events[-1]["type"] == "stopped"
    assert not any(event["type"] == "done" for event in events)
    assert asyncio.run(
        _persisted_message_roles(
            langgraph_v2_migrated_database_url,
            UUID(response.headers["x-run-id"]),
        )
    ) == ["user"]


@pytest.mark.asyncio
async def test_stopped_event_matches_captured_v1_wire_shape() -> None:
    fixture = json.loads(STOPPED_FIXTURE_PATH.read_text())
    emitter = EventEmitter()
    await emitter.emit_stopped()

    legacy_frames = [line async for line in emitter]

    assert [json.loads(line.removeprefix("data: ")) for line in legacy_frames] == (
        fixture["events"]
    )
