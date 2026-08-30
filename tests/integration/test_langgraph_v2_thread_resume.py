from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid4

import fastapi
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult, AnswerStreamChunk
from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import initial_checkpoint_config
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    MessageRecord,
    TurnRecord,
)
from app.langgraph_v2.graph import LinearGraphState, build_linear_graph
from app.langgraph_v2.groundedness import GroundednessAssessment
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.output_assessments import MockOutputAssessmentAudit
from app.langgraph_v2.post_moderation import ModerationDecision
from app.langgraph_v2.question_refinement import (
    QuestionRefinementResult,
    V2ResolvedQuery,
)
from app.models.domain import Document
from tests.integration.test_langgraph_v2_linear_core import (
    UAT_CONTRACT_PATH,
    close_stream_after_first_token,
    parse_sse,
    persistent_linear_app,
    seed_subject_conversation,
    stream_request,
    v2_stream_endpoint,
)


def _thread_resume_endpoint(app: fastapi.FastAPI) -> Any:
    for route in app.router.routes:
        if (
            isinstance(route, APIRoute)
            and route.path == "/v2/threads/{thread_id}/resume/stream"
        ):
            return route.endpoint
    raise LookupError("thread resume endpoint is not registered")


class _AnswerActor:
    def __init__(self) -> None:
        self.calls = 0

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        self.calls += 1
        assert query == "resume me"
        assert [document.id for document in documents] == ["mock-doc-1"]
        assert history == []
        return AnswerResult(answer="recovered answer", usage={"output_tokens": 2})

    async def answer_stream(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta="recovered ")
        yield AnswerStreamChunk(delta="answer")
        yield AnswerStreamChunk(result=result)


class _StreamingAnswerActor:
    def __init__(
        self,
        *,
        answer: str,
        stream_tokens: list[str],
        block_after_first_token: bool = False,
    ) -> None:
        self.answer_text = answer
        self.stream_tokens = stream_tokens
        self.block_after_first_token = block_after_first_token
        self.calls = 0
        self.blocked_on_followup_read = asyncio.Event()
        self.close_completed = asyncio.Event()

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        self.calls += 1
        assert query == "hello"
        assert [document.id for document in documents] == ["mock-doc-1"]
        assert history == []
        return AnswerResult(answer=self.answer_text)

    async def answer_stream(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        if self.block_after_first_token:
            yield AnswerStreamChunk(delta=self.stream_tokens[0])
            self.blocked_on_followup_read.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.close_completed.set()
            return
        for token in self.stream_tokens:
            yield AnswerStreamChunk(delta=token)
        yield AnswerStreamChunk(result=result)


class _AssertingGroundednessActor:
    def __init__(self, expected_answer: str) -> None:
        self.calls = 0
        self.expected_answer = expected_answer

    async def evaluate(
        self, answer: str, documents: list[Document]
    ) -> GroundednessAssessment:
        self.calls += 1
        assert answer == self.expected_answer
        assert [document.id for document in documents] == ["mock-doc-1"]
        return GroundednessAssessment(is_grounded=True, score=1.0, details="advisory")


class _AssertingModerationProvider:
    def __init__(self, expected_answer: str) -> None:
        self.calls = 0
        self.seen_texts: list[str] = []
        self.expected_answer = expected_answer

    async def check(self, text: str) -> ModerationDecision:
        self.calls += 1
        self.seen_texts.append(text)
        assert text in {"hello", self.expected_answer}
        return ModerationDecision(is_flagged=False)


class _BlockingRefinementActor:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.close_completed = asyncio.Event()

    async def refine(
        self, query: str, history: Sequence[ConversationTurn]
    ) -> QuestionRefinementResult:
        del history
        self.calls += 1
        assert query == "hello"
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.close_completed.set()
        raise AssertionError("unreachable")


class _SuccessfulRefinementActor:
    def __init__(self) -> None:
        self.calls = 0

    async def refine(
        self, query: str, history: Sequence[ConversationTurn]
    ) -> QuestionRefinementResult:
        del history
        self.calls += 1
        return QuestionRefinementResult(
            resolved_query=V2ResolvedQuery(original_query=query, standalone_query=query)
        )


async def _interrupt_query_after_answer_token(
    database_url: str,
) -> tuple[str, UUID, datetime, object]:
    await seed_subject_conversation(database_url, "conversation-1")
    answer_actor = _StreamingAnswerActor(
        answer="partial answer [1]",
        stream_tokens=["partial "],
        block_after_first_token=True,
    )
    app = persistent_linear_app(
        database_url,
        answer_actor=answer_actor,
    )
    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        token_frame = await close_stream_after_first_token(
            response.body_iterator, answer_actor.blocked_on_followup_read
        )
        assert answer_actor.close_completed.is_set() is True
        assert token_frame["data"] == "partial "
        return (
            response.headers["x-thread-id"],
            UUID(response.headers["x-turn-id"]),
            (
                await _read_turn(database_url, UUID(response.headers["x-turn-id"]))
            ).resume_deadline,
            app.state.langgraph_v2_postgres_pool,
        )


async def _interrupt_query_during_refinement(
    database_url: str,
) -> tuple[str, UUID, _BlockingRefinementActor, object]:
    conversation_id = "conversation-task41-refinement-resume"
    await seed_subject_conversation(database_url, conversation_id)
    actor = _BlockingRefinementActor()
    app = persistent_linear_app(database_url, refinement_actor=actor)
    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id=conversation_id),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )

        async def consume_stream() -> None:
            async for _ in response.body_iterator:
                pass

        pending_read = asyncio.create_task(consume_stream())
        await actor.started.wait()
        pending_read.cancel()
        with suppress(asyncio.CancelledError):
            await pending_read
        assert actor.close_completed.is_set() is True
        return (
            response.headers["x-thread-id"],
            UUID(response.headers["x-turn-id"]),
            actor,
            app.state.langgraph_v2_postgres_pool,
        )


async def _seed_pre_answer_checkpoint(
    database_url: str,
    *,
    conversation_id: str = "conversation-1",
    interrupt_before: str | None = "answer",
    checkpoint_turn_id: UUID | None = None,
) -> tuple[str, UUID, datetime]:
    context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
    async with AsyncConnectionPool(
        database_url,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        checkpointer = AsyncPostgresSaver(cast(Any, pool))
        await checkpointer.setup()
        messages = ConversationMessageRepository(pool)
        conversation = await messages.resolve_conversation(
            context=context,
            conversation_id=conversation_id,
        )
        turn_id = uuid4()
        turn = await messages.create_turn(
            context=context,
            conversation_id=conversation.conversation_id,
            turn_id=turn_id,
            content="resume me",
            idempotency_key=f"turn:{turn_id}:user",
        )
        graph = build_linear_graph(
            checkpointer,
            tenant_id="tenant-a",
            current_turn_id=turn_id,
            artifact_repository=ArtifactRepository(pool),
            message_repository=messages,
            request_context=context,
            history_token_budget=4096,
            answer_actor=_AnswerActor(),
            groundedness_actor=_AssertingGroundednessActor("recovered answer"),
        )
        state: LinearGraphState = {
            "query": "resume me",
            "conversation_id": conversation.conversation_id,
            "turn_id": str(checkpoint_turn_id or turn_id),
            "client_request_id": None,
        }
        config = initial_checkpoint_config(
            thread_id=conversation.thread_id,
            checkpoint_ns="",
        )
        if interrupt_before is None:
            await graph.ainvoke(state, config=config, durability="sync")
        else:
            await graph.ainvoke(
                state,
                config=config,
                interrupt_before=[interrupt_before],
                durability="sync",
            )
        return conversation.thread_id, turn_id, turn.resume_deadline


async def _read_turn(database_url: str, turn_id: UUID) -> TurnRecord:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        return await ConversationMessageRepository(pool).get_turn(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="conversation-1",
            turn_id=turn_id,
        )


async def _read_messages(
    database_url: str,
    *,
    conversation_id: str = "conversation-1",
) -> list[MessageRecord]:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        return await ConversationMessageRepository(pool).list_messages(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id=conversation_id,
        )


async def _expire_turn(database_url: str, turn_id: UUID) -> None:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        async with pool.connection() as connection:
            await connection.execute(
                """
                UPDATE langgraph_v2.messages
                SET resume_deadline = clock_timestamp() - interval '1 second'
                WHERE tenant_id = %s AND conversation_id = %s
                  AND turn_id = %s AND role = 'user'
                """,
                ("tenant-a", "conversation-1", turn_id),
            )


async def _create_superseding_turn(database_url: str) -> None:
    context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        turn_id = uuid4()
        await ConversationMessageRepository(pool).create_turn(
            context=context,
            conversation_id="conversation-1",
            turn_id=turn_id,
            content="newer question",
            idempotency_key=f"turn:{turn_id}:user",
        )


async def _advance_same_turn_checkpoint(
    database_url: str,
    *,
    turn_id: UUID,
) -> str:
    context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
    async with AsyncConnectionPool(
        database_url,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        checkpointer = AsyncPostgresSaver(cast(Any, pool))
        await checkpointer.setup()
        messages = ConversationMessageRepository(pool)
        conversation = await messages.resolve_conversation(
            context=context,
            conversation_id="conversation-1",
        )
        await messages.create_turn(
            context=context,
            conversation_id=conversation.conversation_id,
            turn_id=turn_id,
            content="resume me",
            idempotency_key=f"turn:{turn_id}:user",
        )
        graph = build_linear_graph(
            checkpointer,
            tenant_id="tenant-a",
            current_turn_id=turn_id,
            artifact_repository=ArtifactRepository(pool),
            message_repository=messages,
            request_context=context,
            history_token_budget=4096,
            answer_actor=_AnswerActor(),
        )
        await graph.ainvoke(
            {
                "query": "resume me",
                "conversation_id": conversation.conversation_id,
                "turn_id": str(turn_id),
                "client_request_id": None,
            },
            config=initial_checkpoint_config(
                thread_id=conversation.thread_id,
                checkpoint_ns="",
            ),
            durability="sync",
        )
        checkpoint = await checkpointer.aget_tuple(
            initial_checkpoint_config(
                thread_id=conversation.thread_id,
                checkpoint_ns="",
            )
        )
        assert checkpoint is not None
        return str(checkpoint.checkpoint["id"])


async def _latest_checkpoint_id(database_url: str, thread_id: str) -> str:
    async with AsyncConnectionPool(
        database_url,
        min_size=1,
        max_size=2,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        saver = AsyncPostgresSaver(cast(Any, pool))
        await saver.setup()
        checkpoint = await saver.aget_tuple(
            initial_checkpoint_config(thread_id=thread_id, checkpoint_ns="")
        )
        assert checkpoint is not None
        return str(checkpoint.checkpoint["id"])


class _BlockingResumeGraph:
    def __init__(self, *, events: list[dict[str, object]]) -> None:
        self.events = events
        self.started = threading.Event()
        self.release = threading.Event()
        self.seen_config: RunnableConfig | None = None

    def astream(
        self,
        graph_input: object | None,
        /,
        *,
        config: RunnableConfig | None = None,
        stream_mode: list[str] | str | None = None,
        durability: str | None = None,
    ) -> AsyncIterator[object]:
        del graph_input, stream_mode, durability
        self.seen_config = config

        async def iterator() -> AsyncIterator[object]:
            self.started.set()
            await asyncio.to_thread(self.release.wait)
            for event in self.events:
                live_event = {
                    key: value for key, value in event.items() if key != "event_key"
                }
                if live_event.get("type") == "done":
                    live_event["checkpoint_terminal"] = True
                yield ("custom", live_event)
            yield ("updates", {"finalization": {"final_response": {}}})

        return iterator()


@pytest.mark.parametrize(
    "interrupt_before",
    [
        "pre_moderation",
        "question_refinement",
        "retrieval",
        "reranking",
        "answer",
        "groundedness",
        "post_moderation",
        "finalization",
    ],
)
def test_thread_resume_recovers_any_incomplete_node_from_fresh_app(
    langgraph_v2_migrated_database_url: str,
    interrupt_before: str,
) -> None:
    thread_id, turn_id, resume_deadline = asyncio.run(
        _seed_pre_answer_checkpoint(
            langgraph_v2_migrated_database_url,
            interrupt_before=interrupt_before,
        )
    )
    answer_actor = _AnswerActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=answer_actor,
    )
    app.state.langgraph_v2_groundedness_actor = _AssertingGroundednessActor(
        "recovered answer"
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )
        repeated = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )

    delivered = parse_sse(response.text)
    contract = json.loads(UAT_CONTRACT_PATH.read_text())
    actual_request = {
        "thread_id": thread_id,
        "expectedTurnId": str(turn_id),
    }
    assert set(actual_request) == set(contract["resume_request_fields"])
    assert isinstance(actual_request["thread_id"], str)
    UUID(actual_request["expectedTurnId"])
    expected_headers = {
        "x-thread-id": thread_id,
        "x-conversation-id": "conversation-1",
        "x-turn-id": str(turn_id),
    }
    assert set(contract["resume_response_headers"]) == set(expected_headers)
    assert {
        header: response.headers[header]
        for header in contract["resume_response_headers"]
    } == expected_headers
    assert response.status_code == 200, response.text
    assert repeated.status_code == 409
    assert "x-run-id" not in response.headers
    assert answer_actor.calls == (
        0
        if interrupt_before in {"groundedness", "post_moderation", "finalization"}
        else 1
    )
    expected_token_count = (
        0
        if interrupt_before in {"groundedness", "post_moderation", "finalization"}
        else 2
    )
    assert sum(event["type"] == "token" for event in delivered) == expected_token_count
    assert delivered[-1]["type"] == "done"
    assert delivered[-1]["data"]["answer"] == "recovered answer"

    turn_after_resume = asyncio.run(
        _read_turn(langgraph_v2_migrated_database_url, turn_id)
    )
    messages = asyncio.run(_read_messages(langgraph_v2_migrated_database_url))
    assert turn_after_resume.resume_deadline == resume_deadline
    assert [
        (message.role, message.content)
        for message in messages
        if message.turn_id == turn_id
    ] == [("user", "resume me"), ("assistant", "recovered answer")]


def test_thread_resume_reexecutes_interrupted_refinement_once_from_checkpoint(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, turn_id, interrupted_actor, interrupted_pool = asyncio.run(
        _interrupt_query_during_refinement(langgraph_v2_migrated_database_url)
    )
    resumed_actor = _SuccessfulRefinementActor()
    answer_actor = _StreamingAnswerActor(
        answer="recovered answer",
        stream_tokens=["recovered ", "answer"],
    )
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        refinement_actor=resumed_actor,
        answer_actor=answer_actor,
    )

    with TestClient(app) as client:
        assert app.state.langgraph_v2_postgres_pool is not interrupted_pool
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
            params={"expectedTurnId": str(turn_id)},
        )

    delivered = parse_sse(response.text)
    messages = asyncio.run(
        _read_messages(
            langgraph_v2_migrated_database_url,
            conversation_id="conversation-task41-refinement-resume",
        )
    )
    assert response.status_code == 200
    assert interrupted_actor.calls == resumed_actor.calls == 1
    assert answer_actor.calls == 1
    assert [event["data"] for event in delivered if event["type"] == "token"] == [
        "recovered ",
        "answer",
    ]
    assert sum(event["type"] == "done" for event in delivered) == 1
    assert delivered[-1]["data"]["answer"] == "recovered answer"
    assert "llm:refine_question" in {event.get("step") for event in delivered}
    assert [
        (message.role, message.content)
        for message in messages
        if message.turn_id == turn_id
    ] == [("user", "hello"), ("assistant", "recovered answer")]


def test_thread_resume_returns_404_for_missing_or_unauthorized_thread(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(langgraph_v2_migrated_database_url)
    )
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=_AnswerActor(),
    )

    with TestClient(app) as client:
        cross_subject = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-b"},
            params={"expectedTurnId": str(turn_id)},
        )
        cross_tenant = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-b", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )
        missing = client.post(
            "/v2/threads/missing-thread/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )

    assert cross_subject.status_code == 404
    assert cross_tenant.status_code == 404
    assert missing.status_code == 404


def test_thread_resume_returns_404_for_missing_expected_turn(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, _turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(langgraph_v2_migrated_database_url)
    )
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=_AnswerActor(),
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(uuid4())},
        )

    assert response.status_code == 404


def test_thread_resume_returns_404_for_expected_turn_from_other_conversation(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, _turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(langgraph_v2_migrated_database_url)
    )
    _, foreign_turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(
            langgraph_v2_migrated_database_url,
            conversation_id="conversation-2",
        )
    )
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=_AnswerActor(),
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(foreign_turn_id)},
        )

    assert response.status_code == 404


def test_thread_resume_returns_410_for_expired_turn_without_graph_execution(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(langgraph_v2_migrated_database_url)
    )
    asyncio.run(_expire_turn(langgraph_v2_migrated_database_url, turn_id))
    answer_actor = _AnswerActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=answer_actor,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )

    assert response.status_code == 410
    assert answer_actor.calls == 0


@pytest.mark.parametrize("checkpoint_state", ["complete", "wrong_turn", "superseded"])
def test_thread_resume_rejects_non_recoverable_checkpoint_without_graph_execution(
    langgraph_v2_migrated_database_url: str,
    checkpoint_state: str,
) -> None:
    if checkpoint_state == "complete":
        thread_id, turn_id, _ = asyncio.run(
            _seed_pre_answer_checkpoint(
                langgraph_v2_migrated_database_url,
                interrupt_before=None,
            )
        )
    elif checkpoint_state == "wrong_turn":
        thread_id, turn_id, _ = asyncio.run(
            _seed_pre_answer_checkpoint(
                langgraph_v2_migrated_database_url,
                checkpoint_turn_id=uuid4(),
            )
        )
    else:
        thread_id, turn_id, _ = asyncio.run(
            _seed_pre_answer_checkpoint(langgraph_v2_migrated_database_url)
        )
        asyncio.run(_create_superseding_turn(langgraph_v2_migrated_database_url))

    answer_actor = _AnswerActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=answer_actor,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )

    assert response.status_code == 409
    assert answer_actor.calls == 0


def test_thread_resume_rechecks_supersession_when_execution_starts(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id, turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(langgraph_v2_migrated_database_url)
    )
    original_get_latest_turn = ConversationMessageRepository.get_latest_turn
    calls = 0

    async def supersede_before_execution_check(
        repository: ConversationMessageRepository,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
    ) -> TurnRecord:
        nonlocal calls
        calls += 1
        if calls == 2:
            await _create_superseding_turn(langgraph_v2_migrated_database_url)
        return await original_get_latest_turn(
            repository,
            context=context,
            conversation_id=conversation_id,
        )

    monkeypatch.setattr(
        ConversationMessageRepository,
        "get_latest_turn",
        supersede_before_execution_check,
    )
    answer_actor = _AnswerActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=answer_actor,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )

    delivered = parse_sse(response.text)
    assert response.status_code == 200
    assert delivered[-1]["type"] == "error"
    assert "has been superseded" in delivered[-1]["data"]
    assert answer_actor.calls == 0


def test_thread_resume_pins_the_authorized_checkpoint_before_execution(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(
            langgraph_v2_migrated_database_url,
        )
    )
    original_checkpoint_id = asyncio.run(
        _latest_checkpoint_id(langgraph_v2_migrated_database_url, thread_id)
    )
    graph = _BlockingResumeGraph(
        events=[
            {
                "event_key": "phase:answer:token:1",
                "type": "token",
                "step": "answer",
                "data": "recovered ",
            },
            {
                "event_key": "recovery:completed:2",
                "type": "done",
                "data": {"answer": "recovered answer"},
            },
        ]
    )
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        graph=graph,
        thread_resume_enabled=True,
        answer_actor=_AnswerActor(),
    )

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            response_future = executor.submit(
                client.post,
                f"/v2/threads/{thread_id}/resume/stream",
                headers={
                    "X-Application-Id": "tenant-a",
                    "X-Subject-Id": "subject-a",
                },
                params={"expectedTurnId": str(turn_id)},
            )
            assert graph.started.wait(timeout=5)
            advanced_checkpoint_id = asyncio.run(
                _advance_same_turn_checkpoint(
                    langgraph_v2_migrated_database_url, turn_id=turn_id
                )
            )
            graph.release.set()
            response = response_future.result(timeout=5)

    delivered = parse_sse(response.text)
    assert response.status_code == 200
    assert response.headers["x-turn-id"] == str(turn_id)
    assert graph.seen_config is not None
    assert "configurable" in graph.seen_config
    configurable = graph.seen_config["configurable"]
    assert configurable["thread_id"] == thread_id
    assert configurable["checkpoint_ns"] == ""
    assert configurable["checkpoint_id"] == original_checkpoint_id
    assert configurable["checkpoint_id"] != advanced_checkpoint_id
    assert [event["type"] for event in delivered] == ["token", "done"]


def test_thread_resume_replays_full_answer_from_interrupted_query_and_preserves_advisory_inputs(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, turn_id, resume_deadline, interrupted_pool = asyncio.run(
        _interrupt_query_after_answer_token(
            langgraph_v2_migrated_database_url,
        )
    )
    messages_before_resume = asyncio.run(
        _read_messages(langgraph_v2_migrated_database_url)
    )
    assert [
        (message.role, message.content)
        for message in messages_before_resume
        if message.turn_id == turn_id
    ] == [("user", "hello")]
    answer_text = "replacement answer [1]"
    answer_actor = _StreamingAnswerActor(
        answer=answer_text,
        stream_tokens=["replacement ", "answer [1]"],
    )
    groundedness_actor = _AssertingGroundednessActor(answer_text)
    moderation_provider = _AssertingModerationProvider(answer_text)
    audit = MockOutputAssessmentAudit()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        moderation_provider=moderation_provider,
        answer_actor=answer_actor,
    )
    app.state.langgraph_v2_groundedness_actor = groundedness_actor
    app.state.langgraph_v2_output_assessment_audit = audit

    with TestClient(app) as client:
        assert app.state.langgraph_v2_postgres_pool is not interrupted_pool
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )

    delivered = parse_sse(response.text)
    assert response.status_code == 200
    assert response.headers["x-thread-id"] == thread_id
    assert response.headers["x-turn-id"] == str(turn_id)
    assert answer_actor.calls == 1
    assert groundedness_actor.calls == 1
    assert moderation_provider.calls == 1
    assert moderation_provider.seen_texts == [answer_text]
    assert [event["data"] for event in delivered if event["type"] == "token"] == [
        "replacement ",
        "answer [1]",
    ]
    assert delivered[-1]["type"] == "done"
    assert delivered[-1]["data"]["answer"] == answer_text

    turn_after_resume = asyncio.run(
        _read_turn(langgraph_v2_migrated_database_url, turn_id)
    )
    messages = asyncio.run(_read_messages(langgraph_v2_migrated_database_url))
    assert turn_after_resume.resume_deadline == resume_deadline
    assert [
        (message.role, message.content)
        for message in messages
        if message.turn_id == turn_id
    ] == [("user", "hello"), ("assistant", answer_text)]
    assert {record.assessment_type for record in audit.records} == {
        "groundedness",
        "post_moderation",
    }


def test_second_interruption_can_resume_before_original_deadline(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, turn_id, resume_deadline, _ = asyncio.run(
        _interrupt_query_after_answer_token(
            langgraph_v2_migrated_database_url,
        )
    )
    blocking_answer_actor = _StreamingAnswerActor(
        answer="replacement answer [1]",
        stream_tokens=["replacement "],
        block_after_first_token=True,
    )
    blocking_app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=blocking_answer_actor,
    )

    async def interrupt_resumed_stream() -> None:
        async with blocking_app.router.lifespan_context(blocking_app):
            response = await _thread_resume_endpoint(blocking_app)(
                thread_id,
                stream_request(blocking_app),
                request_context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                expected_turn_id=turn_id,
            )
            token_frame = await close_stream_after_first_token(
                response.body_iterator,
                blocking_answer_actor.blocked_on_followup_read,
            )
            assert token_frame["data"] == "replacement "

    asyncio.run(interrupt_resumed_stream())

    turn_after_second_interrupt = asyncio.run(
        _read_turn(langgraph_v2_migrated_database_url, turn_id)
    )
    messages_after_interrupt = asyncio.run(
        _read_messages(langgraph_v2_migrated_database_url)
    )
    assert turn_after_second_interrupt.resume_deadline == resume_deadline
    assert [
        (message.role, message.content)
        for message in messages_after_interrupt
        if message.turn_id == turn_id
    ] == [("user", "hello")]

    complete_answer_actor = _StreamingAnswerActor(
        answer="replacement answer [1]",
        stream_tokens=["replacement ", "answer [1]"],
    )
    complete_app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=complete_answer_actor,
    )

    with TestClient(complete_app) as client:
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )

    delivered = parse_sse(response.text)
    assert response.status_code == 200
    assert complete_answer_actor.calls == 1
    assert [event["data"] for event in delivered if event["type"] == "token"] == [
        "replacement ",
        "answer [1]",
    ]
    assert delivered[-1]["data"]["answer"] == "replacement answer [1]"


def test_second_interruption_returns_410_after_original_deadline_expires(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, turn_id, _, _ = asyncio.run(
        _interrupt_query_after_answer_token(
            langgraph_v2_migrated_database_url,
        )
    )
    blocking_answer_actor = _StreamingAnswerActor(
        answer="replacement answer [1]",
        stream_tokens=["replacement "],
        block_after_first_token=True,
    )
    blocking_app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=blocking_answer_actor,
    )

    async def interrupt_resumed_stream() -> None:
        async with blocking_app.router.lifespan_context(blocking_app):
            response = await _thread_resume_endpoint(blocking_app)(
                thread_id,
                stream_request(blocking_app),
                request_context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                expected_turn_id=turn_id,
            )
            token_frame = await close_stream_after_first_token(
                response.body_iterator,
                blocking_answer_actor.blocked_on_followup_read,
            )
            assert token_frame["data"] == "replacement "

    asyncio.run(interrupt_resumed_stream())
    asyncio.run(_expire_turn(langgraph_v2_migrated_database_url, turn_id))

    expired_answer_actor = _StreamingAnswerActor(
        answer="replacement answer [1]",
        stream_tokens=["replacement ", "answer [1]"],
    )
    expired_app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        thread_resume_enabled=True,
        answer_actor=expired_answer_actor,
    )

    with TestClient(expired_app) as client:
        response = client.post(
            f"/v2/threads/{thread_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            params={"expectedTurnId": str(turn_id)},
        )

    assert response.status_code == 410
    assert expired_answer_actor.calls == 0
