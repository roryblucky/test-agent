from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult, AnswerStreamChunk
from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import initial_checkpoint_config
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    MessageRecord,
    TurnRecord,
)
from app.langgraph_v2.graph import TracerState, build_tracer_graph
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.run_events import RunEventRepository
from app.models.domain import Document
from tests.integration.test_langgraph_v2_tracer import (
    parse_sse,
    persistent_tracer_app,
)


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


async def _seed_pre_answer_checkpoint(
    database_url: str,
    *,
    conversation_id: str = "conversation-1",
    interrupt_before: str | None = "answer",
    checkpoint_turn_id: UUID | None = None,
    drop_origin_run_mapping: bool = False,
) -> tuple[str, UUID, datetime]:
    context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
    async with AsyncConnectionPool(
        database_url,
        min_size=1,
        max_size=3,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        messages = ConversationMessageRepository(pool)
        conversation = await messages.resolve_conversation(
            context=context,
            conversation_id=conversation_id,
        )
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id=conversation.conversation_id,
            owner_instance_id="seed-instance",
        )
        turn_id = uuid4()
        turn = await messages.create_turn(
            context=context,
            conversation_id=conversation.conversation_id,
            run_id=run.run_id,
            turn_id=turn_id,
            content="resume me",
            idempotency_key=f"turn:{turn_id}:user",
        )
        phase_context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            artifact_repository=ArtifactRepository(pool),
            message_repository=messages,
            request_context=context,
            history_token_budget=4096,
            current_turn_id=turn_id,
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            checkpointer,
            phase_context=phase_context,
            answer_actor=_AnswerActor(),
        )
        state: TracerState = {
            "query": "resume me",
            "conversation_id": conversation.conversation_id,
            "turn_id": str(checkpoint_turn_id or turn_id),
            "client_request_id": None,
            "events": [],
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
        if drop_origin_run_mapping:
            async with pool.connection() as connection:
                await connection.execute(
                    """
                    DELETE FROM langgraph_v2.runs
                    WHERE tenant_id = %s AND run_id = %s
                    """,
                    ("tenant-a", run.run_id),
                )
        return conversation.thread_id, turn_id, turn.resume_deadline


async def _read_turn(database_url: str, turn_id: UUID) -> TurnRecord:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        return await ConversationMessageRepository(pool).get_turn(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="conversation-1",
            turn_id=turn_id,
        )


async def _read_messages(database_url: str) -> list[MessageRecord]:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        return await ConversationMessageRepository(pool).list_messages(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="conversation-1",
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
        run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="superseding-instance",
        )
        turn_id = uuid4()
        await ConversationMessageRepository(pool).create_turn(
            context=context,
            conversation_id="conversation-1",
            run_id=run.run_id,
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
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        messages = ConversationMessageRepository(pool)
        conversation = await messages.resolve_conversation(
            context=context,
            conversation_id="conversation-1",
        )
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id=conversation.conversation_id,
            owner_instance_id="advance-instance",
        )
        await messages.create_turn(
            context=context,
            conversation_id=conversation.conversation_id,
            run_id=run.run_id,
            turn_id=turn_id,
            content="resume me",
            idempotency_key=f"turn:{turn_id}:user",
        )
        phase_context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            artifact_repository=ArtifactRepository(pool),
            message_repository=messages,
            request_context=context,
            history_token_budget=4096,
            current_turn_id=turn_id,
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            checkpointer,
            phase_context=phase_context,
            answer_actor=_AnswerActor(),
        )
        await graph.ainvoke(
            {
                "query": "resume me",
                "conversation_id": conversation.conversation_id,
                "turn_id": str(turn_id),
                "client_request_id": None,
                "events": [],
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
        saver = AsyncPostgresSaver(pool)
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
        self.seen_config: dict[str, object] | None = None

    def astream(
        self,
        graph_input: object | None,
        *,
        config: dict[str, object] | None = None,
        stream_mode: list[str] | str | None = None,
        durability: str | None = None,
    ) -> AsyncIterator[object]:
        del graph_input, stream_mode, durability
        self.seen_config = config

        async def iterator() -> AsyncIterator[object]:
            self.started.set()
            await asyncio.to_thread(self.release.wait)
            yield ("updates", {"events": self.events})

        return iterator()


@pytest.mark.parametrize(
    "interrupt_before",
    ["pre_moderation", "question_refinement", "retrieval", "reranking", "answer"],
)
def test_thread_resume_recovers_pre_answer_checkpoint_from_fresh_app(
    langgraph_v2_migrated_database_url: str,
    interrupt_before: str,
) -> None:
    thread_id, turn_id, resume_deadline = asyncio.run(
        _seed_pre_answer_checkpoint(
            langgraph_v2_migrated_database_url,
            interrupt_before=interrupt_before,
            drop_origin_run_mapping=True,
        )
    )
    answer_actor = _AnswerActor()
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        resume_enabled=True,
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
    assert response.headers["x-thread-id"] == thread_id
    assert response.headers["x-conversation-id"] == "conversation-1"
    assert response.headers["x-turn-id"] == str(turn_id)
    assert "x-run-id" not in response.headers
    assert answer_actor.calls == 1
    assert [event["type"] for event in delivered if event["type"] == "token"] == [
        "token",
        "token",
    ]
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


def test_thread_resume_returns_404_for_missing_or_unauthorized_thread(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(langgraph_v2_migrated_database_url)
    )
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        resume_enabled=True,
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
    thread_id, turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(langgraph_v2_migrated_database_url)
    )
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        resume_enabled=True,
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
    thread_id, turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(langgraph_v2_migrated_database_url)
    )
    _, foreign_turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(
            langgraph_v2_migrated_database_url,
            conversation_id="conversation-2",
        )
    )
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        resume_enabled=True,
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
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        resume_enabled=True,
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
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        resume_enabled=True,
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


def test_thread_resume_pins_the_authorized_checkpoint_before_execution(
    langgraph_v2_migrated_database_url: str,
) -> None:
    thread_id, turn_id, _ = asyncio.run(
        _seed_pre_answer_checkpoint(
            langgraph_v2_migrated_database_url,
            drop_origin_run_mapping=True,
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
            }
        ]
    )
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        graph=graph,
        resume_enabled=True,
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
    configurable = graph.seen_config["configurable"]
    assert configurable["thread_id"] == thread_id
    assert configurable["checkpoint_ns"] == ""
    assert configurable["checkpoint_id"] == original_checkpoint_id
    assert configurable["checkpoint_id"] != advanced_checkpoint_id
    assert [event["type"] for event in delivered] == ["token", "done"]
