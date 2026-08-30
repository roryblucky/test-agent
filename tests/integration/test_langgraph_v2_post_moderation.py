from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult, AnswerStreamChunk
from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.graph import LinearGraphState, build_linear_graph
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.output_assessments import MockOutputAssessmentAudit
from app.langgraph_v2.pre_moderation import ModerationDecision
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.langgraph_v2_artifact_support import seed_artifact_scope
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
    seed_subject_conversation,
    stream_request,
    v2_stream_endpoint,
)


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id="d1", content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        return RerankingResult(documents=documents)


class _Answer:
    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        del history
        del query, documents
        return AnswerResult(answer="generated answer")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationTurn]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _SafeModeration:
    calls = 0

    async def check(self, text: str) -> ModerationDecision:
        self.calls += 1
        assert text in {"hello", "generated answer"}
        return ModerationDecision(is_flagged=False)


class _FlaggingModeration:
    calls = 0

    async def check(self, text: str) -> ModerationDecision:
        self.calls += 1
        assert text in {"hello", "generated answer"}
        return ModerationDecision(
            is_flagged=text == "generated answer", reason="unsafe output"
        )


class _FailingPostModeration:
    async def check(self, text: str) -> ModerationDecision:
        if text == "hello":
            return ModerationDecision(is_flagged=False)
        assert text == "generated answer"
        raise RuntimeError("post evaluator unavailable")


def _state() -> LinearGraphState:
    return {
        "query": "hello",
        "conversation_id": "c1",
        "client_request_id": None,
    }


@pytest.mark.asyncio
async def test_flagged_answer_persists_original_complete_answer_through_http(
    langgraph_v2_migrated_database_url: str,
) -> None:
    context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )
    moderation = _FlaggingModeration()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        retriever=_Retriever(),
        ranker=_Ranker(),
        moderation_provider=moderation,
        answer_actor=_Answer(),
    )

    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=context,
        )
        frames = [frame async for frame in response.body_iterator]
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            messages = await ConversationMessageRepository(pool).list_messages(
                context=context, conversation_id="conversation-1"
            )

    assert moderation.calls == 2
    events = parse_sse("".join(frames))
    token_text = "".join(
        event["data"] for event in events if event["type"] == "token"
    )
    assert token_text == "generated answer"
    assert events[-1]["type"] == "done"
    assert events[-1]["data"]["answer"] == token_text
    assert [(message.role, message.content) for message in messages] == [
        ("user", "hello"),
        ("assistant", "generated answer"),
    ]


@pytest.mark.asyncio
async def test_safe_answer_passes_post_moderation_unchanged(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        messages = ConversationMessageRepository(pool)
        await messages.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="c1",
        )
        turn_id = uuid4()
        await messages.create_turn(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="c1",
            turn_id=turn_id,
            content="hello",
            idempotency_key=f"turn:{turn_id}:user",
        )
        moderation = _SafeModeration()
        scope = await seed_artifact_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_turn_id=scope.turn_id,
            artifact_repository=ArtifactRepository(pool),
            request_context=scope.context,
            moderation_provider=moderation,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
        )
        result = await graph.ainvoke(_state())

    assert moderation.calls == 2
    assert "answer" in result
    assert result["answer"] == "generated answer"
    assert "post_moderation" in result
    post_moderation = result["post_moderation"]
    assert post_moderation is not None
    assert post_moderation["is_flagged"] is False
    assert "final_response" in result
    final_response = result["final_response"]
    assert final_response is not None
    assert final_response.answer == "generated answer"


@pytest.mark.asyncio
async def test_flagged_answer_remains_canonical_through_finalization(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        messages = ConversationMessageRepository(pool)
        await messages.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="c1",
        )
        turn_id = uuid4()
        await messages.create_turn(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="c1",
            turn_id=turn_id,
            content="hello",
            idempotency_key=f"turn:{turn_id}:user",
        )
        audit = MockOutputAssessmentAudit()
        scope = await seed_artifact_scope(pool, turn_id=turn_id)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            artifact_repository=ArtifactRepository(pool),
            current_turn_id=turn_id,
            request_context=scope.context,
            output_assessment_audit=audit,
            moderation_provider=_FlaggingModeration(),
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
        )
        state = _state()
        state["turn_id"] = str(turn_id)
        result = await graph.ainvoke(state)
        persisted_messages = await messages.list_messages(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="c1",
        )

    assert "answer" in result
    assert result["answer"] == "generated answer"
    assert "final_response" in result
    final_response = result["final_response"]
    assert final_response is not None
    assert final_response.answer == "generated answer"
    assert [message.content for message in persisted_messages] == ["hello"]
    assert "post_moderation" in result
    post_moderation = result["post_moderation"]
    assert post_moderation is not None
    assert post_moderation["is_flagged"] is True
    assert len(audit.records) == 1
    assert audit.records[0].tenant_id == "tenant-a"
    assert audit.records[0].conversation_id == "c1"
    assert audit.records[0].turn_id == turn_id
    assert audit.records[0].assessment_type == "post_moderation"


@pytest.mark.asyncio
async def test_post_moderation_failure_is_advisory_and_reaches_finalization(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        scope = await seed_artifact_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_turn_id=scope.turn_id,
            artifact_repository=ArtifactRepository(pool),
            request_context=scope.context,
            moderation_provider=_FailingPostModeration(),
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
        )

        result = await graph.ainvoke(_state())

    assert "answer" in result
    assert result["answer"] == "generated answer"
    assert "post_moderation_error" in result
    assert result["post_moderation_error"] == "post evaluator unavailable"
    assert "final_response" in result
    final_response = result["final_response"]
    assert final_response is not None
    assert final_response.answer == "generated answer"
