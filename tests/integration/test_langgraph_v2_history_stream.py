from __future__ import annotations

import asyncio

# Public-stream coverage complements the pure selector unit tests.
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult, AnswerStreamChunk
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.question_refinement import (
    QuestionRefinementResult,
    V2ResolvedQuery,
)
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.test_langgraph_v2_tracer import (
    parse_sse,
    persistent_tracer_app,
)

_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "langgraph_v2"
    / "v1_session_continuity_wire.json"
)


class _RefinementActor:
    def __init__(self) -> None:
        self.histories: list[list[ConversationTurn]] = []

    async def refine(
        self,
        query: str,
        history: Sequence[ConversationTurn],
    ) -> QuestionRefinementResult:
        self.histories.append(list(history))
        return QuestionRefinementResult(
            resolved_query=V2ResolvedQuery(original_query=query, standalone_query=query)
        )


class _AnswerActor:
    def __init__(self) -> None:
        self.histories: list[list[ConversationTurn]] = []

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        del documents
        self.histories.append(list(history))
        return AnswerResult(answer=f"answer for {query}")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationTurn]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id=query, content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        del query
        return RerankingResult(documents=documents)


def test_second_public_stream_receives_one_complete_prior_turn(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async def seed() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            await ConversationMessageRepository(pool).resolve_conversation(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="conversation-1",
            )

    asyncio.run(seed())
    refinement = _RefinementActor()
    answer = _AnswerActor()
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=refinement,
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=answer,
    )

    with TestClient(app) as client:
        first = client.post(
            "/v2/query/stream",
            json={"query": "first question", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        second = client.post(
            "/v2/query/stream",
            json={"query": "follow up", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    expected_history = [
        ConversationTurn(user="first question", assistant="answer for first question")
    ]
    fixture = json.loads(_FIXTURE_PATH.read_text())
    assert fixture["requests"] == [
        {"query": "first question", "sessionId": "conversation-1"},
        {"query": "follow up", "sessionId": "conversation-1"},
    ]
    assert first.status_code == second.status_code == 200
    assert refinement.histories == [[], expected_history]
    assert answer.histories == [[], expected_history]
    second_done = parse_sse(second.text)[-1]
    assert {"type": second_done["type"], "data": second_done["data"]} == fixture[
        "second_done"
    ]
