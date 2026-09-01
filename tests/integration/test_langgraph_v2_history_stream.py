from __future__ import annotations

import asyncio

# Public-stream coverage complements the pure selector unit tests.
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult, AnswerStreamChunk
from app.langgraph_v2.history import ConversationExchange
from app.langgraph_v2.question_refinement import (
    QuestionRefinementResult,
    V2ResolvedQuery,
)
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
    seed_subject_conversation,
)

_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "langgraph_v2"
    / "v1_session_continuity_wire.json"
)


class _RefinementActor:
    def __init__(self) -> None:
        self.histories: list[list[ConversationExchange]] = []

    async def refine(
        self,
        query: str,
        history: Sequence[ConversationExchange],
    ) -> QuestionRefinementResult:
        self.histories.append(list(history))
        return QuestionRefinementResult(
            resolved_query=V2ResolvedQuery(original_query=query, standalone_query=query)
        )


class _AnswerActor:
    def __init__(self) -> None:
        self.histories: list[list[ConversationExchange]] = []

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        del documents
        self.histories.append(list(history))
        return AnswerResult(answer=f"answer for {query}")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
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
            await seed_subject_conversation(pool)

    asyncio.run(seed())
    refinement = _RefinementActor()
    answer = _AnswerActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=refinement,
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=answer,
    )

    with TestClient(app) as client:
        first = client.post(
            "/v2/query/stream",
            json={"query": "first question", "sessionId": "00000000-0000-0000-0000-000000000001"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        second = client.post(
            "/v2/query/stream",
            json={"query": "follow up", "sessionId": "00000000-0000-0000-0000-000000000001"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    expected_history = [
        ConversationExchange(user="first question", assistant="answer for first question")
    ]
    fixture = json.loads(_FIXTURE_PATH.read_text())
    assert fixture["requests"] == [
        {"query": "first question", "sessionId": "00000000-0000-0000-0000-000000000001"},
        {"query": "follow up", "sessionId": "00000000-0000-0000-0000-000000000001"},
    ]
    assert first.status_code == second.status_code == 200
    assert refinement.histories == [[], expected_history]
    assert answer.histories == [[], expected_history]
    second_done = parse_sse(second.text)[-1]
    assert {"type": second_done["type"], "data": second_done["data"]} == fixture[
        "second_done"
    ]
