from __future__ import annotations

# Public-stream coverage complements the pure selector unit tests.
from fastapi.testclient import TestClient

from app.langgraph_v2.answer import AnswerResult
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from app.models.workflow import ResolvedQuery
from tests.integration.test_langgraph_v2_tracer import (
    parse_sse,
    persistent_tracer_app,
)


class _RefinementActor:
    def __init__(self) -> None:
        self.histories: list[list[ConversationTurn]] = []

    async def refine(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> ResolvedQuery:
        self.histories.append(history)
        return ResolvedQuery(original_query=query, standalone_query=query)


class _AnswerActor:
    def __init__(self) -> None:
        self.histories: list[list[ConversationTurn]] = []

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: list[ConversationTurn],
    ) -> AnswerResult:
        del documents
        self.histories.append(history)
        return AnswerResult(answer=f"answer for {query}")


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
            headers={"X-Application-Id": "tenant-a"},
        )
        second = client.post(
            "/v2/query/stream",
            json={"query": "follow up", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a"},
        )

    expected_history = [
        ConversationTurn(user="first question", assistant="answer for first question")
    ]
    assert first.status_code == second.status_code == 200
    assert refinement.histories == [[], expected_history]
    assert answer.histories == [[], expected_history]
    assert parse_sse(second.text)[-1]["data"]["answer"] == "answer for follow up"
