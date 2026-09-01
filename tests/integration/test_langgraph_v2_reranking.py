from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.graph import build_linear_graph
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.langgraph_v2_request_support import seed_request_scope
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)


@pytest.mark.asyncio
async def test_ranker_receives_original_documents_and_returns_reordered_evidence(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class Retriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(
                documents=[
                    Document(id="d1", content=f"{query}-1"),
                    Document(id="d2", content=f"{query}-2"),
                ],
                raw_payload={"query": query},
            )

    class Ranker:
        received: list[str] = []

        async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
            assert query == "hello"
            self.received = [document.id for document in documents]
            return RerankingResult(documents=[documents[1], documents[0]])

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        ranker = Ranker()
        scope = await seed_request_scope(pool)
        result = await build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            retriever=Retriever(),
            ranker=ranker,
        ).ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "request_id": "request-1",
            }
        )

        assert ranker.received == ["d1", "d2"]
        assert "evidence" in result
        assert "ranked_evidence" in result
        evidence = result["evidence"]
        ranked_evidence = result["ranked_evidence"]
        assert [item.evidence_id for item in ranked_evidence] == [
            evidence[1].evidence_id,
            evidence[0].evidence_id,
        ]


@pytest.mark.asyncio
async def test_ranker_rejects_duplicate_document_multiplicity(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class DuplicateRetriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(
                documents=[
                    Document(id="a", content=query),
                    Document(id="a", content=f"{query}-second"),
                    Document(id="b", content=query),
                ]
            )

    class InvalidRanker:
        async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
            del query
            return RerankingResult(documents=[documents[0], documents[2], documents[2]])

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        scope = await seed_request_scope(pool)
        result = await build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            retriever=DuplicateRetriever(),
            ranker=InvalidRanker(),
        ).ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "request_id": "request-1",
            }
        )

    assert "halted" in result
    assert result["halted"] is True
    assert "reranking_error" in result
    reranking_error = result["reranking_error"]
    assert isinstance(reranking_error, str)
    assert "every retrieved document exactly once" in reranking_error


@pytest.mark.asyncio
async def test_ranker_preserves_order_for_distinct_documents_with_same_id(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class DuplicateRetriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(
                documents=[
                    Document(id="a", content=f"{query}-first"),
                    Document(id="a", content=f"{query}-second"),
                    Document(id="b", content=query),
                ]
            )

    class ReverseRanker:
        async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
            del query
            return RerankingResult(documents=list(reversed(documents)))

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        scope = await seed_request_scope(pool)
        result = await build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            retriever=DuplicateRetriever(),
            ranker=ReverseRanker(),
        ).ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "request_id": "request-1",
            }
        )

    assert "halted" in result
    assert result["halted"] is False
    assert "evidence" in result
    assert "ranked_evidence" in result
    evidence = result["evidence"]
    ranked_evidence = result["ranked_evidence"]
    assert [item.evidence_id for item in ranked_evidence] == [
        evidence[2].evidence_id,
        evidence[1].evidence_id,
        evidence[0].evidence_id,
    ]


def test_failed_ranker_terminates_public_stream(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class FailingRanker:
        async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
            del query, documents
            raise RuntimeError("ranker unavailable")

    app = persistent_linear_app(
        langgraph_v2_migrated_database_url, ranker=FailingRanker()
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
    events = parse_sse(response.text)
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert events[-1]["data"] == "ranker unavailable"
    assert all(event.get("step") != "finalization" for event in events)
