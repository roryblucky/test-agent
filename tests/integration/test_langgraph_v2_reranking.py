from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.graph import LinearGraphState, build_linear_graph
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.langgraph_v2_artifact_support import seed_artifact_scope
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)


@pytest.mark.asyncio
async def test_ranker_receives_original_documents_and_persists_reordered_refs(
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
        scope = await seed_artifact_scope(pool)
        result = await build_linear_graph(
            tenant_id="tenant-a",
            current_turn_id=scope.turn_id,
            artifact_repository=ArtifactRepository(pool),
            request_context=scope.context,
            retriever=Retriever(),
            ranker=ranker,
        ).ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "client_request_id": None,
            }
        )

        assert ranker.received == ["d1", "d2"]
        assert "artifact_refs" in result
        assert "ranked_refs" in result
        assert [ref["artifact_id"] for ref in result["ranked_refs"]] == [
            result["artifact_refs"][1]["artifact_id"],
            result["artifact_refs"][0]["artifact_id"],
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
        scope = await seed_artifact_scope(pool)
        result = await build_linear_graph(
            tenant_id="tenant-a",
            current_turn_id=scope.turn_id,
            artifact_repository=ArtifactRepository(pool),
            request_context=scope.context,
            retriever=DuplicateRetriever(),
            ranker=InvalidRanker(),
        ).ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "client_request_id": None,
            }
        )

    assert "halted" in result
    assert result["halted"] is True
    assert "reranking_error" in result
    assert "every retrieved document exactly once" in result["reranking_error"]


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
        scope = await seed_artifact_scope(pool)
        result = await build_linear_graph(
            tenant_id="tenant-a",
            current_turn_id=scope.turn_id,
            artifact_repository=ArtifactRepository(pool),
            request_context=scope.context,
            retriever=DuplicateRetriever(),
            ranker=ReverseRanker(),
        ).ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "client_request_id": None,
            }
        )

    assert "halted" in result
    assert result["halted"] is False
    assert "artifact_refs" in result
    assert "ranked_refs" in result
    assert [ref["artifact_id"] for ref in result["ranked_refs"]] == [
        result["artifact_refs"][2]["artifact_id"],
        result["artifact_refs"][1]["artifact_id"],
        result["artifact_refs"][0]["artifact_id"],
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


@pytest.mark.asyncio
async def test_interrupted_reranking_repeats_provider_with_stable_artifact_refs(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class CountingRanker:
        calls = 0

        async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
            self.calls += 1
            del query
            if self.calls == 1:
                raise asyncio.CancelledError
            return RerankingResult(documents=list(reversed(documents)))

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        ranker = CountingRanker()
        scope = await seed_artifact_scope(pool)
        graph = build_linear_graph(
            checkpointer=MemorySaver(),
            tenant_id="tenant-a",
            current_turn_id=scope.turn_id,
            artifact_repository=ArtifactRepository(pool),
            request_context=scope.context,
            ranker=ranker,
        )
        state: LinearGraphState = {
            "query": "hello",
            "conversation_id": "c1",
            "client_request_id": None,
        }
        config: RunnableConfig = {"configurable": {"thread_id": "reranking-recovery"}}
        with pytest.raises(asyncio.CancelledError):
            await graph.ainvoke(state, config)
        checkpoint = await graph.aget_state(config)
        assert checkpoint.next == ("reranking",)
        recovered = await graph.ainvoke(None, config)

        assert ranker.calls == 2
        assert "artifact_refs" in recovered
        assert "ranked_refs" in recovered
        assert len(recovered["ranked_refs"]) == 1
        assert recovered["ranked_refs"] == [recovered["artifact_refs"][0]]
