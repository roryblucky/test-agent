from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.graph import TracerState, build_tracer_graph
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import RunEventRepository
from app.models.domain import Document
from tests.integration.test_langgraph_v2_tracer import parse_sse, persistent_tracer_app


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
        run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        ranker = Ranker()
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="i1",
            execution_epoch=run.execution_epoch,
            artifact_repository=ArtifactRepository(pool),
        )
        result = await build_tracer_graph(
            phase_context=context, retriever=Retriever(), ranker=ranker
        ).ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "client_request_id": None,
                "events": [],
            }
        )

        assert ranker.received == ["d1", "d2"]
        assert "artifact_refs" in result
        assert "ranked_refs" in result
        assert [ref["artifact_id"] for ref in result["ranked_refs"]] == [
            result["artifact_refs"][1]["artifact_id"],
            result["artifact_refs"][0]["artifact_id"],
        ]
        completed = next(
            event
            for event in result["events"]
            if event.get("step") == "reranker" and event["type"] == "step_completed"
        )
        assert completed["data"] == {
            "document_count": 2,
            "selected_ids": ["d2", "d1"],
        }


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
        run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="i1",
            execution_epoch=run.execution_epoch,
            artifact_repository=ArtifactRepository(pool),
        )
        result = await build_tracer_graph(
            phase_context=context,
            retriever=DuplicateRetriever(),
            ranker=InvalidRanker(),
        ).ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "client_request_id": None,
                "events": [],
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
        run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="i1",
            execution_epoch=run.execution_epoch,
            artifact_repository=ArtifactRepository(pool),
        )
        result = await build_tracer_graph(
            phase_context=context,
            retriever=DuplicateRetriever(),
            ranker=ReverseRanker(),
        ).ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "client_request_id": None,
                "events": [],
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

    app = persistent_tracer_app(
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
        run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        repository = PhaseResultRepository(pool)
        ranker = CountingRanker()
        context = PhaseExecutionContext(
            repository=repository,
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="i1",
            execution_epoch=run.execution_epoch,
            artifact_repository=ArtifactRepository(pool),
        )
        graph = build_tracer_graph(
            checkpointer=MemorySaver(), phase_context=context, ranker=ranker
        )
        state: TracerState = {
            "query": "hello",
            "conversation_id": "c1",
            "client_request_id": None,
            "events": [],
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
