from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import (
    AnswerCitation,
    AnswerResult,
    AnswerStreamChunk,
)
from app.langgraph_v2.artifacts import (
    ArtifactNotFound,
    ArtifactRecord,
    ArtifactRepository,
)
from app.langgraph_v2.graph import TracerState, build_tracer_graph
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import RunEventRepository
from app.models.domain import Document
from tests.integration.test_langgraph_v2_tracer import parse_sse, persistent_tracer_app


@pytest.mark.asyncio
async def test_artifact_repository_is_tenant_scoped(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ArtifactRepository(pool)
        artifact = await repository.create(
            tenant_id="tenant-a", artifact_type="document", payload={"id": "d1"}
        )
        assert (
            await repository.get(tenant_id="tenant-a", artifact_id=artifact.artifact_id)
        ).payload == {"id": "d1"}
        with pytest.raises(ArtifactNotFound):
            await repository.get(tenant_id="tenant-b", artifact_id=artifact.artifact_id)


@pytest.mark.asyncio
async def test_retrieval_persists_stable_artifact_refs_across_reexecution(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class CountingRetriever:
        calls = 0

        async def retrieve(self, query: str) -> RetrievalResult:
            self.calls += 1
            return RetrievalResult(
                documents=[Document(id="d1", content=query)],
                raw_payload={"query": query},
            )

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        retriever = CountingRetriever()
        graph = build_tracer_graph(
            phase_context=PhaseExecutionContext(
                repository=PhaseResultRepository(pool),
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id="i1",
                execution_epoch=run.execution_epoch,
                artifact_repository=ArtifactRepository(pool),
            ),
            retriever=retriever,
        )
        state: TracerState = {
            "query": "hello",
            "conversation_id": "c1",
            "client_request_id": None,
            "events": [],
        }
        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)
        assert retriever.calls == 2
        assert "artifact_refs" in first
        assert "artifact_refs" in second
        assert first["artifact_refs"] == second["artifact_refs"]
        assert "artifact_refs" in first
        assert len(first["artifact_refs"]) == 2


def test_artifact_lookup_is_404_across_tenant_boundary(
    langgraph_v2_migrated_database_url: str,
) -> None:
    artifact_id = uuid4()
    with psycopg.connect(
        langgraph_v2_migrated_database_url, autocommit=True
    ) as connection:
        connection.execute(
            "INSERT INTO langgraph_v2.artifacts (tenant_id, artifact_id, artifact_type, payload) VALUES (%s, %s, %s, %s)",
            ("tenant-a", artifact_id, "document", Jsonb({"id": "d1"})),
        )
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)
    with TestClient(app) as client:
        own = client.get(
            f"/v2/artifacts/{artifact_id}", headers={"X-Application-Id": "tenant-a"}
        )
        other = client.get(
            f"/v2/artifacts/{artifact_id}", headers={"X-Application-Id": "tenant-b"}
        )
    assert (own.status_code, other.status_code) == (200, 404)


def test_empty_retrieval_is_explicit_on_public_stream(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class EmptyRetriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(raw_payload={"query": query, "source": "empty"})

    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url, retriever=EmptyRetriever()
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "empty"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
    events = parse_sse(response.text)
    retrieval = next(
        event
        for event in events
        if event.get("step") == "retriever" and event["type"] == "step_completed"
    )
    assert response.status_code == 200
    assert retrieval["data"]["document_count"] == 0
    assert retrieval["data"]["documents"] == []


def test_failed_retrieval_is_error_without_finalization_on_public_stream(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class FailingRetriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            raise RuntimeError(f"provider unavailable for {query}")

    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url, retriever=FailingRetriever()
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "fail"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
    events = parse_sse(response.text)
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert events[-1]["data"] == "provider unavailable for fail"
    assert all(event.get("step") != "finalization" for event in events)


@pytest.mark.asyncio
async def test_retrieval_resume_with_new_run_preserves_artifact_and_citation_identity(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class CountingRetriever:
        calls = 0

        async def retrieve(self, query: str) -> RetrievalResult:
            self.calls += 1
            documents = [
                Document(id="d1", content=f"{query}-one"),
                Document(id="d2", content=f"{query}-two"),
            ]
            if self.calls == 2:
                documents[1] = Document(id="d2", content=f"{query}-two-updated")
                documents.reverse()
            return RetrievalResult(
                documents=documents,
                raw_payload={"query": query, "source": "stable"},
            )

    class InterruptAfterArtifacts:
        def __init__(self, repository: ArtifactRepository) -> None:
            self.repository = repository
            self.interrupted = False
            self.first_document_ids: dict[str, str] = {}

        async def create(self, **kwargs: Any) -> ArtifactRecord:
            artifact = await self.repository.create(**kwargs)
            if kwargs["artifact_type"] == "document" and not self.interrupted:
                self.first_document_ids[str(kwargs["payload"]["id"])] = str(
                    artifact.artifact_id
                )
            if kwargs["artifact_type"] == "retrieval_raw" and not self.interrupted:
                self.interrupted = True
                raise asyncio.CancelledError
            return artifact

        async def get(self, **kwargs: Any) -> ArtifactRecord:
            return await self.repository.get(**kwargs)

    class IdentityRanker:
        async def rank(
            self, query: str, documents: list[Document]
        ) -> RerankingResult:
            del query
            return RerankingResult(documents=documents)

    class CitingAnswer:
        def answer_stream(
            self,
            query: str,
            documents: list[Document],
            history: Sequence[ConversationTurn],
        ) -> AsyncIterator[AnswerStreamChunk]:
            del query, history

            async def stream() -> AsyncIterator[AnswerStreamChunk]:
                assert [document.id for document in documents] == ["d2", "d1"]
                yield AnswerStreamChunk(delta="supported")
                yield AnswerStreamChunk(
                    result=AnswerResult(
                        answer="supported",
                        citations=[AnswerCitation(index=1)],
                    )
                )

            return stream()

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
        artifacts = InterruptAfterArtifacts(ArtifactRepository(pool))
        retriever = CountingRetriever()
        turn_id = uuid4()
        first_context = PhaseExecutionContext(
            repository=repository,
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="i1",
            execution_epoch=run.execution_epoch,
            current_turn_id=turn_id,
            artifact_repository=artifacts,
        )
        checkpointer = MemorySaver()
        first_graph = build_tracer_graph(
            checkpointer=checkpointer,
            phase_context=first_context,
            retriever=retriever,
            ranker=IdentityRanker(),
            answer_actor=CitingAnswer(),
        )
        state: TracerState = {
            "query": "hello",
            "conversation_id": "c1",
            "turn_id": str(turn_id),
            "client_request_id": None,
            "events": [],
        }
        config: RunnableConfig = {
            "configurable": {"thread_id": "retrieval-recovery"}
        }
        with pytest.raises(asyncio.CancelledError):
            await first_graph.ainvoke(state, config)

        resume_run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i2",
        )
        resume_graph = build_tracer_graph(
            checkpointer=checkpointer,
            phase_context=PhaseExecutionContext(
                repository=PhaseResultRepository(pool),
                tenant_id="tenant-a",
                run_id=resume_run.run_id,
                owner_instance_id="i2",
                execution_epoch=resume_run.execution_epoch,
                current_turn_id=turn_id,
                artifact_repository=artifacts,
            ),
            retriever=retriever,
            ranker=IdentityRanker(),
            answer_actor=CitingAnswer(),
        )
        recovered = await resume_graph.ainvoke(None, config)

        assert retriever.calls == 2
        assert "artifact_refs" in recovered
        document_refs = recovered["artifact_refs"][:2]
        assert document_refs[0]["artifact_id"] != artifacts.first_document_ids["d2"]
        assert document_refs[1]["artifact_id"] == artifacts.first_document_ids["d1"]
        assert "citations" in recovered
        citation = recovered["citations"][0]
        assert citation.evidence_id == document_refs[0]["artifact_id"]
        assert citation.metadata == {"artifact_id": document_refs[0]["artifact_id"]}
        original_d2 = await artifacts.get(
            tenant_id="tenant-a",
            artifact_id=UUID(artifacts.first_document_ids["d2"]),
        )
        resumed_d2 = await artifacts.get(
            tenant_id="tenant-a",
            artifact_id=UUID(document_refs[0]["artifact_id"]),
        )
        assert original_d2.payload["content"] == "hello-two"
        assert resumed_d2.payload["content"] == "hello-two-updated"
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT count(*) FROM langgraph_v2.artifacts WHERE tenant_id = %s",
                    ("tenant-a",),
                )
                row = await cursor.fetchone()
                assert row is not None
                assert row[0] == 4
