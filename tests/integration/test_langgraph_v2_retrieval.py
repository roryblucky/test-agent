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
    ArtifactInvariantConflict,
    ArtifactNotFound,
    ArtifactRecord,
    ArtifactRepository,
    ArtifactScope,
)
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.graph import LinearGraphState, build_linear_graph
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.langgraph_v2_artifact_support import seed_artifact_scope
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)


@pytest.mark.asyncio
async def test_artifact_repository_is_tenant_scoped(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ArtifactRepository(pool)
        scope = await seed_artifact_scope(pool)
        artifact = await repository.create(
            scope=scope, artifact_type="document", payload={"id": "d1"}
        )
        repeated = await repository.create(
            scope=scope,
            artifact_id=artifact.artifact_id,
            artifact_type="document",
            payload={"id": "d1"},
        )
        assert repeated.created_at == artifact.created_at
        assert (
            await repository.get(scope=scope, artifact_id=artifact.artifact_id)
        ).payload == {"id": "d1"}
        with pytest.raises(ArtifactInvariantConflict):
            await repository.create(
                scope=scope,
                artifact_id=artifact.artifact_id,
                artifact_type="document",
                payload={"id": "changed"},
            )
        wrong_subject = ArtifactScope(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-b"),
            conversation_id=scope.conversation_id,
            turn_id=scope.turn_id,
        )
        with pytest.raises(ArtifactNotFound):
            await repository.get(scope=wrong_subject, artifact_id=artifact.artifact_id)
        with pytest.raises(ArtifactNotFound):
            await repository.create(
                scope=wrong_subject,
                artifact_id=artifact.artifact_id,
                artifact_type="document",
                payload={"id": "d1"},
            )
        with pytest.raises(ArtifactNotFound):
            await repository.get(
                scope=ArtifactScope(
                    context=TrustedRequestContext(
                        tenant_id="tenant-b", subject_id="subject-a"
                    ),
                    conversation_id=scope.conversation_id,
                    turn_id=scope.turn_id,
                ),
                artifact_id=artifact.artifact_id,
            )


@pytest.mark.asyncio
async def test_artifact_table_rejects_nonexistent_turn(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        scope = await seed_artifact_scope(pool)
        async with pool.connection() as connection:
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                async with connection.transaction():
                    await connection.execute(
                        """INSERT INTO langgraph_v2.artifacts
                        (tenant_id, artifact_id, conversation_id, turn_id,
                         artifact_type, payload)
                        VALUES (%s, %s, %s, %s, 'document', %s)""",
                        (
                            scope.context.tenant_id,
                            uuid4(),
                            scope.conversation_id,
                            uuid4(),
                            Jsonb({"id": "wrong-turn"}),
                        ),
                    )


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
        scope = await seed_artifact_scope(pool)
        retriever = CountingRetriever()
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_turn_id=scope.turn_id,
            artifact_repository=ArtifactRepository(pool),
            request_context=scope.context,
            retriever=retriever,
        )
        state: LinearGraphState = {
            "query": "hello",
            "conversation_id": "c1",
            "client_request_id": None,
        }
        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)
        assert retriever.calls == 2
        assert "artifact_refs" in first
        assert "artifact_refs" in second
        assert first["artifact_refs"] == second["artifact_refs"]
        assert "artifact_refs" in first
        assert len(first["artifact_refs"]) == 2


def test_empty_retrieval_is_explicit_on_public_stream(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class EmptyRetriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(raw_payload={"query": query, "source": "empty"})

    app = persistent_linear_app(
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

    app = persistent_linear_app(
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
        async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
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
        artifacts = InterruptAfterArtifacts(ArtifactRepository(pool))
        retriever = CountingRetriever()
        turn_id = uuid4()
        scope = await seed_artifact_scope(pool, turn_id=turn_id)
        checkpointer = MemorySaver()
        first_graph = build_linear_graph(
            checkpointer=checkpointer,
            tenant_id="tenant-a",
            current_turn_id=turn_id,
            artifact_repository=artifacts,
            request_context=scope.context,
            retriever=retriever,
            ranker=IdentityRanker(),
            answer_actor=CitingAnswer(),
        )
        state: LinearGraphState = {
            "query": "hello",
            "conversation_id": "c1",
            "turn_id": str(turn_id),
            "client_request_id": None,
        }
        config: RunnableConfig = {"configurable": {"thread_id": "retrieval-recovery"}}
        with pytest.raises(asyncio.CancelledError):
            await first_graph.ainvoke(state, config)

        resume_graph = build_linear_graph(
            checkpointer=checkpointer,
            tenant_id="tenant-a",
            current_turn_id=turn_id,
            artifact_repository=artifacts,
            request_context=scope.context,
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
            scope=scope,
            artifact_id=UUID(artifacts.first_document_ids["d2"]),
        )
        resumed_d2 = await artifacts.get(
            scope=scope,
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
