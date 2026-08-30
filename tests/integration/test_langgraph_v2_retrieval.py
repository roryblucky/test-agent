from __future__ import annotations

from typing import Any
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.artifacts import ArtifactNotFound, ArtifactRepository
from app.langgraph_v2.graph import TracerState, build_tracer_graph
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
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
async def test_retrieval_persists_artifacts_and_replays_without_second_call(
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
        assert retriever.calls == 1
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
async def test_retrieval_replays_after_commit_window_crash(
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

    class CrashAfterRetrievalCommit(PhaseResultRepository):
        crashed = False

        async def commit(self, **kwargs: Any):  # type: ignore[no-untyped-def]
            result = await super().commit(**kwargs)
            if kwargs["phase"].phase_name == "retrieval" and not self.crashed:
                self.crashed = True
                raise RuntimeError("crash after retrieval commit")
            return result

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        repository = CrashAfterRetrievalCommit(pool)
        retriever = CountingRetriever()
        context = PhaseExecutionContext(
            repository=repository,
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="i1",
            execution_epoch=run.execution_epoch,
            artifact_repository=ArtifactRepository(pool),
        )
        graph = build_tracer_graph(phase_context=context, retriever=retriever)
        state: TracerState = {
            "query": "hello",
            "conversation_id": "c1",
            "client_request_id": None,
            "events": [],
        }
        with pytest.raises(RuntimeError, match="crash after retrieval commit"):
            await graph.ainvoke(state)
        recovered = await graph.ainvoke(state)
        assert retriever.calls == 1
        assert "artifact_refs" in recovered
        assert len(recovered["artifact_refs"]) == 2
        events = await RunEventRepository(pool).list_events("tenant-a", run.run_id)
        assert len({event.event_key for event in events}) == len(events)
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT count(*) FROM langgraph_v2.artifacts WHERE tenant_id = %s",
                    ("tenant-a",),
                )
                row = await cursor.fetchone()
                assert row is not None
                assert row[0] == 2
