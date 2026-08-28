from __future__ import annotations

from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.artifacts import ArtifactNotFound, ArtifactRepository
from app.langgraph_v2.graph import build_tracer_graph
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import RunEventRepository
from app.models.domain import Document


@pytest.mark.asyncio
async def test_artifact_repository_is_tenant_scoped(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(langgraph_v2_migrated_database_url, min_size=1, max_size=2) as pool:
        repository = ArtifactRepository(pool)
        artifact = await repository.create(tenant_id="tenant-a", artifact_type="document", payload={"id": "d1"})
        assert (await repository.get(tenant_id="tenant-a", artifact_id=artifact.artifact_id)).payload == {"id": "d1"}
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
            return RetrievalResult(documents=[Document(id="d1", content=query)], raw_payload={"query": query})

    async with AsyncConnectionPool(langgraph_v2_migrated_database_url, min_size=1, max_size=2) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(tenant_id="tenant-a", run_id=uuid4(), conversation_id="c1", owner_instance_id="i1")
        retriever = CountingRetriever()
        graph = build_tracer_graph(
            phase_context=PhaseExecutionContext(
                repository=PhaseResultRepository(pool), tenant_id="tenant-a", run_id=run.run_id,
                owner_instance_id="i1", execution_epoch=run.execution_epoch,
            ),
            artifact_repository=ArtifactRepository(pool), retriever=retriever,
        )
        state = {"query": "hello", "conversation_id": "c1", "client_request_id": None, "events": []}
        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)
        assert retriever.calls == 1
        assert first["artifact_refs"] == second["artifact_refs"]
        assert len(first["artifact_refs"]) == 2
