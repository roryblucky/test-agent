from __future__ import annotations

from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult
from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.graph import build_tracer_graph
from app.langgraph_v2.groundedness import GroundednessResult
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import RunEventRepository
from app.models.domain import Document


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id="d1", content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        return RerankingResult(documents=documents)


class _Answer:
    async def answer(self, query: str, documents: list[Document]) -> AnswerResult:
        return AnswerResult(answer="answer")


class _Groundedness:
    calls = 0

    async def evaluate(self, answer: str, documents: list[Document]) -> GroundednessResult:
        self.calls += 1
        assert answer == "answer"
        assert documents[0].id == "d1"
        return GroundednessResult(is_grounded=False, score=0.2, details="advisory")


@pytest.mark.asyncio
async def test_low_groundedness_is_advisory_and_replayed(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(langgraph_v2_migrated_database_url, min_size=1, max_size=2) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a", run_id=uuid4(), conversation_id="c1", owner_instance_id="i1"
        )
        evaluator = _Groundedness()
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool), artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a", run_id=run.run_id, owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            phase_context=context, retriever=_Retriever(), ranker=_Ranker(),
            answer_actor=_Answer(), groundedness_actor=evaluator,
        )
        state = {"query": "hello", "conversation_id": "c1", "client_request_id": None, "events": []}
        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)

    assert evaluator.calls == 1
    assert first["answer"] == second["answer"] == "answer"
    assert first["groundedness"] == second["groundedness"]
    assert first["groundedness"].score == 0.2
    assert any(event.get("step") == "groundedness" for event in first["events"])


@pytest.mark.asyncio
async def test_groundedness_failure_is_explicit_and_replayed(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class Failing:
        calls = 0

        async def evaluate(self, answer: str, documents: list[Document]) -> GroundednessResult:
            self.calls += 1
            raise RuntimeError("evaluator unavailable")

    async with AsyncConnectionPool(langgraph_v2_migrated_database_url, min_size=1, max_size=2) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a", run_id=uuid4(), conversation_id="c1", owner_instance_id="i1"
        )
        evaluator = Failing()
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool), artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a", run_id=run.run_id, owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            phase_context=context, retriever=_Retriever(), ranker=_Ranker(),
            answer_actor=_Answer(), groundedness_actor=evaluator,
        )
        state = {"query": "hello", "conversation_id": "c1", "client_request_id": None, "events": []}
        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)

    assert evaluator.calls == 1
    assert first["groundedness_error"] == second["groundedness_error"] == "evaluator unavailable"
    assert any(event["type"] == "error" for event in first["events"])
