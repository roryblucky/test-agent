from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool
from pydantic_ai.usage import RunUsage

from app.langgraph_v2.answer import AnswerCitation, AnswerResult
from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.graph import build_tracer_graph
from app.langgraph_v2.groundedness import (
    GroundednessAssessment,
    GroundednessOutput,
    GroundednessResult,
    PydanticAIGroundednessActor,
)
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import RunEventRepository
from app.models.domain import Document
from tests.integration.test_langgraph_v2_tracer import parse_sse, persistent_tracer_app


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id="d1", content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        return RerankingResult(documents=documents)


class _Answer:
    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        return AnswerResult(
            answer="answer [1]",
            citations=[AnswerCitation(index=1, quoted_text=query)],
        )


class _Groundedness:
    calls = 0

    async def evaluate(self, answer: str, documents: list[Document]) -> GroundednessResult:
        self.calls += 1
        assert answer == "answer [1]"
        assert documents[0].id == "d1"
        return GroundednessResult(is_grounded=False, score=0.2, details="advisory")


class _UncitedAnswer:
    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        del query, documents, history
        return AnswerResult(answer="answer without a source")


class _EmptyDocumentGroundedness:
    async def evaluate(self, answer: str, documents: list[Document]) -> GroundednessResult:
        assert answer == "answer without a source"
        assert documents == []
        return GroundednessResult(is_grounded=False, score=0.0, details="uncited")


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
        phase = await context.repository.get_completed("tenant-a", run.run_id, "groundedness")

    assert evaluator.calls == 1
    assert first["answer"] == second["answer"] == "answer [1]"
    assert first["groundedness"] == second["groundedness"]
    assert first["groundedness"].score == 0.2
    assert phase is not None
    assert phase.normalized_result["usage"] == {}
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


@pytest.mark.asyncio
async def test_groundedness_uses_only_cited_documents(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(langgraph_v2_migrated_database_url, min_size=1, max_size=2) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a", run_id=uuid4(), conversation_id="c1", owner_instance_id="i1"
        )
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool), artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a", run_id=run.run_id, owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            phase_context=context, retriever=_Retriever(), ranker=_Ranker(),
            answer_actor=_UncitedAnswer(), groundedness_actor=_EmptyDocumentGroundedness(),
        )
        result = await graph.ainvoke(
            {"query": "hello", "conversation_id": "c1", "client_request_id": None, "events": []}
        )

    assert result["groundedness"].score == 0.0


@pytest.mark.asyncio
async def test_groundedness_reuses_atomic_commit_after_crash_window(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class CrashAfterGroundednessCommit(PhaseResultRepository):
        crashed = False

        async def commit(self, **kwargs):  # type: ignore[no-untyped-def]
            result = await super().commit(**kwargs)
            if kwargs["phase"].phase_name == "groundedness" and not self.crashed:
                self.crashed = True
                raise RuntimeError("crash after groundedness commit")
            return result

    async with AsyncConnectionPool(langgraph_v2_migrated_database_url, min_size=1, max_size=2) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a", run_id=uuid4(), conversation_id="c1", owner_instance_id="i1"
        )
        evaluator = _Groundedness()
        context = PhaseExecutionContext(
            repository=CrashAfterGroundednessCommit(pool), artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a", run_id=run.run_id, owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            phase_context=context, retriever=_Retriever(), ranker=_Ranker(),
            answer_actor=_Answer(), groundedness_actor=evaluator,
        )
        state = {"query": "hello", "conversation_id": "c1", "client_request_id": None, "events": []}
        with pytest.raises(RuntimeError, match="crash after groundedness commit"):
            await graph.ainvoke(state)
        recovered = await graph.ainvoke(state)
        events = await runs.list_events("tenant-a", run.run_id)

    assert evaluator.calls == 1
    assert recovered["groundedness"].score == 0.2
    assert len({event.event_key for event in events}) == len(events)
    assert sum(event.event_key == "phase:groundedness:step_completed:1" for event in events) == 1


@pytest.mark.asyncio
async def test_groundedness_rejects_out_of_range_scores(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class InvalidScore:
        async def evaluate(self, answer: str, documents: list[Document]) -> GroundednessResult:
            del answer, documents
            return GroundednessResult(is_grounded=True, score=2.0)

    async with AsyncConnectionPool(langgraph_v2_migrated_database_url, min_size=1, max_size=2) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a", run_id=uuid4(), conversation_id="c1", owner_instance_id="i1"
        )
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool), artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a", run_id=run.run_id, owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            phase_context=context, retriever=_Retriever(), ranker=_Ranker(),
            answer_actor=_Answer(), groundedness_actor=InvalidScore(),
        )
        result = await graph.ainvoke(
            {"query": "hello", "conversation_id": "c1", "client_request_id": None, "events": []}
        )

    assert result.get("groundedness") is None
    assert "less than or equal to 1" in result["groundedness_error"]


@pytest.mark.asyncio
async def test_groundedness_actor_setup_failure_terminalizes_run(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=_Answer(),
    )
    with patch(
        "app.langgraph_v2.api._resolve_groundedness_actor",
        side_effect=RuntimeError("groundedness model is unavailable"),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/v2/query/stream",
                json={"query": "hello"},
                headers={"X-Application-Id": "tenant-a"},
            )

    events = parse_sse(response.text)
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert events[-1]["data"] == "groundedness model is unavailable"
    async with AsyncConnectionPool(langgraph_v2_migrated_database_url, min_size=1, max_size=2) as pool:
        run = await RunEventRepository(pool).get_run(
            "tenant-a", UUID(response.headers["x-run-id"])
        )
    assert run.status == "failed"


@pytest.mark.asyncio
async def test_pydantic_groundedness_actor_preserves_usage() -> None:
    class Result:
        output = GroundednessOutput(is_grounded=True, score=0.9)

        def usage(self) -> RunUsage:
            return RunUsage(input_tokens=4, output_tokens=2)

    class AgentStub:
        async def run(self, prompt: str) -> Result:
            assert "Evidence:" in prompt
            return Result()

    actor = PydanticAIGroundednessActor(AgentStub())  # type: ignore[arg-type]
    result = await actor.evaluate("answer", [Document(id="d1", content="evidence")])

    assert isinstance(result, GroundednessAssessment)
    assert result.usage["input_tokens"] == 4
    assert result.usage["output_tokens"] == 2
