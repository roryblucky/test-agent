from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool
from pydantic_ai.usage import RunUsage

from app.langgraph_v2.answer import AnswerCitation, AnswerResult, AnswerStreamChunk
from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.graph import TracerState, build_tracer_graph
from app.langgraph_v2.groundedness import (
    GroundednessAssessment,
    GroundednessOutput,
    PydanticAIGroundednessActor,
)
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.output_assessments import MockOutputAssessmentAudit
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

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationTurn]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _Groundedness:
    calls = 0

    async def evaluate(
        self, answer: str, documents: list[Document]
    ) -> GroundednessAssessment:
        self.calls += 1
        assert answer == "answer [1]"
        assert documents[0].id == "d1"
        return GroundednessAssessment(is_grounded=False, score=0.2, details="advisory")


class _UncitedAnswer:
    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        del query, documents, history
        return AnswerResult(answer="answer without a source")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationTurn]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _EmptyDocumentGroundedness:
    async def evaluate(
        self, answer: str, documents: list[Document]
    ) -> GroundednessAssessment:
        assert answer == "answer without a source"
        assert documents == []
        return GroundednessAssessment(is_grounded=False, score=0.0, details="uncited")


class _FailingAssessmentAudit:
    async def record(self, assessment: object) -> None:
        del assessment
        raise RuntimeError("audit unavailable")


@pytest.mark.asyncio
async def test_low_groundedness_is_advisory_without_phase_journal(
    langgraph_v2_migrated_database_url: str,
) -> None:
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
        evaluator = _Groundedness()
        audit = MockOutputAssessmentAudit()
        turn_id = uuid4()
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            current_turn_id=turn_id,
            output_assessment_audit=audit,
        )
        graph = build_tracer_graph(
            phase_context=context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
            groundedness_actor=evaluator,
        )
        state: TracerState = {
            "query": "hello",
            "conversation_id": "c1",
            "turn_id": str(turn_id),
            "client_request_id": None,
            "events": [],
        }
        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)
        phase = await context.repository.get_completed(
            "tenant-a", run.run_id, "groundedness"
        )

    assert evaluator.calls == 2
    assert "answer" in first
    assert "answer" in second
    assert first["answer"] == second["answer"] == "answer [1]"
    assert "groundedness" in first
    assert "groundedness" in second
    assert first["groundedness"] == second["groundedness"]
    assert "groundedness" in first
    assert first["groundedness"].score == 0.2
    assert first["events"][-1]["type"] == "done"
    assert first["events"][-1]["data"]["answer"] == "answer [1]"
    groundedness_records = [
        record for record in audit.records if record.assessment_type == "groundedness"
    ]
    assert len(groundedness_records) == 2
    groundedness_audit = groundedness_records[0]
    assert groundedness_audit.tenant_id == "tenant-a"
    assert groundedness_audit.conversation_id == "c1"
    assert groundedness_audit.turn_id == turn_id
    assert groundedness_audit.assessment_id == (
        f"turn:{turn_id}:assessment:groundedness"
    )
    assert phase is None
    assert any(event.get("step") == "groundedness" for event in first["events"])


@pytest.mark.asyncio
async def test_groundedness_failure_is_explicit_on_each_execution(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class Failing:
        calls = 0

        async def evaluate(
            self, answer: str, documents: list[Document]
        ) -> GroundednessAssessment:
            self.calls += 1
            raise RuntimeError("evaluator unavailable")

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
        evaluator = Failing()
        audit = MockOutputAssessmentAudit()
        turn_id = uuid4()
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            current_turn_id=turn_id,
            output_assessment_audit=audit,
        )
        graph = build_tracer_graph(
            phase_context=context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
            groundedness_actor=evaluator,
        )
        state: TracerState = {
            "query": "hello",
            "conversation_id": "c1",
            "turn_id": str(turn_id),
            "client_request_id": None,
            "events": [],
        }
        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)

    assert evaluator.calls == 2
    assert "groundedness_error" in first
    assert "groundedness_error" in second
    assert (
        first["groundedness_error"]
        == second["groundedness_error"]
        == "evaluator unavailable"
    )
    assert first["events"][-1]["type"] == "done"
    assert first["events"][-1]["data"]["answer"] == "answer [1]"
    groundedness_records = [
        record for record in audit.records if record.assessment_type == "groundedness"
    ]
    assert len(groundedness_records) == 2
    groundedness_audit = groundedness_records[0]
    assert groundedness_audit.result == {
        "failed": True,
        "error": "evaluator unavailable",
    }
    failure_events = [
        event
        for event in first["events"]
        if event["event_key"] == "phase:groundedness:error:1"
    ]
    assert failure_events == [
        {
            "event_key": "phase:groundedness:error:1",
            "type": "step_completed",
            "step": "groundedness",
            "data": {"failed": True, "error": "evaluator unavailable"},
            "sequence": failure_events[0]["sequence"],
            "journal_policy": "checkpoint_only",
        }
    ]


@pytest.mark.asyncio
async def test_groundedness_uses_only_cited_documents(
    langgraph_v2_migrated_database_url: str,
) -> None:
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
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            phase_context=context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_UncitedAnswer(),
            groundedness_actor=_EmptyDocumentGroundedness(),
        )
        result = await graph.ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "client_request_id": None,
                "events": [],
            }
        )

    assert "groundedness" in result
    assert result["groundedness"].score == 0.0


@pytest.mark.asyncio
async def test_groundedness_rejects_out_of_range_scores(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class InvalidScore:
        async def evaluate(
            self, answer: str, documents: list[Document]
        ) -> GroundednessAssessment:
            del answer, documents
            return GroundednessAssessment(is_grounded=True, score=2.0)

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
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            phase_context=context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
            groundedness_actor=InvalidScore(),
        )
        result = await graph.ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "client_request_id": None,
                "events": [],
            }
        )

    assert result.get("groundedness") is None
    assert "groundedness_error" in result
    assert "less than or equal to 1" in result["groundedness_error"]


@pytest.mark.asyncio
async def test_groundedness_actor_setup_failure_is_advisory(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=_Answer(),
    )
    audit = MockOutputAssessmentAudit()
    app.state.langgraph_v2_output_assessment_audit = audit
    with patch(
        "app.langgraph_v2.api._resolve_groundedness_actor",
        side_effect=RuntimeError("groundedness model is unavailable"),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/v2/query/stream",
                json={"query": "hello"},
                headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
            )

    events = parse_sse(response.text)
    assert response.status_code == 200
    groundedness_failure = next(
        event
        for event in events
        if event.get("type") == "step_completed"
        and event.get("data")
        == {
            "failed": True,
            "error": "groundedness model is unavailable",
        }
    )
    assert groundedness_failure["type"] == "step_completed"
    assert groundedness_failure["data"] == {
        "failed": True,
        "error": "groundedness model is unavailable",
    }
    assert events[-1]["type"] == "done"
    assert events[-1]["data"]["answer"] == "answer [1]"
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        run = await RunEventRepository(pool).get_run(
            "tenant-a", UUID(response.headers["x-run-id"])
        )
    assert run.status == "completed"
    groundedness_audit = next(
        record for record in audit.records if record.assessment_type == "groundedness"
    )
    assert groundedness_audit.tenant_id == "tenant-a"
    assert groundedness_audit.conversation_id == response.headers["x-conversation-id"]
    assert groundedness_audit.turn_id == UUID(response.headers["x-turn-id"])
    assert groundedness_audit.result == {
        "failed": True,
        "error": "groundedness model is unavailable",
    }


@pytest.mark.asyncio
async def test_assessment_audit_failure_does_not_gate_answer(
    langgraph_v2_migrated_database_url: str,
) -> None:
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
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            current_turn_id=uuid4(),
            output_assessment_audit=_FailingAssessmentAudit(),
        )
        graph = build_tracer_graph(
            phase_context=context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
            groundedness_actor=_Groundedness(),
        )

        result = await graph.ainvoke(
            TracerState(
                query="hello",
                conversation_id="c1",
                client_request_id=None,
                events=[],
            )
        )

    assert "answer" in result
    assert result["answer"] == "answer [1]"
    assert result["events"][-1]["type"] == "done"


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
