from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool
from pydantic_ai.usage import RunUsage

from app.langgraph_v2.answer import AnswerCitation, AnswerResult, AnswerStreamChunk
from app.langgraph_v2.conversation_context import ConversationExchange
from app.langgraph_v2.graph import LinearGraphState, build_linear_graph
from app.langgraph_v2.groundedness import (
    GroundednessAssessment,
    GroundednessOutput,
    PydanticAIGroundednessActor,
)
from app.langgraph_v2.output_assessments import MockOutputAssessmentAudit
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.langgraph_v2_request_support import seed_request_scope
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
    seed_subject_conversation,
)


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
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        return AnswerResult(
            answer="answer [1]",
            citations=[AnswerCitation(index=1, quoted_text=query)],
        )

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
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
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        del query, documents, history
        return AnswerResult(answer="answer without a source")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
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
async def test_low_groundedness_is_advisory_on_each_execution(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        evaluator = _Groundedness()
        audit = MockOutputAssessmentAudit()
        request_id = uuid4()
        await seed_request_scope(pool, request_id=request_id)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=request_id,
            output_assessment_audit=audit,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
            groundedness_actor=evaluator,
        )
        state: LinearGraphState = {
            "query": "hello",
            "conversation_id": "c1",
            "request_id": str(request_id),
        }
        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)

    assert evaluator.calls == 2
    assert "answer" in first
    assert "answer" in second
    assert first["answer"] == second["answer"] == "answer [1]"
    assert "groundedness" in first
    assert "groundedness" in second
    assert first["groundedness"] == second["groundedness"]
    first_groundedness = first["groundedness"]
    assert first_groundedness is not None
    assert first_groundedness.score == 0.2
    assert "final_response" in first
    first_response = first["final_response"]
    assert first_response is not None
    assert first_response.answer == "answer [1]"
    groundedness_records = [
        record for record in audit.records if record.assessment_type == "groundedness"
    ]
    assert len(groundedness_records) == 2
    groundedness_audit = groundedness_records[0]
    assert groundedness_audit.tenant_id == "tenant-a"
    assert groundedness_audit.conversation_id == "c1"
    assert groundedness_audit.request_id == str(request_id)
    assert groundedness_audit.assessment_id.startswith("assessment:")
    assert {record.assessment_id for record in groundedness_records} == {
        groundedness_audit.assessment_id
    }


def test_low_groundedness_preserves_http_tokens_done_and_assistant_message(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "00000000-0000-0000-0000-000000000004"
    asyncio.run(
        seed_subject_conversation(
            langgraph_v2_migrated_database_url,
            conversation_id,
        )
    )
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=_Answer(),
    )
    app.state.langgraph_v2_groundedness_actor = _Groundedness()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": conversation_id},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    events = parse_sse(response.text)
    token_text = "".join(event["data"] for event in events if event["type"] == "token")
    assert token_text == "answer [1]"
    assert events[-1]["type"] == "done"
    assert events[-1]["data"]["answer"] == token_text


def test_new_request_does_not_inherit_prior_groundedness_when_evaluation_fails(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class SucceedsThenFails:
        calls = 0

        async def evaluate(
            self, answer: str, documents: list[Document]
        ) -> GroundednessAssessment:
            del answer, documents
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second-request evaluator unavailable")
            return GroundednessAssessment(
                is_grounded=False,
                score=0.2,
                details="first-request assessment",
            )

    conversation_id = "00000000-0000-0000-0000-000000000005"
    asyncio.run(
        seed_subject_conversation(
            langgraph_v2_migrated_database_url,
            conversation_id,
        )
    )
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=_Answer(),
    )
    app.state.langgraph_v2_groundedness_actor = SucceedsThenFails()
    headers = {
        "X-Application-Id": "tenant-a",
        "X-Subject-Id": "subject-a",
    }

    with TestClient(app) as client:
        first = client.post(
            "/v2/query/stream",
            json={
                "query": "first",
                "sessionId": conversation_id,
                "clientRequestId": "first-request",
            },
            headers=headers,
        )
        second = client.post(
            "/v2/query/stream",
            json={
                "query": "second",
                "sessionId": conversation_id,
                "clientRequestId": "second-request",
            },
            headers=headers,
        )

    first_done = parse_sse(first.text)[-1]
    second_events = parse_sse(second.text)
    second_done = second_events[-1]
    assert first_done["type"] == "done"
    assert first_done["data"]["groundedness"] == {
        "is_grounded": False,
        "score": 0.2,
        "details": "first-request assessment",
    }
    assert second_done["type"] == "done"
    assert second_done["data"]["query"] == "second"
    assert second_done["data"]["groundedness"] is None


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
        evaluator = Failing()
        audit = MockOutputAssessmentAudit()
        request_id = uuid4()
        await seed_request_scope(pool, request_id=request_id)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=request_id,
            output_assessment_audit=audit,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
            groundedness_actor=evaluator,
        )
        state: LinearGraphState = {
            "query": "hello",
            "conversation_id": "c1",
            "request_id": str(request_id),
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
    assert "final_response" in first
    first_response = first["final_response"]
    assert first_response is not None
    assert first_response.answer == "answer [1]"
    groundedness_records = [
        record for record in audit.records if record.assessment_type == "groundedness"
    ]
    assert len(groundedness_records) == 2
    groundedness_audit = groundedness_records[0]
    assert groundedness_audit.result == {
        "failed": True,
        "error": "evaluator unavailable",
    }


@pytest.mark.asyncio
async def test_groundedness_uses_only_cited_documents(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        scope = await seed_request_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_UncitedAnswer(),
            groundedness_actor=_EmptyDocumentGroundedness(),
        )
        result = await graph.ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "request_id": "request-1",
            }
        )

    assert "groundedness" in result
    groundedness = result["groundedness"]
    assert groundedness is not None
    assert groundedness.score == 0.0


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
        scope = await seed_request_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
            groundedness_actor=InvalidScore(),
        )
        result = await graph.ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "request_id": "request-1",
            }
        )

    assert result.get("groundedness") is None
    assert "groundedness_error" in result
    groundedness_error = result["groundedness_error"]
    assert groundedness_error is not None
    assert "less than or equal to 1" in groundedness_error


@pytest.mark.asyncio
async def test_groundedness_actor_setup_failure_is_advisory(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=_Answer(),
    )
    audit = MockOutputAssessmentAudit()
    app.state.langgraph_v2_output_assessment_audit = audit
    with patch(
        "app.langgraph_v2.linear_runtime._resolve_groundedness_actor",
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
    groundedness_audit = next(
        record for record in audit.records if record.assessment_type == "groundedness"
    )
    assert groundedness_audit.tenant_id == "tenant-a"
    assert groundedness_audit.conversation_id == response.headers["x-conversation-id"]
    assert groundedness_audit.request_id == response.headers["x-request-id"]
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
        scope = await seed_request_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            output_assessment_audit=_FailingAssessmentAudit(),
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
            groundedness_actor=_Groundedness(),
        )

        result = await graph.ainvoke(
            LinearGraphState(
                query="hello",
                conversation_id="c1",
                request_id="request-1",
            )
        )

    assert "answer" in result
    assert result["answer"] == "answer [1]"
    assert "final_response" in result
    final_response = result["final_response"]
    assert final_response is not None
    assert final_response.answer == "answer [1]"


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
