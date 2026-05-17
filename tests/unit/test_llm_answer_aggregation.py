"""Unit tests for LLM answer evidence source selection."""

from datetime import UTC, datetime
from typing import Any, Self
from unittest.mock import MagicMock

import pytest
from pydantic_ai.usage import RunUsage

from app.config.models import FlowStep, FlowStepType
from app.models.domain import Document
from app.models.workflow import AggregatedEvidenceBundle, EvidenceItem
from app.services.flow_context import FlowContext
from app.services.handlers.llm import LLMHandler


class FakeTextStream:
    """Async context manager for answer streaming tests."""

    def __init__(self, output: str) -> None:
        self._output = output

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def stream_text(self):
        yield self._output

    def usage(self) -> RunUsage:
        return RunUsage()

    def new_messages(self) -> list[Any]:
        return []


class FakeAnswerAgent:
    """Fake answer agent that captures deps passed by LLMHandler."""

    def __init__(self, output: str = "answer") -> None:
        self.output = output
        self.last_prompt: str | None = None
        self.last_deps: Any | None = None
        self.last_message_history: Any | None = None
        self.run_count = 0

    def run_stream(self, *args: Any, **kwargs: Any) -> FakeTextStream:
        self.run_count += 1
        self.last_prompt = args[0] if args else None
        self.last_deps = kwargs["deps"]
        self.last_message_history = kwargs.get("message_history")
        return FakeTextStream(self.output)


def _handler(fake_agent: FakeAnswerAgent) -> LLMHandler:
    handler = LLMHandler(MagicMock())
    handler._agent_cache[("answer", "pro")] = fake_agent
    return handler


def _evidence(evidence_id: str, content: str) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        source="aggregation-test",
        source_type="document",
        title="Evidence title",
        content=content,
        retrieved_at=datetime(2026, 5, 17, tzinfo=UTC),
        score=0.91,
    )


@pytest.mark.asyncio
async def test_llm_answer_prefers_aggregated_evidence(mock_emitter) -> None:
    """When aggregation exists, answer prompt uses the evidence bundle."""
    fake_agent = FakeAnswerAgent(output="aggregated answer")
    handler = _handler(fake_agent)
    ctx = FlowContext(query="query", emitter=mock_emitter)
    ctx.message_history = ["previous turn"]
    ctx.documents = [Document(id="legacy-doc", content="legacy document content")]
    ctx.aggregated_evidence = AggregatedEvidenceBundle(
        user_query="query",
        standalone_query="standalone query",
        tenant_id="tenant-a",
        evidence=[_evidence("ev1", "aggregated evidence content")],
        synthesis_allowed=True,
    )

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
    )

    assert result.llm_response == "aggregated answer"
    assert fake_agent.last_prompt == "query"
    assert fake_agent.last_message_history is None
    assert fake_agent.last_deps is not None
    assert "aggregated evidence content" in fake_agent.last_deps.reference_data
    assert "Standalone Query: standalone query" in fake_agent.last_deps.reference_data
    assert "legacy document content" not in fake_agent.last_deps.reference_data
    mock_emitter.emit_token.assert_any_await("aggregated answer")
    mock_emitter.emit_answer_delta.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_answer_buffers_when_approved_answer_policy_enabled(
    mock_emitter,
) -> None:
    """High-compliance answer step buffers draft text before review."""
    fake_agent = FakeAnswerAgent(output="buffered draft")
    handler = _handler(fake_agent)
    ctx = FlowContext(query="query", emitter=mock_emitter)
    ctx.metadata["streaming_policy"] = "approved_answer_only"
    ctx.ranked_documents = [Document(id="doc1", content="ranked document content")]

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
    )

    assert result.llm_response == "buffered draft"
    mock_emitter.emit_token.assert_not_awaited()
    mock_emitter.emit_answer_delta.assert_not_awaited()
    mock_emitter.emit_progress.assert_any_await("answer_buffering")
    mock_emitter.emit_step_completed.assert_any_await(
        "llm:answer",
        {"model": "pro", "buffered": True},
    )


@pytest.mark.asyncio
async def test_llm_answer_blocks_when_aggregation_disallows_synthesis(
    mock_emitter,
) -> None:
    """Disallowed aggregation bundle returns blocked response without model call."""
    fake_agent = FakeAnswerAgent(output="should not be used")
    handler = _handler(fake_agent)
    ctx = FlowContext(query="query", emitter=mock_emitter)
    ctx.aggregated_evidence = AggregatedEvidenceBundle(
        user_query="query",
        standalone_query="query",
        tenant_id="tenant-a",
        evidence=[],
        missing_evidence=["ev1"],
        synthesis_allowed=False,
        synthesis_block_reason="Missing required evidence: ev1",
    )

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
    )

    assert result.llm_response == "Missing required evidence: ev1"
    assert fake_agent.run_count == 0
    mock_emitter.emit_step_completed.assert_any_await(
        "llm:answer",
        {"model": "pro", "blocked": True},
    )


@pytest.mark.asyncio
async def test_llm_answer_keeps_document_fallback_without_aggregation() -> None:
    """Old RAG flows still synthesize from ranked/documents when no bundle exists."""
    fake_agent = FakeAnswerAgent(output="legacy answer")
    handler = _handler(fake_agent)
    ctx = FlowContext(query="query")
    ctx.refined_query = "standalone query"
    ctx.ranked_documents = [Document(id="doc1", content="ranked document content")]

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
    )

    assert result.llm_response == "legacy answer"
    assert fake_agent.last_prompt == "query"
    assert fake_agent.last_deps is not None
    assert "Standalone Query: standalone query" in fake_agent.last_deps.reference_data
    assert "ranked document content" in fake_agent.last_deps.reference_data
