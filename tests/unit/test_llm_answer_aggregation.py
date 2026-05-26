"""Unit tests for LLM answer evidence source selection."""

from datetime import UTC, datetime
from typing import Any, Self
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage

from app.config.models import FlowStep, FlowStepType
from app.models.domain import Document
from app.models.workflow import (
    AggregatedEvidence,
    AggregatedEvidenceBundle,
    IntentResult,
)
from app.services.flow_context import FlowContext
from app.services.handlers.llm import LLMHandler


class FakeTextStream:
    """Async context manager for answer streaming tests."""

    def __init__(self, output: str, messages: list[Any] | None = None) -> None:
        self._output = output
        self._messages = messages or []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def stream_text(self):
        yield self._output

    def usage(self) -> RunUsage:
        return RunUsage()

    def new_messages(self) -> list[Any]:
        return self._messages


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
        new_messages = []
        if self.last_prompt is not None:
            new_messages.append(
                ModelRequest(parts=[UserPromptPart(self.last_prompt)])
            )
        return FakeTextStream(self.output, new_messages)


def _handler(fake_agent: FakeAnswerAgent) -> LLMHandler:
    handler = LLMHandler(MagicMock())
    handler._agent_cache[("answer", "pro")] = fake_agent
    return handler


def _evidence(evidence_id: str, content: str) -> AggregatedEvidence:
    return AggregatedEvidence(
        evidence_id=evidence_id,
        source="aggregation-test",
        title="Evidence title",
        content=content,
        tool_call_id="search_documents:1",
        published_at=datetime(2026, 5, 17, tzinfo=UTC),
        score=0.91,
    )


def _structured_evidence(evidence_id: str) -> AggregatedEvidence:
    return AggregatedEvidence(
        evidence_id=evidence_id,
        source="watchlist",
        evidence_type="structured_record",
        content=None,
        structured_facts={
            "ticker": "700 HK",
            "metric": "target_price",
            "value": 420,
            "currency": "HKD",
        },
        tool_call_id="watchlist:1",
        original_item_id="row1",
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
        selected_evidence=[_evidence("ev1", "aggregated evidence content")],
        synthesis_allowed=True,
    )

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
    )

    assert result.llm_response == "aggregated answer"
    assert fake_agent.last_prompt is not None
    assert "<runtime_answer_input>" in fake_agent.last_prompt
    assert "<original_user_query>\nquery\n</original_user_query>" in fake_agent.last_prompt
    assert "<conversation_reference>" not in fake_agent.last_prompt
    assert "aggregated evidence content" in fake_agent.last_prompt
    assert "| type=document_chunk" in fake_agent.last_prompt
    assert "Standalone Query: standalone query" in fake_agent.last_prompt
    assert fake_agent.last_message_history is None
    assert fake_agent.last_deps is not None
    assert not hasattr(fake_agent.last_deps, "reference_data")
    assert "legacy document content" not in fake_agent.last_prompt
    assert result.new_messages
    persisted_request = result.new_messages[0]
    assert isinstance(persisted_request, ModelRequest)
    persisted_part = persisted_request.parts[0]
    assert isinstance(persisted_part, UserPromptPart)
    assert persisted_part.content == "query"
    mock_emitter.emit_token.assert_any_await("aggregated answer")


@pytest.mark.asyncio
async def test_llm_answer_formats_structured_evidence_facts(mock_emitter) -> None:
    """Structured records are passed to answer as structured facts."""
    fake_agent = FakeAnswerAgent(output="structured answer")
    handler = _handler(fake_agent)
    ctx = FlowContext(query="query", emitter=mock_emitter)
    ctx.aggregated_evidence = AggregatedEvidenceBundle(
        user_query="query",
        standalone_query="standalone query",
        tenant_id="tenant-a",
        selected_evidence=[_structured_evidence("ev1")],
        synthesis_allowed=True,
    )

    await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
    )

    assert fake_agent.last_prompt is not None
    assert "| type=structured_record" in fake_agent.last_prompt
    assert "Structured Facts:" in fake_agent.last_prompt
    assert '"metric": "target_price"' in fake_agent.last_prompt
    assert '"value": 420' in fake_agent.last_prompt


def test_llm_answer_static_prompt_defines_evidence_usage_rules() -> None:
    """Answer system prompt teaches the model how to use structured evidence."""
    instructions = LLMHandler(MagicMock())._build_layered_instructions("answer")

    assert "<evidence_usage_rules>" in instructions
    assert "document_chunk" in instructions
    assert "structured_record" in instructions
    assert "Structured Facts as the source of truth" in instructions
    assert "<citation_rules>" in instructions


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
        selected_evidence=[],
        missing_tasks=["search_documents"],
        synthesis_allowed=False,
        synthesis_block_reason="Missing required tasks: search_documents",
    )

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
    )

    assert result.llm_response == "Missing required tasks: search_documents"
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
    assert fake_agent.last_prompt is not None
    assert "<runtime_answer_input>" in fake_agent.last_prompt
    assert "<original_user_query>\nquery\n</original_user_query>" in fake_agent.last_prompt
    assert fake_agent.last_deps is not None
    assert not hasattr(fake_agent.last_deps, "reference_data")
    assert "Standalone Query: standalone query" in fake_agent.last_prompt
    assert "ranked document content" in fake_agent.last_prompt


@pytest.mark.asyncio
async def test_llm_answer_includes_conversation_reference_for_history_intent() -> None:
    """Conversation intents receive sanitized prior chat as answer data."""
    fake_agent = FakeAnswerAgent(output="history summary")
    handler = _handler(fake_agent)
    ctx = FlowContext(query="总结一下刚才的讨论")
    ctx.intent = IntentResult(intent="summarize_history", confidence=0.94)
    ctx.message_history = [
        ModelRequest(parts=[UserPromptPart("我们讨论了 planner 的职责边界。")]),
        ModelResponse(parts=[TextPart("Planner 只负责编排任务和工具。")]),
    ]

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
    )

    assert result.llm_response == "history summary"
    assert fake_agent.last_prompt is not None
    assert "<conversation_reference>" in fake_agent.last_prompt
    assert "[user]\n我们讨论了 planner 的职责边界。" in fake_agent.last_prompt
    assert "[assistant]\nPlanner 只负责编排任务和工具。" in fake_agent.last_prompt


@pytest.mark.asyncio
async def test_llm_answer_omits_conversation_reference_for_normal_intent() -> None:
    """Normal RAG intents do not expose chat history to answer synthesis."""
    fake_agent = FakeAnswerAgent(output="normal answer")
    handler = _handler(fake_agent)
    ctx = FlowContext(query="微软怎么看？")
    ctx.intent = IntentResult(intent="market_outlook", confidence=0.9)
    ctx.message_history = [
        ModelRequest(parts=[UserPromptPart("苹果怎么看？")]),
        ModelResponse(parts=[TextPart("苹果的历史回答不应进入本次答案。")]),
    ]

    await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
    )

    assert fake_agent.last_prompt is not None
    assert "<conversation_reference>" not in fake_agent.last_prompt
    assert "苹果的历史回答不应进入本次答案。" not in fake_agent.last_prompt
