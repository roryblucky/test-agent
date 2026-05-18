"""Unit tests for LLM compliance review mode."""

from datetime import UTC, datetime
from typing import Any, Self
from unittest.mock import MagicMock

import pytest
from pydantic_ai.usage import RunUsage

from app.config.models import FlowStep, FlowStepType
from app.models.workflow import (
    AggregatedEvidence,
    AggregatedEvidenceBundle,
    ComplianceReviewResult,
)
from app.services.flow_context import FlowContext
from app.services.handlers.llm import LLMHandler


class FakeStructuredStream:
    """Async context manager for structured review tests."""

    def __init__(self, output: Any) -> None:
        self._output = output

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get_output(self) -> Any:
        return self._output

    def usage(self) -> RunUsage:
        return RunUsage()


class FakeComplianceAgent:
    """Fake compliance review agent that captures its prompt deps."""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.last_prompt: str | None = None
        self.last_deps: Any | None = None
        self.run_count = 0

    def run_stream(self, prompt: str, *, deps: Any, **kwargs: Any) -> FakeStructuredStream:
        self.run_count += 1
        self.last_prompt = prompt
        self.last_deps = deps
        return FakeStructuredStream(self.output)


def _handler(fake_agent: FakeComplianceAgent) -> LLMHandler:
    handler = LLMHandler(MagicMock())
    handler._agent_cache[("compliance_review", "fast")] = fake_agent
    return handler


def _aggregated_bundle() -> AggregatedEvidenceBundle:
    return AggregatedEvidenceBundle(
        user_query="query",
        standalone_query="query",
        tenant_id="tenant-a",
        selected_evidence=[
            AggregatedEvidence(
                evidence_id="ev1",
                source="test-source",
                content="approved evidence",
                tool_call_id="search_documents:1",
                published_at=datetime(2026, 5, 17, tzinfo=UTC),
            )
        ],
        synthesis_allowed=True,
    )


@pytest.mark.asyncio
async def test_compliance_review_passes_and_keeps_answer(mock_emitter) -> None:
    """Passing review records the result and keeps the draft as final answer."""
    fake_agent = FakeComplianceAgent(
        ComplianceReviewResult(passed=True, reason="Looks compliant.")
    )
    handler = _handler(fake_agent)
    ctx = FlowContext(query="query", emitter=mock_emitter)
    ctx.metadata["streaming_policy"] = "approved_answer_only"
    ctx.llm_response = "draft answer"
    ctx.aggregated_evidence = _aggregated_bundle()

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="compliance_review", model="fast"),
    )

    assert result.llm_response == "draft answer"
    assert result.draft_answer == "draft answer"
    assert result.compliance_review == ComplianceReviewResult(
        passed=True,
        reason="Looks compliant.",
    )
    assert fake_agent.last_deps is not None
    assert "draft answer" in fake_agent.last_deps.reference_data
    assert "approved evidence" in fake_agent.last_deps.reference_data
    mock_emitter.emit_token.assert_not_awaited()
    mock_emitter.emit_answer_delta.assert_any_await("draft answer")
    mock_emitter.emit_step_completed.assert_any_await(
        "llm:compliance_review",
        {"model": "fast", "passed": True, "violation_count": 0},
    )


@pytest.mark.asyncio
async def test_compliance_review_failure_replaces_answer_with_safe_response(
    mock_emitter,
) -> None:
    """Failing review prevents release of the draft answer."""
    fake_agent = FakeComplianceAgent(
        ComplianceReviewResult(
            passed=False,
            reason="Unsupported claim.",
            violations=["unsupported_claim"],
            required_changes=["Remove unsupported claim."],
            safe_response="I cannot answer from the available evidence.",
        )
    )
    handler = _handler(fake_agent)
    ctx = FlowContext(query="query", emitter=mock_emitter)
    ctx.metadata["streaming_policy"] = "approved_answer_only"
    ctx.llm_response = "unsafe draft"
    ctx.aggregated_evidence = _aggregated_bundle()

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="compliance_review", model="fast"),
    )

    assert result.draft_answer == "unsafe draft"
    assert result.llm_response == "I cannot answer from the available evidence."
    assert result.compliance_review is not None
    assert result.compliance_review.passed is False
    assert result.compliance_review.violations == ["unsupported_claim"]
    mock_emitter.emit_answer_delta.assert_any_await(
        "I cannot answer from the available evidence."
    )


@pytest.mark.asyncio
async def test_compliance_review_failure_without_safe_response_blocks_answer() -> None:
    """Failing review without a safe response uses a generic blocked response."""
    fake_agent = FakeComplianceAgent(
        {
            "passed": False,
            "reason": "Policy issue.",
            "violations": ["policy_issue"],
        }
    )
    handler = _handler(fake_agent)
    ctx = FlowContext(query="query")
    ctx.llm_response = "unsafe draft"

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="compliance_review", model="fast"),
    )

    assert result.draft_answer == "unsafe draft"
    assert result.llm_response.startswith(
        "The draft answer could not be released"
    )
    assert result.compliance_review is not None
    assert result.compliance_review.passed is False


@pytest.mark.asyncio
async def test_compliance_review_requires_prior_llm_response() -> None:
    """Compliance review fails closed when there is no draft answer."""
    handler = _handler(FakeComplianceAgent(ComplianceReviewResult(passed=True)))

    with pytest.raises(ValueError, match="requires a prior LLM response"):
        await handler.handle(
            FlowContext(query="query"),
            FlowStep(type=FlowStepType.LLM, mode="compliance_review", model="fast"),
        )
