"""Unit tests for analysis observability."""

from datetime import UTC, datetime

import pytest

from app.config.models import FlowStep, FlowStepType
from app.models.workflow import (
    AggregatedEvidenceBundle,
    ComplianceReviewResult,
    EvidenceItem,
    PlannerOutput,
    ToolObservation,
)
from app.services.flow_context import FlowContext
from app.services.handlers.analysis import AnalysisHandler


def _evidence(evidence_id: str) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        source="analysis-test",
        content="Evidence content",
        retrieved_at=datetime(2026, 5, 17, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_analysis_includes_phase7_streaming_and_review_fields(
    mock_emitter,
) -> None:
    """Analysis captures streaming policy and high-compliance review status."""
    ctx = FlowContext(query="query", session_id="session-a", emitter=mock_emitter)
    ctx.metadata["streaming_policy"] = "approved_answer_only"
    ctx.tool_observations.append(
        ToolObservation(tool_name="search", status="success", evidence_ids=["ev1"])
    )
    ctx.evidence_store["ev1"] = _evidence("ev1")
    ctx.planner_output = PlannerOutput(
        can_synthesize=True,
        reason="Evidence is sufficient.",
        evidence_ids=["ev1"],
    )
    ctx.aggregated_evidence = AggregatedEvidenceBundle(
        user_query="query",
        standalone_query="query",
        tenant_id="tenant-a",
        synthesis_allowed=True,
        evidence=[ctx.evidence_store["ev1"]],
    )
    ctx.compliance_review = ComplianceReviewResult(
        passed=False,
        violations=["unsupported_claim"],
    )

    result = await AnalysisHandler().handle(
        ctx,
        FlowStep(type=FlowStepType.ANALYSIS),
    )

    analysis = result.metadata["analysis"]
    assert analysis["streaming_policy"] == "approved_answer_only"
    assert analysis["tool_observation_count"] == 1
    assert analysis["evidence_count"] == 1
    assert analysis["planner_can_synthesize"] is True
    assert analysis["planner_evidence_count"] == 1
    assert analysis["aggregated_evidence_count"] == 1
    assert analysis["synthesis_allowed"] is True
    assert analysis["compliance_passed"] is False
    assert analysis["compliance_violation_count"] == 1
    mock_emitter.emit_step_completed.assert_any_await("analysis", analysis)
