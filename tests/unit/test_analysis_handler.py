"""Unit tests for analysis observability."""

import unittest.mock
from datetime import UTC, datetime

import pytest

from app.config.models import FlowStep, FlowStepType
from app.models.workflow import (
    AggregatedEvidence,
    AggregatedEvidenceBundle,
    PlannerOutput,
    ToolObservation,
    ToolResultRecord,
)
from app.services.flow_context import FlowContext
from app.services.handlers.analysis import AnalysisHandler


def _evidence(evidence_id: str) -> AggregatedEvidence:
    return AggregatedEvidence(
        evidence_id=evidence_id,
        source="analysis-test",
        content="Evidence content",
        tool_call_id="search_documents:1",
        published_at=datetime(2026, 5, 17, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_analysis_includes_phase7_streaming_and_review_fields(
    mock_emitter: unittest.mock.AsyncMock,
) -> None:
    """Analysis captures streaming policy and execution data."""
    ctx = FlowContext(query="query", session_id="session-a", emitter=mock_emitter)
    ctx.metadata["streaming_policy"] = "token"
    ctx.tool_observations.append(
        ToolObservation(
            tool_name="search",
            status="success",
            task_status_hint="completed",
            result_count=1,
        )
    )
    ctx.tool_results.append(
        ToolResultRecord(
            tool_call_id="search_documents:1",
            tool_name="search",
            source="search",
            normalized_items=[],
        )
    )
    ctx.evidence_store["ev1"] = _evidence("ev1")
    ctx.planner_output = PlannerOutput(
        can_continue_to_aggregation=True,
        reason="Evidence is sufficient.",
        completed_tasks=["search"],
    )
    ctx.aggregated_evidence = AggregatedEvidenceBundle(
        user_query="query",
        standalone_query="query",
        tenant_id="tenant-a",
        synthesis_allowed=True,
        selected_evidence=[ctx.evidence_store["ev1"]],
    )

    result = await AnalysisHandler().handle(
        ctx,
        FlowStep(type=FlowStepType.ANALYSIS),
    )

    analysis = result.metadata["analysis"]
    assert analysis["streaming_policy"] == "token"
    assert analysis["tool_observation_count"] == 1
    assert analysis["tool_result_count"] == 1
    assert analysis["evidence_count"] == 1
    assert analysis["planner_can_continue_to_aggregation"] is True
    assert analysis["planner_completed_task_count"] == 1
    assert analysis["planner_missing_task_count"] == 0
    assert analysis["planner_failed_task_count"] == 0
    assert analysis["aggregated_evidence_count"] == 1
    assert analysis["synthesis_allowed"] is True
    mock_emitter.emit_step_completed.assert_any_await("analysis", analysis)
