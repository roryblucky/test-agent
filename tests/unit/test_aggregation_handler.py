"""Unit tests for aggregation handler."""

from datetime import UTC, datetime

import pytest

from app.config.models import FlowStep, FlowStepType
from app.models.workflow import EvidenceItem, IntentResult, PlannerOutput
from app.services.flow_context import FlowContext
from app.services.handlers.aggregation import AggregationHandler


def _evidence(evidence_id: str, content: str = "Evidence content") -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        source="test-source",
        content=content,
        retrieved_at=datetime(2026, 5, 17, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_aggregation_builds_bundle_from_planner_evidence_ids(
    mock_emitter,
) -> None:
    """Aggregation reads full evidence from context and writes synthesis bundle."""
    ctx = FlowContext(query="original query", emitter=mock_emitter)
    ctx.metadata["tenant_id"] = "tenant-a"
    ctx.refined_query = "standalone query"
    ctx.intent = IntentResult(intent="knowledge_query", confidence=0.9)
    ctx.evidence_store["ev1"] = _evidence("ev1", "First evidence")
    ctx.evidence_store["ev2"] = _evidence("ev2", "Second evidence")
    ctx.planner_output = PlannerOutput(
        active_skills=["generic-search"],
        can_synthesize=True,
        reason="Evidence found.",
        evidence_ids=["ev1", "ev2", "ev1"],
    )

    result = await AggregationHandler().handle(
        ctx,
        FlowStep(type=FlowStepType.AGGREGATION),
    )

    bundle = result.aggregated_evidence
    assert bundle is not None
    assert bundle.user_query == "original query"
    assert bundle.standalone_query == "standalone query"
    assert bundle.tenant_id == "tenant-a"
    assert bundle.intent == "knowledge_query"
    assert bundle.active_skills == ["generic-search"]
    assert [item.id for item in bundle.evidence] == ["ev1", "ev2"]
    assert bundle.synthesis_allowed is True
    assert bundle.synthesis_block_reason is None
    mock_emitter.emit_step_completed.assert_any_await(
        "aggregation",
        {
            "evidence_count": 2,
            "missing_evidence_count": 0,
            "synthesis_allowed": True,
        },
    )


@pytest.mark.asyncio
async def test_aggregation_blocks_when_required_evidence_missing() -> None:
    """Missing evidence IDs are surfaced and synthesis is not allowed."""
    ctx = FlowContext(query="original query")
    ctx.evidence_store["ev1"] = _evidence("ev1")
    ctx.planner_output = PlannerOutput(
        can_synthesize=True,
        reason="Planner thinks synthesis is possible.",
        evidence_ids=["ev1", "missing-ev"],
    )

    result = await AggregationHandler().handle(
        ctx,
        FlowStep(type=FlowStepType.AGGREGATION),
    )

    bundle = result.aggregated_evidence
    assert bundle is not None
    assert [item.id for item in bundle.evidence] == ["ev1"]
    assert bundle.missing_evidence == ["missing-ev"]
    assert bundle.synthesis_allowed is False
    assert bundle.synthesis_block_reason == "Missing required evidence: missing-ev"


@pytest.mark.asyncio
async def test_aggregation_requires_planner_output() -> None:
    """Aggregation fails closed if planner output is absent."""
    with pytest.raises(ValueError, match="requires ctx.planner_output"):
        await AggregationHandler().handle(
            FlowContext(query="query"),
            FlowStep(type=FlowStepType.AGGREGATION),
        )
