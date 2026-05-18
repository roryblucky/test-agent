"""Unit tests for aggregation handler."""

import pytest

from app.config.models import FlowStep, FlowStepType
from app.models.workflow import (
    IntentResult,
    NormalizedToolResultItem,
    PlannerOutput,
    ToolResultRecord,
)
from app.services.flow_context import FlowContext
from app.services.handlers.aggregation import AggregationHandler


def _tool_result(
    tool_call_id: str,
    item_id: str,
    content: str = "Evidence content",
    score: float = 0.9,
) -> ToolResultRecord:
    return ToolResultRecord(
        tool_call_id=tool_call_id,
        tool_name="search_documents",
        source="search_documents",
        normalized_items=[
            NormalizedToolResultItem(
                item_id=item_id,
                content=content,
                score=score,
            )
        ],
    )


@pytest.mark.asyncio
async def test_aggregation_builds_bundle_from_tool_results(
    mock_emitter,
) -> None:
    """Aggregation selects evidence from normalized tool results."""
    ctx = FlowContext(query="original query", emitter=mock_emitter)
    ctx.metadata["tenant_id"] = "tenant-a"
    ctx.refined_query = "standalone query"
    ctx.intent = IntentResult(intent="knowledge_query", confidence=0.9)
    ctx.active_skills = ["generic-search"]
    ctx.tool_results.append(
        _tool_result("search_documents:1", "doc1", "First evidence", 0.7)
    )
    ctx.tool_results.append(
        _tool_result("search_documents:2", "doc2", "Second evidence", 0.9)
    )
    ctx.planner_output = PlannerOutput(
        can_continue_to_aggregation=True,
        reason="Evidence found.",
        completed_tasks=["search_documents"],
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
    assert [item.content for item in bundle.selected_evidence] == [
        "Second evidence",
        "First evidence",
    ]
    assert all(
        item.evidence_id.startswith("evidence:")
        for item in bundle.selected_evidence
    )
    assert list(result.evidence_store) == [
        item.evidence_id for item in bundle.selected_evidence
    ]
    assert bundle.synthesis_allowed is True
    assert bundle.synthesis_block_reason is None
    mock_emitter.emit_step_completed.assert_any_await(
        "aggregation",
        {
            "selected_evidence_count": 2,
            "excluded_evidence_count": 0,
            "missing_task_count": 0,
            "failed_task_count": 0,
            "synthesis_allowed": True,
        },
    )


@pytest.mark.asyncio
async def test_aggregation_blocks_when_required_task_missing() -> None:
    """Missing required tasks are surfaced and synthesis is not allowed."""
    ctx = FlowContext(query="original query")
    ctx.tool_results.append(_tool_result("search_documents:1", "doc1"))
    ctx.planner_output = PlannerOutput(
        can_continue_to_aggregation=True,
        reason="Planner thinks synthesis is possible.",
        missing_tasks=["search_documents"],
    )

    result = await AggregationHandler().handle(
        ctx,
        FlowStep(type=FlowStepType.AGGREGATION),
    )

    bundle = result.aggregated_evidence
    assert bundle is not None
    assert [item.content for item in bundle.selected_evidence] == ["Evidence content"]
    assert bundle.missing_tasks == ["search_documents"]
    assert bundle.synthesis_allowed is False
    assert bundle.synthesis_block_reason == "Missing required tasks: search_documents"


@pytest.mark.asyncio
async def test_aggregation_requires_planner_output() -> None:
    """Aggregation fails closed if planner output is absent."""
    with pytest.raises(ValueError, match="requires ctx.planner_output"):
        await AggregationHandler().handle(
            FlowContext(query="query"),
            FlowStep(type=FlowStepType.AGGREGATION),
        )
