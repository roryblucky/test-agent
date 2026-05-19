"""Unit tests for aggregation handler."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.config.models import FlowStep, FlowStepType
from app.models.workflow import (
    IntentResult,
    NormalizedToolResultItem,
    PlannerOutput,
    PlannerTask,
    ToolResultRecord,
)
from app.services.flow_context import FlowContext
from app.services.handlers.aggregation import AggregationHandler


def _tool_result(
    tool_call_id: str,
    item_id: str,
    content: str = "Evidence content",
    score: float | None = 0.9,
    source: str = "search_documents",
    metadata: dict[str, Any] | None = None,
    published_at: datetime | None = None,
) -> ToolResultRecord:
    return ToolResultRecord(
        tool_call_id=tool_call_id,
        tool_name="search_documents",
        source=source,
        normalized_items=[
            NormalizedToolResultItem(
                item_id=item_id,
                content=content,
                score=score,
                published_at=published_at,
                metadata=metadata or {},
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
        planned_tasks=[
            PlannerTask(
                task_id="search_documents",
                description="Search normalized evidence.",
                status="missing",
                tool_name="search_documents",
            )
        ],
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
async def test_aggregation_ignores_legacy_filter_settings() -> None:
    """Aggregation does not enforce source, score, max count, or global age."""
    ctx = FlowContext(query="original query")
    old_date = datetime.now(UTC) - timedelta(days=30)
    ctx.tool_results.append(
        _tool_result("search_documents:1", "trusted-high", "High evidence", 0.9)
    )
    ctx.tool_results.append(
        _tool_result("search_documents:2", "trusted-low", "Low evidence", 0.1)
    )
    ctx.tool_results.append(
        _tool_result(
            "search_documents:3",
            "external-high",
            "External evidence",
            0.9,
            "external",
        )
    )
    ctx.tool_results.append(
        _tool_result(
            "nl_to_sql:1",
            "row1",
            "Structured record evidence",
            None,
            "watchlist",
            published_at=old_date,
        )
    )
    ctx.planner_output = PlannerOutput(
        can_continue_to_aggregation=True,
        reason="Planner completed required tasks.",
        completed_tasks=["search_documents"],
    )

    result = await AggregationHandler().handle(
        ctx,
        FlowStep(
            type=FlowStepType.AGGREGATION,
            settings={
                "allowedSources": ["search_documents"],
                "minRelevanceScore": 0.35,
                "maxEvidence": 1,
                "maxAgeDays": 1,
            },
        ),
    )

    bundle = result.aggregated_evidence
    assert bundle is not None
    assert [item.content for item in bundle.selected_evidence] == [
        "High evidence",
        "Low evidence",
        "External evidence",
        "Structured record evidence",
    ]
    assert bundle.excluded_evidence == []


@pytest.mark.asyncio
async def test_aggregation_excludes_empty_content_and_duplicates() -> None:
    """Aggregation keeps generic evidence hygiene checks."""
    ctx = FlowContext(query="original query")
    ctx.tool_results.append(_tool_result("search_documents:1", "doc1", "Evidence"))
    ctx.tool_results.append(_tool_result("search_documents:2", "doc2", "   "))
    ctx.tool_results.append(_tool_result("search_documents:3", "doc1", "Duplicate"))
    ctx.planner_output = PlannerOutput(
        can_continue_to_aggregation=True,
        reason="Planner completed required tasks.",
        completed_tasks=["search_documents"],
    )

    result = await AggregationHandler().handle(
        ctx,
        FlowStep(type=FlowStepType.AGGREGATION),
    )

    bundle = result.aggregated_evidence
    assert bundle is not None
    assert [item.content for item in bundle.selected_evidence] == ["Evidence"]
    assert [item.reason for item in bundle.excluded_evidence] == [
        "empty_content",
        "duplicate",
    ]


@pytest.mark.asyncio
async def test_aggregation_freshness_contract_excludes_stale_metadata_date() -> None:
    """Freshness is enforced only when a result item declares a contract."""
    ctx = FlowContext(query="original query")
    ctx.tool_results.append(
        _tool_result(
            "watchlist:1",
            "row1",
            "Old watchlist record",
            None,
            "watchlist",
            metadata={
                "as_of_date": (
                    datetime.now(UTC) - timedelta(days=5)
                ).isoformat(),
                "freshness_policy": {
                    "date_field": "as_of_date",
                    "max_age_days": 1,
                },
            },
        )
    )
    ctx.planner_output = PlannerOutput(
        can_continue_to_aggregation=True,
        reason="Planner completed required tasks.",
        completed_tasks=["watchlist"],
    )

    result = await AggregationHandler().handle(
        ctx,
        FlowStep(type=FlowStepType.AGGREGATION),
    )

    bundle = result.aggregated_evidence
    assert bundle is not None
    assert bundle.selected_evidence == []
    assert [item.reason for item in bundle.excluded_evidence] == ["stale"]
    assert "as_of_date older than 1 day" in (
        bundle.excluded_evidence[0].detail or ""
    )


@pytest.mark.asyncio
async def test_aggregation_freshness_contract_excludes_missing_required_date() -> None:
    """A freshness contract can require a date to be present."""
    ctx = FlowContext(query="original query")
    ctx.tool_results.append(
        _tool_result(
            "watchlist:1",
            "row1",
            "Undated watchlist record",
            None,
            "watchlist",
            metadata={
                "freshness_policy": {
                    "date_field": "as_of_date",
                    "max_age_days": 1,
                    "fail_if_missing_date": True,
                },
            },
        )
    )
    ctx.planner_output = PlannerOutput(
        can_continue_to_aggregation=True,
        reason="Planner completed required tasks.",
        completed_tasks=["watchlist"],
    )

    result = await AggregationHandler().handle(
        ctx,
        FlowStep(type=FlowStepType.AGGREGATION),
    )

    bundle = result.aggregated_evidence
    assert bundle is not None
    assert bundle.selected_evidence == []
    assert [item.reason for item in bundle.excluded_evidence] == ["stale"]
    assert bundle.excluded_evidence[0].detail == "missing freshness date: as_of_date"


@pytest.mark.asyncio
async def test_aggregation_freshness_contract_keeps_fresh_metadata_date() -> None:
    """Fresh results with an explicit freshness contract remain selectable."""
    ctx = FlowContext(query="original query")
    ctx.tool_results.append(
        _tool_result(
            "watchlist:1",
            "row1",
            "Fresh watchlist record",
            None,
            "watchlist",
            metadata={
                "as_of_date": datetime.now(UTC).isoformat(),
                "freshness_policy": {
                    "date_field": "as_of_date",
                    "max_age_days": 1,
                },
            },
        )
    )
    ctx.planner_output = PlannerOutput(
        can_continue_to_aggregation=True,
        reason="Planner completed required tasks.",
        completed_tasks=["watchlist"],
    )

    result = await AggregationHandler().handle(
        ctx,
        FlowStep(type=FlowStepType.AGGREGATION),
    )

    bundle = result.aggregated_evidence
    assert bundle is not None
    assert [item.content for item in bundle.selected_evidence] == [
        "Fresh watchlist record"
    ]
    assert bundle.excluded_evidence == []


@pytest.mark.asyncio
async def test_aggregation_requires_planner_output() -> None:
    """Aggregation fails closed if planner output is absent."""
    with pytest.raises(ValueError, match="requires ctx.planner_output"):
        await AggregationHandler().handle(
            FlowContext(query="query"),
            FlowStep(type=FlowStepType.AGGREGATION),
        )
