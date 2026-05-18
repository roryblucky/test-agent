"""Tests for workflow execution contracts."""

from app.api.schemas import QueryResponse
from app.models.workflow import (
    AggregatedEvidence,
    IntentResult,
    NormalizedToolResultItem,
    PlannerOutput,
    PlannerTask,
    ToolCallRecord,
    ToolObservation,
    ToolResultRecord,
)
from app.services.flow_context import FlowContext


def test_intent_result_accepts_legacy_payload() -> None:
    """Old intent outputs remain valid while new fields default safely."""
    result = IntentResult(intent="knowledge_query", confidence=0.93)

    assert result.intent == "knowledge_query"
    assert result.sub_intents == []
    assert result.candidate_skills == []
    assert result.required_data_sources == []
    assert result.needs_clarification is False


def test_flow_context_phase1_fields_default_empty() -> None:
    """New workflow fields are optional and do not affect old context creation."""
    first = FlowContext(query="What is RAG?")
    second = FlowContext(query="Another question")

    assert first.resolved_query is None
    assert first.intent is None
    assert first.active_skills == []
    assert first.tool_observations == []
    assert first.tool_calls == []
    assert first.tool_results == []
    assert first.evidence_store == {}
    assert first.planner_output is None
    assert first.aggregated_evidence is None
    assert first.compliance_review is None

    first.active_skills.append("search")
    assert second.active_skills == []


def test_planner_output_derives_task_lists_from_planned_tasks() -> None:
    """PlannerOutput preserves planner-authored task status as source of truth."""
    output = PlannerOutput(
        can_continue_to_aggregation=False,
        reason="One task is missing.",
        planned_tasks=[
            PlannerTask(
                task_id="search_documents",
                description="Search approved sources.",
                status="completed",
                tool_name="search_documents",
            ),
            PlannerTask(
                task_id="rank_documents",
                description="Rank selected candidates.",
                status="missing",
                tool_name="rank_documents",
                reason="No rankable candidates.",
            ),
            PlannerTask(
                task_id="skip_optional",
                description="Optional task not needed.",
                status="skipped",
            ),
        ],
    )

    assert output.completed_tasks == ["search_documents"]
    assert output.missing_tasks == ["rank_documents"]
    assert output.partial_tasks == []
    assert output.stale_tasks == []
    assert output.failed_tasks == []
    assert output.used_tools == ["search_documents", "rank_documents"]


def test_query_response_keeps_legacy_top_level_shape() -> None:
    """API response keeps existing top-level fields even with Phase 1 context data."""
    ctx = FlowContext(
        query="legacy query",
        session_id="session-1",
    )
    ctx.refined_query = "refined legacy query"
    ctx.intent = IntentResult(
        intent="knowledge_query",
        confidence=0.8,
        candidate_skills=["generic-search"],
    )
    ctx.llm_response = "Legacy answer"
    ctx.tool_results.append(
        ToolResultRecord(
            tool_call_id="search_documents:1",
            tool_name="search_documents",
            source="search_documents",
            normalized_items=[
                NormalizedToolResultItem(
                    item_id="doc1",
                    content="Evidence content",
                )
            ],
        )
    )
    ctx.evidence_store["ev1"] = AggregatedEvidence(
        evidence_id="ev1",
        source="test-source",
        content="Evidence content",
        tool_call_id="search_documents:1",
    )
    ctx.tool_observations.append(
        ToolObservation(
            tool_name="search_documents",
            status="success",
            task_status_hint="completed",
            result_count=1,
        )
    )
    ctx.tool_calls.append(
        ToolCallRecord(
            tool_call_id="search_documents:1",
            tool_name="search_documents",
            input_payload={"query": "legacy query"},
            status="success",
            result_count=1,
        )
    )
    ctx.planner_output = PlannerOutput(
        can_continue_to_aggregation=True,
        reason="Evidence found.",
        completed_tasks=["search_documents"],
    )

    response = QueryResponse.from_flow_context(ctx)
    payload = response.model_dump(by_alias=True)

    assert payload["query"] == "legacy query"
    assert payload["refined_query"] == "refined legacy query"
    assert payload["intent"]["intent"] == "knowledge_query"
    assert payload["answer"] == "Legacy answer"
    assert payload["sessionId"] == "session-1"
    assert "resolved_query" not in payload
    assert "planner_output" not in payload
    assert "tool_observations" not in payload
    assert "evidence_store" not in payload
