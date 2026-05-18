"""Handler for analysis (observability) step."""

from __future__ import annotations

import time

from app.config.models import FlowStep
from app.core.telemetry import trace_span
from app.services.flow_context import FlowContext


class AnalysisHandler:
    """Handles pipeline analysis and observability."""

    @trace_span("analysis")
    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Aggregate pipeline execution data."""
        pipeline_start = ctx.metadata.get("pipeline_start")
        elapsed = time.time() - pipeline_start if pipeline_start else None
        planner_output = ctx.planner_output
        aggregated_evidence = ctx.aggregated_evidence
        compliance_review = ctx.compliance_review

        analysis = {
            "pipeline_duration_seconds": round(elapsed, 3) if elapsed else None,
            "session_id": ctx.session_id,
            "query": ctx.query,
            "refined_query": ctx.refined_query,
            "answer_length": len(ctx.llm_response) if ctx.llm_response else 0,
            "documents_retrieved": len(ctx.documents),
            "documents_ranked": len(ctx.ranked_documents),
            "is_grounded": (
                ctx.groundedness_result.is_grounded if ctx.groundedness_result else None
            ),
            "token_usage": ctx.metadata.get("coordinator_usage"),
            "streaming_policy": ctx.metadata.get("streaming_policy", "token"),
            "tool_call_count": len(ctx.tool_calls),
            "tool_observation_count": len(ctx.tool_observations),
            "tool_result_count": len(ctx.tool_results),
            "evidence_count": len(ctx.evidence_store),
            "planner_task_count": (
                len(planner_output.planned_tasks) if planner_output else None
            ),
            "planner_can_continue_to_aggregation": (
                planner_output.can_continue_to_aggregation if planner_output else None
            ),
            "planner_completed_task_count": (
                len(planner_output.completed_tasks) if planner_output else None
            ),
            "planner_missing_task_count": (
                len(planner_output.missing_tasks) if planner_output else None
            ),
            "planner_failed_task_count": (
                len(planner_output.failed_tasks) if planner_output else None
            ),
            "aggregated_evidence_count": (
                len(aggregated_evidence.selected_evidence)
                if aggregated_evidence
                else None
            ),
            "synthesis_allowed": (
                aggregated_evidence.synthesis_allowed if aggregated_evidence else None
            ),
            "compliance_passed": (
                compliance_review.passed if compliance_review else None
            ),
            "compliance_violation_count": (
                len(compliance_review.violations) if compliance_review else None
            ),
        }

        ctx.metadata["analysis"] = analysis

        if ctx.emitter:
            await ctx.emitter.emit_step_completed("analysis", analysis)

        return ctx
