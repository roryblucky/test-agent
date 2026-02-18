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
        }

        ctx.metadata["analysis"] = analysis

        if ctx.emitter:
            await ctx.emitter.emit_step_completed("analysis", analysis)

        return ctx
