"""Handler for deterministic evidence aggregation."""

from __future__ import annotations

from app.config.models import FlowStep
from app.core.telemetry import trace_span
from app.models.workflow import AggregatedEvidenceBundle
from app.services.flow_context import FlowContext


class AggregationHandler:
    """Build the synthesis evidence bundle from planner-selected evidence."""

    @trace_span("aggregation")
    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Aggregate evidence from the execution context.

        The planner may only reference evidence IDs. This handler is the bridge
        that loads full evidence items from ``ctx.evidence_store`` for synthesis.
        """
        if ctx.planner_output is None:
            raise ValueError("Aggregation requires ctx.planner_output")

        requested_ids = self._dedupe(ctx.planner_output.evidence_ids)
        evidence = []
        missing_evidence = list(ctx.planner_output.missing_evidence)

        for evidence_id in requested_ids:
            item = ctx.evidence_store.get(evidence_id)
            if item is None:
                missing_evidence.append(evidence_id)
                continue
            evidence.append(item)

        missing_evidence = self._dedupe(missing_evidence)
        stale_evidence = self._dedupe(ctx.planner_output.stale_evidence)
        conflicts = self._dedupe(ctx.planner_output.conflicting_evidence)
        synthesis_allowed = (
            ctx.planner_output.can_synthesize
            and bool(evidence)
            and not missing_evidence
        )
        block_reason = self._block_reason(
            has_evidence=bool(evidence),
            planner_reason=ctx.planner_output.reason,
            missing_evidence=missing_evidence,
            planner_allowed=ctx.planner_output.can_synthesize,
        )

        ctx.aggregated_evidence = AggregatedEvidenceBundle(
            user_query=ctx.query,
            standalone_query=ctx.refined_query or ctx.query,
            tenant_id=str(ctx.metadata.get("tenant_id") or ""),
            intent=ctx.intent.intent if ctx.intent else None,
            active_skills=ctx.planner_output.active_skills,
            evidence=evidence,
            missing_evidence=missing_evidence,
            stale_evidence=stale_evidence,
            conflicts=conflicts,
            synthesis_allowed=synthesis_allowed,
            synthesis_block_reason=block_reason,
        )

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "aggregation",
                {
                    "evidence_count": len(evidence),
                    "missing_evidence_count": len(missing_evidence),
                    "synthesis_allowed": synthesis_allowed,
                },
            )

        return ctx

    @staticmethod
    def _block_reason(
        *,
        has_evidence: bool,
        planner_reason: str,
        missing_evidence: list[str],
        planner_allowed: bool,
    ) -> str | None:
        if not planner_allowed:
            return planner_reason
        if missing_evidence:
            return f"Missing required evidence: {', '.join(missing_evidence)}"
        if not has_evidence:
            return "No evidence available for synthesis."
        return None

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))
