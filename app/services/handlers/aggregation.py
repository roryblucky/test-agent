"""Handler for deterministic evidence aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config.models import FlowStep
from app.core.telemetry import trace_span
from app.models.workflow import (
    AggregatedEvidence,
    AggregatedEvidenceBundle,
    ExcludedEvidence,
)
from app.services.flow_context import FlowContext


@dataclass(frozen=True)
class AggregationPolicy:
    """Typed aggregation policy normalized from step settings."""

    min_relevance_score: float = 0.35
    max_evidence: int = 8
    allowed_sources: list[str] | None = None
    max_age_days: int | None = None

    @classmethod
    def from_step(cls, step: FlowStep) -> AggregationPolicy:
        """Normalize optional step settings into a typed policy."""
        settings = step.settings or {}
        return cls(
            min_relevance_score=float(
                cls._get(settings, "min_relevance_score", "minRelevanceScore", 0.35)
            ),
            max_evidence=int(cls._get(settings, "max_evidence", "maxEvidence", 8)),
            allowed_sources=cls._list_or_none(
                cls._get(settings, "allowed_sources", "allowedSources", None)
            ),
            max_age_days=cls._int_or_none(
                cls._get(settings, "max_age_days", "maxAgeDays", None)
            ),
        )

    @staticmethod
    def _get(settings: dict[str, Any], snake: str, camel: str, default: Any) -> Any:
        return settings.get(snake, settings.get(camel, default))

    @staticmethod
    def _list_or_none(value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        return int(value) if value is not None else None


class AggregationHandler:
    """Build the synthesis evidence bundle from normalized tool results."""

    @trace_span("aggregation")
    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Aggregate evidence from the execution context."""
        if ctx.planner_output is None:
            raise ValueError("Aggregation requires ctx.planner_output")

        policy = AggregationPolicy.from_step(step)
        selected_evidence, excluded_evidence = self._select_evidence(ctx, policy)
        missing_tasks = self._dedupe(ctx.planner_output.missing_tasks)
        partial_tasks = self._dedupe(ctx.planner_output.partial_tasks)
        stale_tasks = self._dedupe(ctx.planner_output.stale_tasks)
        failed_tasks = self._dedupe(ctx.planner_output.failed_tasks)
        synthesis_allowed = (
            ctx.planner_output.can_continue_to_aggregation
            and bool(selected_evidence)
            and not missing_tasks
            and not partial_tasks
            and not stale_tasks
            and not failed_tasks
        )
        block_reason = self._block_reason(
            has_evidence=bool(selected_evidence),
            planner_reason=ctx.planner_output.reason,
            missing_tasks=missing_tasks,
            partial_tasks=partial_tasks,
            stale_tasks=stale_tasks,
            failed_tasks=failed_tasks,
            planner_allowed=ctx.planner_output.can_continue_to_aggregation,
        )

        ctx.evidence_store = {
            item.evidence_id: item for item in selected_evidence
        }
        ctx.aggregated_evidence = AggregatedEvidenceBundle(
            user_query=ctx.query,
            standalone_query=ctx.refined_query or ctx.query,
            tenant_id=str(ctx.metadata.get("tenant_id") or ""),
            intent=ctx.intent.intent if ctx.intent else None,
            active_skills=ctx.active_skills,
            selected_evidence=selected_evidence,
            missing_tasks=missing_tasks,
            partial_tasks=partial_tasks,
            stale_tasks=stale_tasks,
            failed_tasks=failed_tasks,
            excluded_evidence=excluded_evidence,
            synthesis_allowed=synthesis_allowed,
            synthesis_block_reason=block_reason,
        )

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "aggregation",
                {
                    "selected_evidence_count": len(selected_evidence),
                    "excluded_evidence_count": len(excluded_evidence),
                    "missing_task_count": len(missing_tasks),
                    "failed_task_count": len(failed_tasks),
                    "synthesis_allowed": synthesis_allowed,
                },
            )

        return ctx

    def _select_evidence(
        self,
        ctx: FlowContext,
        policy: AggregationPolicy,
    ) -> tuple[list[AggregatedEvidence], list[ExcludedEvidence]]:
        selected: list[AggregatedEvidence] = []
        excluded: list[ExcludedEvidence] = []
        seen: set[tuple[str, str]] = set()

        for record in ctx.tool_results:
            for item in record.normalized_items:
                content = item.content.strip()
                identity = (record.source, item.item_id)
                if not content:
                    excluded.append(
                        ExcludedEvidence(
                            tool_call_id=record.tool_call_id,
                            item_id=item.item_id,
                            reason="empty_content",
                        )
                    )
                    continue
                if identity in seen:
                    excluded.append(
                        ExcludedEvidence(
                            tool_call_id=record.tool_call_id,
                            item_id=item.item_id,
                            reason="duplicate",
                        )
                    )
                    continue
                if (
                    policy.allowed_sources is not None
                    and record.source not in policy.allowed_sources
                ):
                    excluded.append(
                        ExcludedEvidence(
                            tool_call_id=record.tool_call_id,
                            item_id=item.item_id,
                            reason="source_not_allowed",
                            detail=record.source,
                        )
                    )
                    continue
                if self._is_stale(item.published_at, policy):
                    excluded.append(
                        ExcludedEvidence(
                            tool_call_id=record.tool_call_id,
                            item_id=item.item_id,
                            reason="stale",
                        )
                    )
                    continue
                if (
                    item.score is not None
                    and item.score < policy.min_relevance_score
                ):
                    excluded.append(
                        ExcludedEvidence(
                            tool_call_id=record.tool_call_id,
                            item_id=item.item_id,
                            reason="low_relevance",
                            detail=str(item.score),
                        )
                    )
                    continue

                seen.add(identity)
                evidence = AggregatedEvidence(
                    evidence_id="pending",
                    source=record.source,
                    title=item.title,
                    content=content,
                    url=item.url,
                    published_at=item.published_at,
                    tool_call_id=record.tool_call_id,
                    relevance=self._relevance_from_score(item.score),
                    score=item.score,
                    metadata={**item.metadata, "item_id": item.item_id},
                )
                selected.append(evidence)

        selected.sort(
            key=lambda evidence: evidence.score if evidence.score is not None else 0.0,
            reverse=True,
        )
        overflow = selected[policy.max_evidence :]
        selected = selected[: policy.max_evidence]
        excluded.extend(
            [
                ExcludedEvidence(
                    tool_call_id=item.tool_call_id,
                    item_id=str(item.metadata.get("item_id", item.evidence_id)),
                    reason="max_evidence_exceeded",
                )
                for item in overflow
            ]
        )
        selected = [
            evidence.model_copy(
                update={
                    "evidence_id": (
                        f"evidence:{index}:{evidence.metadata['item_id']}"
                    )
                }
            )
            for index, evidence in enumerate(selected, start=1)
        ]
        return selected, excluded

    @staticmethod
    def _is_stale(published_at: datetime | None, policy: AggregationPolicy) -> bool:
        if published_at is None or policy.max_age_days is None:
            return False
        cutoff = datetime.now(UTC) - timedelta(days=policy.max_age_days)
        return published_at < cutoff

    @staticmethod
    def _block_reason(
        *,
        has_evidence: bool,
        planner_reason: str,
        missing_tasks: list[str],
        partial_tasks: list[str],
        stale_tasks: list[str],
        failed_tasks: list[str],
        planner_allowed: bool,
    ) -> str | None:
        if not planner_allowed:
            return planner_reason
        if failed_tasks:
            return f"Failed required tasks: {', '.join(failed_tasks)}"
        if missing_tasks:
            return f"Missing required tasks: {', '.join(missing_tasks)}"
        if stale_tasks:
            return f"Stale required tasks: {', '.join(stale_tasks)}"
        if partial_tasks:
            return f"Partial required tasks: {', '.join(partial_tasks)}"
        if not has_evidence:
            return "No evidence available for synthesis."
        return None

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _relevance_from_score(score: float | None) -> str:
        if score is None:
            return "medium"
        if score >= 0.75:
            return "high"
        if score < 0.35:
            return "low"
        return "medium"
