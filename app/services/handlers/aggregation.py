"""Handler for deterministic evidence aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from app.config.models import FlowStep
from app.core.telemetry import trace_span
from app.models.workflow import (
    AggregatedEvidence,
    AggregatedEvidenceBundle,
    EvidenceRelevance,
    ExcludedEvidence,
    NormalizedToolResultItem,
)
from app.services.citation_extractor import safe_parse_page_number
from app.services.flow_context import FlowContext


@dataclass(frozen=True)
class FreshnessPolicy:
    """Optional freshness contract attached to an individual result item."""

    date_field: str = "published_at"
    max_age_days: int | None = None
    fail_if_missing_date: bool = False

    @classmethod
    def from_item(cls, item: NormalizedToolResultItem) -> FreshnessPolicy | None:
        """Read an explicit freshness contract from item metadata."""
        raw_policy = item.metadata.get("freshness_policy") or item.metadata.get(
            "freshnessPolicy"
        )
        if not isinstance(raw_policy, dict):
            return None

        return cls(
            date_field=str(
                cls._get(raw_policy, "date_field", "dateField", "published_at")
            ),
            max_age_days=cls._int_or_none(
                cls._get(raw_policy, "max_age_days", "maxAgeDays", None)
            ),
            fail_if_missing_date=bool(
                cls._get(
                    raw_policy,
                    "fail_if_missing_date",
                    "failIfMissingDate",
                    False,
                )
            ),
        )

    @staticmethod
    def _get(settings: dict[str, Any], snake: str, camel: str, default: Any) -> Any:
        return settings.get(snake, settings.get(camel, default))

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        return int(value) if value is not None else None


class AggregationHandler:
    """Build the synthesis evidence bundle from normalized tool results."""

    @trace_span("aggregation")
    async def handle(self, ctx: FlowContext, _step: FlowStep) -> FlowContext:
        """Aggregate evidence from the execution context."""
        if ctx.planner_output is None:
            raise ValueError("Aggregation requires ctx.planner_output")

        selected_evidence, excluded_evidence = self._select_evidence(ctx)
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
    ) -> tuple[list[AggregatedEvidence], list[ExcludedEvidence]]:
        selected: list[AggregatedEvidence] = []
        excluded: list[ExcludedEvidence] = []
        seen: set[tuple[str, str]] = set()

        for record in ctx.tool_results:
            for item in record.normalized_items:
                content = item.content.strip() if item.content else ""
                identity = (record.source, item.item_id)

                if item.item_type == "document_chunk" and not content:
                    excluded.append(
                        ExcludedEvidence(
                            tool_call_id=record.tool_call_id,
                            item_id=item.item_id,
                            reason="empty_content",
                        )
                    )
                    continue
                if item.item_type == "structured_record" and not item.structured_facts:
                    excluded.append(
                        ExcludedEvidence(
                            tool_call_id=record.tool_call_id,
                            item_id=item.item_id,
                            reason="empty_structured_facts",
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
                freshness_exclusion = self._freshness_exclusion(
                    record.tool_call_id,
                    item,
                )
                if freshness_exclusion is not None:
                    excluded.append(freshness_exclusion)
                    continue

                seen.add(identity)
                evidence = AggregatedEvidence(
                    evidence_id="pending",
                    source=record.source,
                    title=item.title,
                    evidence_type=item.item_type,
                    content=content or None,
                    structured_facts=item.structured_facts,
                    original_item_id=item.item_id,
                    url=item.url,
                    published_at=item.published_at,
                    tool_call_id=record.tool_call_id,
                    relevance=self._relevance_for_item(item),
                    score=item.score,
                    metadata={**item.metadata, "item_id": item.item_id},
                )
                selected.append(evidence)

        if selected and all(evidence.score is not None for evidence in selected):
            selected.sort(
                key=lambda evidence: evidence.score
                if evidence.score is not None
                else 0.0,
                reverse=True,
            )

        selected = [
            evidence.model_copy(
                update={
                    "evidence_id": f"evidence:{index}:{evidence.metadata['item_id']}",
                    "citation_index": index,
                    "source_type": evidence.metadata.get("source_type") or evidence.source,
                    "page_number": safe_parse_page_number(
                        evidence.metadata.get("page_number")
                        or evidence.metadata.get("pageNumber")
                    ),
                    "section": evidence.title,
                }
            )
            for index, evidence in enumerate(selected, start=1)
        ]
        return selected, excluded

    def _freshness_exclusion(
        self,
        tool_call_id: str,
        item: NormalizedToolResultItem,
    ) -> ExcludedEvidence | None:
        policy = FreshnessPolicy.from_item(item)
        if policy is None:
            return None

        item_date = self._item_date(item, policy)
        if item_date is None:
            if policy.fail_if_missing_date:
                return ExcludedEvidence(
                    tool_call_id=tool_call_id,
                    item_id=item.item_id,
                    reason="stale",
                    detail=f"missing freshness date: {policy.date_field}",
                )
            return None

        if policy.max_age_days is None:
            return None

        cutoff = datetime.now(UTC) - timedelta(days=policy.max_age_days)
        if item_date < cutoff:
            return ExcludedEvidence(
                tool_call_id=tool_call_id,
                item_id=item.item_id,
                reason="stale",
                detail=(
                    f"{policy.date_field} older than "
                    f"{policy.max_age_days} day freshness window"
                ),
            )
        return None

    @staticmethod
    def _item_date(
        item: NormalizedToolResultItem,
        policy: FreshnessPolicy,
    ) -> datetime | None:
        if policy.date_field == "published_at":
            return AggregationHandler._normalize_datetime(item.published_at)

        for raw_value in (
            getattr(item, policy.date_field, None),
            item.metadata.get(policy.date_field),
            item.structured_facts.get(policy.date_field),
        ):
            if raw_value is not None:
                return AggregationHandler._normalize_datetime(raw_value)
        return None

    @staticmethod
    def _normalize_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value.astimezone(UTC)
        if isinstance(value, date):
            return datetime.combine(value, time.min, tzinfo=UTC)
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                try:
                    return datetime.combine(
                        date.fromisoformat(value),
                        time.min,
                        tzinfo=UTC,
                    )
                except ValueError:
                    return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        return None

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
    def _relevance_for_item(item: NormalizedToolResultItem) -> EvidenceRelevance:
        raw_relevance = item.metadata.get("relevance")
        if raw_relevance in ("high", "medium", "low"):
            return raw_relevance
        if item.item_type == "structured_record":
            return "high"
        return AggregationHandler._relevance_from_score(item.score)

    @staticmethod
    def _relevance_from_score(score: float | None) -> EvidenceRelevance:
        if score is None:
            return "medium"
        if score >= 0.75:
            return "high"
        if score < 0.35:
            return "low"
        return "medium"
