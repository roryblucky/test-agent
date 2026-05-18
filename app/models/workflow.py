"""Typed workflow contracts for platform execution.

These models are domain-neutral. Business-specific vocabulary belongs in
tenant/domain contracts, skill instructions, or tenant config.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field


class ResolvedQuery(BaseModel):
    """Structured query resolution output for complex workflows."""

    original_query: str
    standalone_query: str
    language: str = "zh-CN"

    subject_text: str | None = None
    subject_type: str = "unknown"
    normalized_subject_name: str | None = None
    aliases: list[str] = Field(default_factory=list)

    time_range_text: str | None = None
    lookback_days: int | None = None

    ambiguity: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntentResult(BaseModel):
    """Domain-neutral intent classification result."""

    intent: str
    confidence: float
    sub_intents: list[str] = Field(default_factory=list)

    candidate_skills: list[str] = Field(default_factory=list)
    required_data_sources: list[str] = Field(default_factory=list)

    needs_clarification: bool = False
    clarification_question: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    """Legacy normalized evidence item kept for classic RAG compatibility."""

    id: str
    source: str
    content: str
    retrieved_at: datetime

    source_type: str | None = None
    title: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ToolObservationStatus = Literal["success", "empty", "partial", "stale", "error"]
TaskStatusHint = Literal["completed", "missing", "partial", "stale", "failed"]
EvidenceRelevance = Literal["high", "medium", "low"]


class ToolObservation(BaseModel):
    """Lightweight tool execution signal intended for planner consumption.

    This model deliberately excludes evidence IDs, snippets, summaries, and
    raw result content. Full normalized results are stored separately on the
    workflow context for aggregation.
    """

    tool_name: str
    status: ToolObservationStatus
    task_status_hint: TaskStatusHint
    result_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None


class ToolCallRecord(BaseModel):
    """Audit record for a single tool call."""

    tool_call_id: str
    tool_name: str
    task_id: str | None = None
    input_payload: dict[str, Any]
    status: ToolObservationStatus
    result_count: int = 0

    compiled_filter: str | None = None
    latency_ms: int | None = None
    error_code: str | None = None
    error: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None


class NormalizedToolResultItem(BaseModel):
    """A normalized tool result item stored for later evidence processing."""

    item_id: str
    content: str

    title: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResultRecord(BaseModel):
    """Normalized tool result payload stored in workflow context."""

    tool_call_id: str
    tool_name: str
    source: str

    task_id: str | None = None
    normalized_items: list[NormalizedToolResultItem] = Field(default_factory=list)
    raw_result_ref: str | None = None


class PlannerOutput(BaseModel):
    """Structured output for ``agent:planner`` steps."""

    completed_tasks: list[str] = Field(default_factory=list)
    missing_tasks: list[str] = Field(default_factory=list)
    partial_tasks: list[str] = Field(default_factory=list)
    stale_tasks: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)

    used_tools: list[str] = Field(default_factory=list)
    can_continue_to_aggregation: bool = Field(
        validation_alias=AliasChoices(
            "can_continue_to_aggregation",
            "can_synthesize",
        )
    )
    reason: str


class AggregatedEvidence(BaseModel):
    """Evidence selected by the aggregation node for synthesis."""

    evidence_id: str
    source: str
    content: str
    tool_call_id: str

    title: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    relevance: EvidenceRelevance = "medium"
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AggregatedEvidenceBundle(BaseModel):
    """Evidence bundle consumed by synthesis steps."""

    user_query: str
    standalone_query: str
    tenant_id: str
    synthesis_allowed: bool

    intent: str | None = None
    active_skills: list[str] = Field(default_factory=list)

    selected_evidence: list[AggregatedEvidence] = Field(
        default_factory=list,
        validation_alias=AliasChoices("selected_evidence", "evidence"),
    )
    missing_tasks: list[str] = Field(default_factory=list)
    partial_tasks: list[str] = Field(default_factory=list)
    stale_tasks: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)
    excluded_evidence: list[dict[str, Any]] = Field(default_factory=list)

    synthesis_block_reason: str | None = None

    @property
    def evidence(self) -> list[AggregatedEvidence]:
        """Compatibility alias for older callers."""
        return self.selected_evidence


class ComplianceReviewResult(BaseModel):
    """Structured compliance review result for buffered synthesis."""

    passed: bool
    reason: str | None = None
    violations: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
    safe_response: str | None = None
