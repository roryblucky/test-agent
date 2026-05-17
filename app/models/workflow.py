"""Typed workflow contracts for platform execution.

These models are domain-neutral. Business-specific vocabulary belongs in
tenant/domain contracts, skill instructions, or tenant config.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    """Normalized evidence item produced by tools or retrieval."""

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


ToolObservationStatus = Literal["success", "empty", "partial", "error"]
DataFreshness = Literal["fresh", "stale", "unknown"]
EvidenceRelevance = Literal["high", "medium", "low", "none", "unknown"]


class ToolObservation(BaseModel):
    """Lightweight tool result intended for planner consumption."""

    tool_name: str
    status: ToolObservationStatus
    evidence_ids: list[str] = Field(default_factory=list)

    summary_for_planner: str | None = None
    entities_found: list[str] = Field(default_factory=list)
    data_freshness: DataFreshness = "unknown"
    relevance: EvidenceRelevance = "unknown"

    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """Audit record for a single tool call."""

    tool_name: str
    input_payload: dict[str, Any]
    status: str

    compiled_filter: str | None = None
    output_evidence_ids: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
    error: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None


class PlannerOutput(BaseModel):
    """Structured output for ``agent:planner`` steps."""

    can_synthesize: bool
    reason: str

    active_skills: list[str] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    required_tools_missing: list[str] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    stale_evidence: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)


class AggregatedEvidenceBundle(BaseModel):
    """Evidence bundle consumed by synthesis steps."""

    user_query: str
    standalone_query: str
    tenant_id: str
    synthesis_allowed: bool

    intent: str | None = None
    active_skills: list[str] = Field(default_factory=list)

    evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    stale_evidence: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    synthesis_block_reason: str | None = None


class ComplianceReviewResult(BaseModel):
    """Structured compliance review result for buffered synthesis."""

    passed: bool
    reason: str | None = None
    violations: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
    safe_response: str | None = None
