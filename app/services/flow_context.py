"""Pipeline execution context passed between flow steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from app.models.domain import (
    Document,
    GroundednessResult,
    ModerationResult,
)
from app.models.workflow import (
    AggregatedEvidenceBundle,
    ComplianceReviewResult,
    EvidenceItem,
    IntentResult,
    PlannerOutput,
    ResolvedQuery,
    ToolCallRecord,
    ToolObservation,
)
from app.services.events import EventEmitter


@dataclass
class FlowContext:
    """Mutable context that accumulates results as a pipeline executes.

    Each flow step reads from and writes to fields on this object.
    The ``emitter`` allows the API layer to receive structured SSE events
    (step start/completed, LLM tokens, results) in real time.
    """

    # Input
    query: str

    # Session / conversation continuity
    session_id: str | None = None
    message_history: list[ModelMessage] = field(default_factory=list)
    new_messages: list[ModelMessage] = field(default_factory=list)

    # Resolver / intent
    refined_query: str | None = None
    resolved_query: ResolvedQuery | None = None
    intent: IntentResult | None = None

    # Classic RAG
    documents: list[Document] = field(default_factory=list)
    ranked_documents: list[Document] = field(default_factory=list)

    # Agent / tools / evidence
    active_skills: list[str] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    evidence_store: dict[str, EvidenceItem] = field(default_factory=dict)
    planner_output: PlannerOutput | None = None
    aggregated_evidence: AggregatedEvidenceBundle | None = None

    # Answer / safety
    llm_response: str | None = None
    draft_answer: Any | None = None
    compliance_review: ComplianceReviewResult | None = None
    moderation_result: ModerationResult | None = None
    groundedness_result: GroundednessResult | None = None
    clarification_request: Any | None = None  # Using Any to avoid circular import with schemas

    # Extensible metadata bucket
    metadata: dict[str, Any] = field(default_factory=dict)

    # Event emitter for SSE streaming (step events + tokens)
    emitter: EventEmitter | None = None

    # Global token consumption tracking
    total_usage: RunUsage = field(default_factory=RunUsage)

    def add_usage(self, usage: RunUsage) -> None:
        """Accumulate token usage from a run into the total context tracker."""
        usage_data = getattr(usage, "__dict__", {})
        self.total_usage.requests += usage_data.get(
            "requests", getattr(usage, "requests", 0)
        )
        self.total_usage.input_tokens += usage_data.get(
            "input_tokens", usage_data.get("request_tokens", 0)
        )
        self.total_usage.output_tokens += usage_data.get(
            "output_tokens", usage_data.get("response_tokens", 0)
        )
        self.total_usage.tool_calls += usage_data.get(
            "tool_calls", getattr(usage, "tool_calls", 0)
        )
