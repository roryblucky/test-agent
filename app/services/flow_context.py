"""Pipeline execution context passed between flow steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.domain import (
    Document,
    GroundednessResult,
    IntentResult,
    ModerationResult,
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

    # Populated by flow steps
    refined_query: str | None = None
    intent: IntentResult | None = None
    documents: list[Document] = field(default_factory=list)
    ranked_documents: list[Document] = field(default_factory=list)
    llm_response: str | None = None
    moderation_result: ModerationResult | None = None
    groundedness_result: GroundednessResult | None = None

    # Extensible metadata bucket
    metadata: dict[str, Any] = field(default_factory=dict)

    # Event emitter for SSE streaming (step events + tokens)
    emitter: EventEmitter | None = None
