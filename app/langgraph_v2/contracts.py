"""Wire contracts owned by the clean-room v2 runtime."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.models.domain import GroundednessResult
from app.models.workflow import CitationReference

GraphEventJournalPolicy = Literal["checkpoint_only", "transport_journal"]
TracerEventType = Literal[
    "step_start",
    "step_completed",
    "token",
    "citations",
    "error",
    "done",
    "stopped",
]


class V2QueryRequest(BaseModel):
    """Legacy-compatible query input with optional additive idempotency data."""

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(min_length=1)
    conversation_id: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("conversation_id", "sessionId"),
            serialization_alias="sessionId",
        ),
    ] = None
    client_request_id: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("client_request_id", "clientRequestId"),
            serialization_alias="clientRequestId",
            min_length=1,
            max_length=128,
            pattern="^[A-Za-z0-9._:-]+$",
        ),
    ] = None


class TracerQueryResponse(BaseModel):
    """Minimal v1-shaped final response emitted by the tracer."""

    query: str
    refined_query: str | None = None
    intent: None = None
    answer: str | None = None
    documents: list[Any] = Field(default_factory=list)
    moderation: dict[str, Any] | None = None
    groundedness: GroundednessResult | None = None
    clarification: None = None
    conversation_id: str = Field(serialization_alias="session_id")
    metadata: dict[str, Any] = Field(default_factory=dict)
    citations: list[CitationReference] = Field(default_factory=list[CitationReference])


class TracerStreamEvent(BaseModel):
    """One additive-sequence SSE event produced by the v2 tracer."""

    event_key: str = Field(min_length=1)
    type: TracerEventType
    sequence: int = Field(ge=1)
    step: str | None = None
    data: Any = None

    def to_sse(self) -> str:
        r"""Serialize the event using the established ``data: JSON\n\n`` frame."""
        payload = self.model_dump(exclude={"event_key"}, exclude_none=True)
        return f"data: {json.dumps(payload, default=str)}\n\n"


class TracerGraphEvent(BaseModel):
    """Internal event carried through Graph State."""

    event_key: str = Field(min_length=1)
    type: TracerEventType
    sequence: int = Field(ge=1)
    step: str | None = None
    data: Any = None
    journal_policy: GraphEventJournalPolicy = "transport_journal"
