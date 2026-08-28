"""Wire contracts owned by the clean-room v2 runtime."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.workflow import CitationReference


class V2QueryRequest(BaseModel):
    """Legacy-compatible query input with optional additive idempotency data."""

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(min_length=1)
    conversation_id: str | None = Field(default=None, alias="sessionId")
    client_request_id: str | None = Field(
        default=None,
        alias="clientRequestId",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )


class TracerQueryResponse(BaseModel):
    """Minimal v1-shaped final response emitted by the tracer."""

    query: str
    refined_query: str | None = None
    intent: None = None
    answer: str | None = None
    documents: list[Any] = Field(default_factory=list)
    moderation: None = None
    groundedness: None = None
    clarification: None = None
    conversation_id: str = Field(serialization_alias="session_id")
    metadata: dict[str, Any] = Field(default_factory=dict)
    citations: list[CitationReference] = Field(default_factory=list)


class TracerStreamEvent(BaseModel):
    """One additive-sequence SSE event produced by the v2 tracer."""

    event_key: str = Field(min_length=1)
    type: Literal[
        "step_start", "step_completed", "token", "citations", "error", "done"
    ]
    sequence: int = Field(ge=1)
    step: str | None = None
    data: Any = None

    def to_sse(self) -> str:
        r"""Serialize the event using the established ``data: JSON\n\n`` frame."""
        payload = self.model_dump(exclude={"event_key"}, exclude_none=True)
        return f"data: {json.dumps(payload, default=str)}\n\n"
