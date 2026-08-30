"""Wire contracts owned by the clean-room v2 runtime."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.models.domain import GroundednessResult
from app.models.workflow import CitationReference

LinearEventType = Literal[
    "step_start",
    "step_completed",
    "progress",
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
            validation_alias=AliasChoices("sessionId", "conversation_id", "session_id"),
            serialization_alias="sessionId",
        ),
    ] = None
    client_request_id: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("clientRequestId", "client_request_id"),
            serialization_alias="clientRequestId",
            min_length=1,
            max_length=128,
            pattern="^[A-Za-z0-9._:-]+$",
        ),
    ] = None


class LinearQueryResponse(BaseModel):
    """Minimal v1-shaped final response emitted by the Linear Graph."""

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


class LiveStreamEvent(BaseModel):
    """One public event emitted by the live request-owned Graph stream."""

    type: LinearEventType
    step: str | None = None
    data: Any = None
    checkpoint_terminal: bool = Field(default=False, exclude=True)

    def to_stream_payload(self) -> dict[str, Any]:
        """Include private delivery metadata for LangGraph's custom channel."""
        payload = self.model_dump(exclude_none=True)
        if self.checkpoint_terminal:
            payload["checkpoint_terminal"] = True
        return payload

    def to_sse(self) -> str:
        r"""Serialize the event using the established ``data: JSON\n\n`` frame."""
        payload = self.model_dump(exclude_none=True)
        return f"data: {json.dumps(payload, default=str)}\n\n"
