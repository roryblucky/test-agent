"""Request and response schemas for the API layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.domain import (
    Document,
    GroundednessResult,
    IntentResult,
    ModerationResult,
)
from app.services.flow_context import FlowContext


class QueryRequest(BaseModel):
    """Incoming query request body."""

    query: str = Field(..., min_length=1, description="The user's question")
    session_id: str | None = Field(
        None,
        alias="sessionId",
        description="Session ID for multi-turn conversation continuity",
    )


class QueryResponse(BaseModel):
    """Full (non-streaming) query response."""

    query: str
    refined_query: str | None = None
    intent: IntentResult | None = None
    answer: str | None = None
    documents: list[Document] = Field(default_factory=list)
    moderation: ModerationResult | None = None
    groundedness: GroundednessResult | None = None
    session_id: str | None = Field(None, alias="sessionId")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_flow_context(cls, ctx: FlowContext) -> QueryResponse:
        return cls(
            query=ctx.query,
            refined_query=ctx.refined_query,
            intent=ctx.intent,
            answer=ctx.llm_response,
            documents=ctx.ranked_documents or ctx.documents,
            moderation=ctx.moderation_result,
            groundedness=ctx.groundedness_result,
            session_id=ctx.session_id,
            metadata=ctx.metadata,
        )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    tenants: list[str] = Field(default_factory=list)
