"""Request and response schemas for the API layer."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.models.domain import (
    Document,
    GroundednessResult,
    IntentResult,
    ModerationResult,
)
from app.models.workflow import CitationReference
from app.services.flow_context import FlowContext


class QuestionAnswerSelector(BaseModel):
    """Option selector for clarification requests."""

    question: str = Field(description="需要用户澄清的具体问题")
    options: list[str] = Field(description="提供给用户的可选快速回答列表")


class ClarificationRequest(BaseModel):
    """Structured request to ask the user for clarification before proceeding."""

    response: str = Field(description="向用户解释为什么需要澄清的回复话术")
    quick_questions: list[QuestionAnswerSelector] | None = Field(
        None, description="结构化的追问列表"
    )


class QueryRequest(BaseModel):
    """Incoming query request body."""

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(..., min_length=1, description="The user's question")
    session_id: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("session_id", "sessionId"),
            serialization_alias="sessionId",
            description="Session ID for multi-turn conversation continuity",
        ),
    ] = None


class QueryResponse(BaseModel):
    """Full (non-streaming) query response."""

    model_config = ConfigDict(populate_by_name=True)

    query: str
    refined_query: str | None = None
    intent: IntentResult | None = None
    answer: str | None = None
    documents: list[Document] = Field(default_factory=list[Document])
    moderation: ModerationResult | None = None
    groundedness: GroundednessResult | None = None
    clarification: ClarificationRequest | None = None
    session_id: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("session_id", "sessionId"),
            serialization_alias="sessionId",
        ),
    ] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    citations: list[CitationReference] = Field(default_factory=list[CitationReference])

    @classmethod
    def from_flow_context(cls, ctx: FlowContext) -> QueryResponse:
        """Create a QueryResponse from a finalized FlowContext."""
        # Attach global token usage to the metadata block
        final_meta = dict(ctx.metadata) if ctx.metadata else {}
        if ctx.total_usage:
            usage_data = getattr(ctx.total_usage, "__dict__", {})
            input_tokens = usage_data.get("input_tokens", 0)
            output_tokens = usage_data.get("output_tokens", 0)
            final_meta["usage"] = {
                "requests": ctx.total_usage.requests,
                "request_tokens": input_tokens,
                "response_tokens": output_tokens,
                "total_tokens": ctx.total_usage.total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        citations_raw = final_meta.pop("citations", [])
        citations = [
            c
            if isinstance(c, CitationReference)
            else CitationReference.model_validate(c)
            for c in citations_raw
        ]

        return cls(
            query=ctx.query,
            refined_query=ctx.refined_query,
            intent=ctx.intent,
            answer=ctx.llm_response,
            documents=ctx.ranked_documents or ctx.documents,
            moderation=ctx.moderation_result,
            groundedness=ctx.groundedness_result,
            clarification=ctx.clarification_request,
            session_id=ctx.session_id,
            metadata=final_meta,
            citations=citations,
        )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    tenants: list[str] = Field(default_factory=list)
