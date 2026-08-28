"""Domain models shared across the application."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.workflow import IntentResult

__all__ = [
    "Document",
    "GroundednessResult",
    "IntentResult",
    "LLMResponse",
    "ModerationResult",
    "RefinedQuestion",
    "TokenUsage",
]


class Document(BaseModel):
    """A retrieved document with content and metadata."""

    id: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float | None = None

    source_url: str | None = None
    source_type: str | None = None
    page_number: int | None = None
    section_title: str | None = None



class TokenUsage(BaseModel):
    """Token usage statistics from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Response from an LLM call."""

    content: str
    model: str
    usage: TokenUsage | None = None


class ModerationResult(BaseModel):
    """Result from a content moderation check."""

    is_flagged: bool
    categories: dict[str, float] = Field(default_factory=dict)
    reason: str | None = None


class GroundednessResult(BaseModel):
    """Result from a groundedness check."""

    is_grounded: bool
    score: float
    details: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class RefinedQuestion(BaseModel):
    """Output of the question refinement agent."""

    refined_query: str
    keywords: list[str] = Field(default_factory=list)
