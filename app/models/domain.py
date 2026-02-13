"""Domain models shared across the application."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A retrieved document with content and metadata."""

    id: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float | None = None


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


class RefinedQuestion(BaseModel):
    """Output of the question refinement agent."""

    refined_query: str
    keywords: list[str] = Field(default_factory=list)


class IntentResult(BaseModel):
    """Output of the intent recognition agent."""

    intent: str
    confidence: float
    sub_intents: list[str] = Field(default_factory=list)
