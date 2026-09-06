"""Trusted intent policy and immutable Agent Research Scope."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.models.workflow import IntentResult


class SpecialistDescriptor(BaseModel):
    """Compact, prompt-visible description of one eligible Specialist."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class AgentIntentPolicy(BaseModel):
    """Trusted policy owned by platform configuration, never model output."""

    model_config = ConfigDict(frozen=True)

    intent: str = Field(min_length=1)
    description: str = Field(min_length=1)
    specialist_descriptors: tuple[SpecialistDescriptor, ...] = ()
    allowed_tool_ids: frozenset[str] = frozenset()
    allowed_sources: frozenset[str] = frozenset()
    allowed_queries: frozenset[str] = frozenset()
    as_of_date: date = Field(default_factory=date.today)
    max_evidence_age_days: int = Field(default=7, ge=0)


class ResearchScope(BaseModel):
    """Immutable scope resolved from one trusted Intent policy."""

    model_config = ConfigDict(frozen=True)

    intent: str
    specialist_descriptors: tuple[SpecialistDescriptor, ...]
    allowed_tool_ids: frozenset[str]
    allowed_sources: frozenset[str]
    allowed_queries: frozenset[str]
    as_of_date: date
    max_evidence_age_days: int


def resolve_research_scope(
    intent: IntentResult,
    policies: Mapping[str, AgentIntentPolicy],
) -> ResearchScope:
    """Resolve a model-selected Intent through trusted policy only."""
    policy = policies.get(intent.intent)
    if policy is None:
        raise ValueError("Agent Intent is not configured")
    return ResearchScope(
        intent=policy.intent,
        specialist_descriptors=policy.specialist_descriptors,
        allowed_tool_ids=policy.allowed_tool_ids,
        allowed_sources=policy.allowed_sources,
        allowed_queries=policy.allowed_queries,
        as_of_date=policy.as_of_date,
        max_evidence_age_days=policy.max_evidence_age_days,
    )
