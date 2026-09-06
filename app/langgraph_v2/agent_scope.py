"""Trusted intent policy and immutable Agent Research Scope."""

from __future__ import annotations

from collections.abc import Mapping

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


class ResearchScope(BaseModel):
    """Immutable scope resolved from one trusted Intent policy."""

    model_config = ConfigDict(frozen=True)

    intent: str
    specialist_descriptors: tuple[SpecialistDescriptor, ...]


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
    )
