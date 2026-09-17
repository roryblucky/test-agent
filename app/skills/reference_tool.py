"""Shared model-facing contract for Agent Skills Tier 3 reference loading."""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.skills.loader import SkillReferenceLoadError
from app.skills.schema import ReferenceDocument


class SkillReferenceRegistry(Protocol):
    """Registry seam required by the shared Tier 3 Tool implementation."""

    async def load_activated_references(
        self, tenant_id: str, skill_name: str
    ) -> list[ReferenceDocument]:
        """Resolve the cached Tier 2 identity and read current Tier 3 documents."""
        ...


class SkillReferenceContent(BaseModel):
    """One model-visible reference without its trusted storage identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    content: str


class SkillReferencesLoaded(BaseModel):
    """Complete current reference set returned by Tier 3 loading."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["loaded"] = "loaded"
    skill_name: str = Field(min_length=1)
    references: tuple[SkillReferenceContent, ...]


class SkillReferencesFailed(BaseModel):
    """Bounded model-visible failure for a Tier 3 request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["failed"] = "failed"
    skill_name: str = Field(min_length=1)
    reason: Literal["not-configured", "not-activated", "load-failed"]


type SkillReferenceResult = SkillReferencesLoaded | SkillReferencesFailed


async def load_skill_references(
    *,
    registry: SkillReferenceRegistry | None,
    tenant_id: str,
    activated_skill_names: Collection[str],
    skill_name: str,
) -> SkillReferenceResult:
    """Load live references after current-invocation activation authorization."""
    if registry is None:
        return SkillReferencesFailed(
            skill_name=skill_name,
            reason="not-configured",
        )
    if skill_name not in activated_skill_names:
        return SkillReferencesFailed(
            skill_name=skill_name,
            reason="not-activated",
        )
    try:
        references = await registry.load_activated_references(tenant_id, skill_name)
    except SkillReferenceLoadError:
        return SkillReferencesFailed(
            skill_name=skill_name,
            reason="load-failed",
        )
    return SkillReferencesLoaded(
        skill_name=skill_name,
        references=tuple(
            SkillReferenceContent(
                filename=reference.filename,
                content=reference.content,
            )
            for reference in references
        ),
    )
