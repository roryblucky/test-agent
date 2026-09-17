"""Invocation-local progressive access to shared Agent Skills."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.skills.loader import SkillReferenceLoadError
from app.skills.schema import ReferenceDocument, SkillDefinition, SkillSummary

_MAX_SKILL_SUMMARIES = 20


class SkillRuntimeRegistry(Protocol):
    """Tier 2 and Tier 3 portion of the shared Agent Skills registry contract."""

    async def activate(self, tenant_id: str, skill_name: str) -> SkillDefinition | None:
        """Load or return the activated Tier 2 definition."""
        ...

    async def load_activated_references(
        self, tenant_id: str, skill_name: str
    ) -> list[ReferenceDocument]:
        """Resolve the cached Tier 2 identity and read current Tier 3 documents."""
        ...


class SkillPin(BaseModel):
    """Stable identity recorded after one accepted Skill activation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: str | None = Field(default=None, min_length=1)


class ActivatedSkill(BaseModel):
    """Invocation-local full instructions plus their immutable pin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pin: SkillPin
    instructions: str = Field(min_length=1)


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
    reason: Literal["not-activated", "load-failed"]


type SkillReferenceResult = SkillReferencesLoaded | SkillReferencesFailed


def _skill_pin(definition: SkillDefinition) -> SkillPin:
    version_value = definition.metadata.skill_metadata.get("version")
    if version_value is not None and (
        not isinstance(version_value, str) or not version_value.strip()
    ):
        raise ValueError("Skill metadata.version must be a non-blank string")
    canonical = json.dumps(
        {
            "metadata": definition.metadata.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
            "instructions": definition.instructions,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SkillPin(
        name=definition.metadata.name,
        version=version_value.strip() if isinstance(version_value, str) else None,
        content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


@dataclass
class SkillInvocation:
    """One Specialist's fixed eligible Skill view and activation state."""

    summaries: tuple[SkillSummary, ...]
    _eligible: dict[str, SkillSummary]
    _effective_tool_ids: frozenset[str]
    _registry: SkillRuntimeRegistry
    _tenant_id: str
    _activated: dict[str, ActivatedSkill] = field(
        default_factory=lambda: dict[str, ActivatedSkill]()
    )

    @property
    def pins(self) -> tuple[SkillPin, ...]:
        """Return pins in first-activation order without instruction content."""
        return tuple(item.pin for item in self._activated.values())

    @property
    def effective_tool_ids(self) -> frozenset[str]:
        """Expose the unchanged frozen business Tool set for contract tests."""
        return self._effective_tool_ids

    async def activate(self, skill_name: str) -> ActivatedSkill:
        """Load and disclose one eligible Skill without rebinding any Tool."""
        summary = self._eligible.get(skill_name)
        if summary is None:
            raise ValueError("Skill is not eligible")
        activated = self._activated.get(skill_name)
        if activated is None:
            definition = await self._registry.activate(self._tenant_id, skill_name)
            if (
                definition is None
                or definition.tenant_id != self._tenant_id
                or definition.metadata.name != skill_name
            ):
                raise ValueError("Skill is not eligible")
            activated = ActivatedSkill(
                pin=_skill_pin(definition),
                instructions=definition.instructions,
            )
            self._activated[skill_name] = activated
        return activated

    def activation_tool(self) -> Callable[[str], Awaitable[str]]:
        """Expose activation to this invocation's PydanticAI actor only."""

        async def activate_skill(skill_name: str) -> str:
            """Activate an eligible Skill by name for the current task."""
            already_activated = skill_name in self._activated
            activated = await self.activate(skill_name)
            return "" if already_activated else activated.instructions

        return activate_skill

    def reference_tool(
        self,
    ) -> Callable[[str], Awaitable[SkillReferenceResult]]:
        """Expose live Tier 3 reference loading to this invocation only."""

        async def load_reference(skill_name: str) -> SkillReferenceResult:
            """Load all current references for a Skill activated in this task."""
            if skill_name not in self._activated:
                return SkillReferencesFailed(
                    skill_name=skill_name,
                    reason="not-activated",
                )
            try:
                references = await self._registry.load_activated_references(
                    self._tenant_id,
                    skill_name,
                )
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

        return load_reference


@dataclass(frozen=True)
class SkillCatalog:
    """Adapt discovered Agent Skills into fixed invocation-local views."""

    registry: SkillRuntimeRegistry
    tenant_id: str
    summaries: Sequence[SkillSummary]

    def __post_init__(self) -> None:
        summaries = tuple(summary.model_copy(deep=True) for summary in self.summaries)
        object.__setattr__(self, "summaries", summaries)
        names = tuple(summary.name for summary in summaries)
        if len(set(names)) != len(names):
            raise ValueError("Skill summary names must be unique")
        if any(summary.tenant_id != self.tenant_id for summary in summaries):
            raise ValueError("Skill summary Tenant does not match")

    @property
    def names(self) -> frozenset[str]:
        """Return every valid Skill name in this Tenant catalog."""
        return frozenset(summary.name for summary in self.summaries)

    def begin_invocation(
        self,
        *,
        specialist_skill_names: Sequence[str],
        effective_tool_ids: frozenset[str],
    ) -> SkillInvocation:
        """Freeze eligible Skills in Specialist declaration order."""
        summaries_by_name = {summary.name: summary for summary in self.summaries}
        eligible_summaries: list[SkillSummary] = []
        for name in specialist_skill_names:
            summary = summaries_by_name.get(name)
            if summary is None:
                continue
            if not set(summary.required_tools) <= effective_tool_ids:
                continue
            eligible_summaries.append(summary)
            if len(eligible_summaries) == _MAX_SKILL_SUMMARIES:
                break
        eligible = {summary.name: summary for summary in eligible_summaries}
        return SkillInvocation(
            summaries=tuple(eligible_summaries),
            _eligible=eligible,
            _effective_tool_ids=effective_tool_ids,
            _registry=self.registry,
            _tenant_id=self.tenant_id,
        )
