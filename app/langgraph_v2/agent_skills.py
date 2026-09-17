"""Trusted progressive Skill selection for one Specialist invocation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

_MAX_SKILL_SUMMARIES = 20


class SkillSummary(BaseModel):
    """Prompt-visible discovery metadata without full instructions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SkillPin(BaseModel):
    """Stable identity recorded after one accepted Skill activation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class SkillReference(BaseModel):
    """One full invocation-local Skill reference document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ActivatedSkill(BaseModel):
    """Invocation-local full instructions plus their immutable pin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pin: SkillPin
    instructions: str = Field(min_length=1)
    references: tuple[SkillReference, ...] = ()


@dataclass(frozen=True)
class SkillRegistration:
    """One trusted Skill definition kept behind the registry seam."""

    name: str
    version: str
    description: str
    instructions: str
    required_tool_ids: frozenset[str] = frozenset()
    allowed_tool_ids: frozenset[str] = frozenset()
    references: tuple[SkillReference, ...] = ()

    @property
    def pin(self) -> SkillPin:
        """Return the immutable identity of this exact instruction content."""
        canonical = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "instructions": self.instructions,
                "required_tool_ids": sorted(self.required_tool_ids),
                "allowed_tool_ids": sorted(self.allowed_tool_ids),
                "references": [
                    reference.model_dump(mode="json") for reference in self.references
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return SkillPin(
            name=self.name,
            version=self.version,
            content_hash=hashlib.sha256(canonical.encode()).hexdigest(),
        )


@dataclass
class SkillInvocation:
    """One Specialist's fixed eligible Skill view and activation state."""

    summaries: tuple[SkillSummary, ...]
    _eligible: dict[str, SkillRegistration]
    _effective_tool_ids: frozenset[str]
    _activated: dict[str, ActivatedSkill] = field(
        default_factory=lambda: dict[str, ActivatedSkill]()
    )

    @property
    def pins(self) -> tuple[SkillPin, ...]:
        """Return activation pins in first-activation order without instructions."""
        return tuple(item.pin for item in self._activated.values())

    @property
    def effective_tool_ids(self) -> frozenset[str]:
        """Expose the unchanged frozen business Tool set for contract tests."""
        return self._effective_tool_ids

    def activate(self, skill_name: str) -> ActivatedSkill:
        """Load one eligible Skill without granting or rebinding a Tool."""
        skill = self._eligible.get(skill_name)
        if skill is None:
            raise ValueError("Skill is not eligible")
        if self._activated and skill_name not in self._activated:
            raise ValueError("Specialist may activate only one Skill")
        if not skill.required_tool_ids <= self._effective_tool_ids:
            raise ValueError("Skill required Tool is not eligible")
        activated = self._activated.get(skill_name)
        if activated is None:
            activated = ActivatedSkill(
                pin=skill.pin,
                instructions=skill.instructions,
                references=skill.references,
            )
            self._activated[skill_name] = activated
        return activated

    def activation_tool(self) -> Callable[[str], ActivatedSkill]:
        """Expose activation to this invocation's PydanticAI actor only."""

        def activate_skill(skill_name: str) -> ActivatedSkill:
            """Activate one eligible Skill by name for the current task."""
            return self.activate(skill_name)

        return activate_skill


@dataclass(frozen=True)
class SpecialistSkillRegistry:
    """Create bounded invocation-local Skill views from trusted registration."""

    registrations: Sequence[SkillRegistration]
    tenant_eligible_names: frozenset[str]
    shared_skill_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        names = tuple(skill.name for skill in self.registrations)
        if len(set(names)) != len(names):
            raise ValueError("Skill registration names must be unique")

    def begin_invocation(
        self,
        *,
        specialist_skill_names: frozenset[str],
        scope_skill_names: frozenset[str],
        effective_tool_ids: frozenset[str],
    ) -> SkillInvocation:
        """Freeze one Specialist's eligible discovery view in registration order."""
        selectable_names = self.shared_skill_names | specialist_skill_names
        eligible_names = (
            selectable_names & self.tenant_eligible_names & scope_skill_names
        )
        eligible_registrations = tuple(
            skill for skill in self.registrations if skill.name in eligible_names
        )[:_MAX_SKILL_SUMMARIES]
        eligible = {skill.name: skill for skill in eligible_registrations}
        summaries = tuple(
            SkillSummary(name=skill.name, description=skill.description)
            for skill in eligible_registrations
        )
        return SkillInvocation(
            summaries=summaries,
            _eligible=eligible,
            _effective_tool_ids=effective_tool_ids,
        )
