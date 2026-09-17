"""Invocation-local progressive access to startup-cached Agent Skills."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from app.skills.schema import SkillDefinition

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
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: str | None = Field(default=None, min_length=1)


class ActivatedSkill(BaseModel):
    """Invocation-local full instructions plus their immutable pin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pin: SkillPin
    instructions: str = Field(min_length=1)


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
    _eligible: dict[str, SkillDefinition]
    _effective_tool_ids: frozenset[str]
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

    def activate(self, skill_name: str) -> ActivatedSkill:
        """Disclose one eligible cached Skill without rebinding any Tool."""
        definition = self._eligible.get(skill_name)
        if definition is None:
            raise ValueError("Skill is not eligible")
        activated = self._activated.get(skill_name)
        if activated is None:
            activated = ActivatedSkill(
                pin=_skill_pin(definition),
                instructions=definition.instructions,
            )
            self._activated[skill_name] = activated
        return activated

    def activation_tool(self) -> Callable[[str], ActivatedSkill]:
        """Expose activation to this invocation's PydanticAI actor only."""

        def activate_skill(skill_name: str) -> ActivatedSkill:
            """Activate an eligible Skill by name for the current task."""
            return self.activate(skill_name)

        return activate_skill


@dataclass(frozen=True)
class SkillCatalog:
    """Adapt startup-cached Agent Skills into fixed invocation-local views."""

    definitions: Sequence[SkillDefinition]

    def __post_init__(self) -> None:
        definitions = tuple(
            definition.model_copy(deep=True) for definition in self.definitions
        )
        object.__setattr__(self, "definitions", definitions)
        names = tuple(definition.metadata.name for definition in definitions)
        if len(set(names)) != len(names):
            raise ValueError("Skill definition names must be unique")
        for definition in definitions:
            if not definition.instructions.strip():
                raise ValueError("Skill instructions must not be blank")
            _skill_pin(definition)

    @property
    def names(self) -> frozenset[str]:
        """Return every valid Skill name in this Tenant catalog."""
        return frozenset(definition.metadata.name for definition in self.definitions)

    def begin_invocation(
        self,
        *,
        specialist_skill_names: Sequence[str],
        effective_tool_ids: frozenset[str],
    ) -> SkillInvocation:
        """Freeze eligible Skills in Specialist declaration order."""
        definitions_by_name = {
            definition.metadata.name: definition for definition in self.definitions
        }
        eligible_definitions: list[SkillDefinition] = []
        for name in specialist_skill_names:
            definition = definitions_by_name.get(name)
            if definition is None:
                continue
            if not set(definition.metadata.required_tools) <= effective_tool_ids:
                continue
            eligible_definitions.append(definition)
            if len(eligible_definitions) == _MAX_SKILL_SUMMARIES:
                break
        eligible = {
            definition.metadata.name: definition for definition in eligible_definitions
        }
        summaries = tuple(
            SkillSummary(
                name=definition.metadata.name,
                description=definition.metadata.description,
            )
            for definition in eligible_definitions
        )
        return SkillInvocation(
            summaries=summaries,
            _eligible=eligible,
            _effective_tool_ids=effective_tool_ids,
        )
