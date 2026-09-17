"""Invocation-local progressive access to shared Agent Skills."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.skills.reference_tool import (
    SkillReferenceRegistry,
    SkillReferenceResult,
    load_skill_references,
)
from app.skills.schema import SkillDefinition, SkillSummary


class SkillRuntimeRegistry(SkillReferenceRegistry, Protocol):
    """Tier 2 and Tier 3 portion of the shared Agent Skills registry contract."""

    async def activate(self, tenant_id: str, skill_name: str) -> SkillDefinition | None:
        """Load or return the activated Tier 2 definition."""
        ...


@dataclass
class SkillInvocation:
    """One Specialist's fixed eligible Skill view and activation state."""

    summaries: tuple[SkillSummary, ...]
    _registry: SkillRuntimeRegistry
    _tenant_id: str
    _activated: dict[str, str] = field(default_factory=lambda: dict[str, str]())

    async def activate(self, skill_name: str) -> str:
        """Load and disclose one eligible Skill without rebinding any Tool."""
        if skill_name not in {summary.name for summary in self.summaries}:
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
            activated = definition.instructions
            self._activated[skill_name] = activated
        return activated

    def activation_tool(self) -> Callable[[str], Awaitable[str]]:
        """Expose activation to this invocation's PydanticAI actor only."""

        async def activate_skill(skill_name: str) -> str:
            """Activate an eligible Skill by name for the current task."""
            already_activated = skill_name in self._activated
            activated = await self.activate(skill_name)
            return "" if already_activated else activated

        return activate_skill

    def reference_tool(
        self,
    ) -> Callable[[str], Awaitable[SkillReferenceResult]]:
        """Expose live Tier 3 reference loading to this invocation only."""

        async def load_reference(skill_name: str) -> SkillReferenceResult:
            """Load all current references for a Skill activated in this task."""
            return await load_skill_references(
                registry=self._registry,
                tenant_id=self._tenant_id,
                activated_skill_names=self._activated,
                skill_name=skill_name,
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

    def allowed_tool_ids_for(
        self, specialist_skill_names: Sequence[str]
    ) -> frozenset[str]:
        """Return the Tool ceiling declared by this Specialist's valid Skills."""
        selected = set(specialist_skill_names)
        return frozenset(
            tool_id
            for summary in self.summaries
            if summary.name in selected
            for tool_id in summary.allowed_tools
        )

    def begin_invocation(
        self,
        *,
        specialist_skill_names: Sequence[str],
    ) -> SkillInvocation:
        """Freeze eligible Skills in Specialist declaration order."""
        summaries_by_name = {summary.name: summary for summary in self.summaries}
        eligible_summaries: list[SkillSummary] = []
        for name in specialist_skill_names:
            summary = summaries_by_name.get(name)
            if summary is None:
                continue
            eligible_summaries.append(summary)
        return SkillInvocation(
            summaries=tuple(eligible_summaries),
            _registry=self.registry,
            _tenant_id=self.tenant_id,
        )
