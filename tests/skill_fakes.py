"""Small Agent Skills adapters for tests that do not exercise storage."""

from collections.abc import Sequence

from app.langgraph_v2.agent_skills import SkillCatalog
from app.skills.schema import ReferenceDocument, SkillDefinition


class StaticSkillRegistry:
    """Return fixed Tier 2 definitions through the runtime registry seam."""

    def __init__(self, definitions: Sequence[SkillDefinition]) -> None:
        self._references = {
            definition.metadata.name: tuple(
                reference.model_copy(deep=True)
                for reference in definition.references
            )
            for definition in definitions
        }
        self._definitions = {
            definition.metadata.name: definition.model_copy(
                deep=True,
                update={"references": []},
            )
            for definition in definitions
        }

    async def activate(self, tenant_id: str, skill_name: str) -> SkillDefinition | None:
        definition = self._definitions.get(skill_name)
        if definition is None or definition.tenant_id != tenant_id:
            return None
        return definition.model_copy(deep=True)

    async def load_activated_references(
        self, tenant_id: str, skill_name: str
    ) -> list[ReferenceDocument]:
        definition = self._definitions.get(skill_name)
        if definition is None or definition.tenant_id != tenant_id:
            return []
        return [
            reference.model_copy(deep=True)
            for reference in self._references[skill_name]
        ]


def static_skill_catalog(
    definitions: Sequence[SkillDefinition],
) -> SkillCatalog:
    """Build a synchronous fixed catalog for non-storage-focused tests."""
    copied = tuple(definition.model_copy(deep=True) for definition in definitions)
    tenant_ids = {definition.tenant_id for definition in copied}
    if len(tenant_ids) != 1:
        raise ValueError("Static Skill definitions must belong to one Tenant")
    return SkillCatalog(
        registry=StaticSkillRegistry(copied),
        tenant_id=next(iter(tenant_ids)),
        summaries=tuple(definition.to_summary() for definition in copied),
    )
