"""Specialist-owned progressive Skill activation contracts."""

from collections.abc import Sequence
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from app.agents.agent_deps import AgentDeps
from app.agents.tools import load_skill_references_tool
from app.langgraph_v2.agent_skills import SkillCatalog
from app.services.tenant_manager import TenantProviders
from app.skills.loader import (
    SkillDiscoveryResult,
    SkillReferenceLoadError,
    SkillReferenceLoadFailureReason,
)
from app.skills.registry import TenantSkillRegistry
from app.skills.schema import (
    ReferenceDocument,
    SkillDefinition,
    SkillMetadata,
    SkillSummary,
)


def _skill(
    name: str,
    *,
    version: str | None = "2026.09",
    required_tool_ids: tuple[str, ...] = (),
    allowed_tool_ids: tuple[str, ...] = (),
    instructions: str | None = None,
) -> SkillDefinition:
    metadata: dict[str, str] = {}
    if version is not None:
        metadata["version"] = version
    return SkillDefinition(
        metadata=SkillMetadata(
            name=name,
            description=f"{name} summary",
            skill_metadata=metadata,
            required_tools=list(required_tool_ids),
            allowed_tools=list(allowed_tool_ids),
        ),
        instructions=instructions or f"{name} full-instructions-sentinel",
        tenant_id="tenant-a",
        source_path=f"tenants/tenant-a/skills/{name}/SKILL.md",
    )


class _SkillLoader:
    def __init__(self, definitions: Sequence[SkillDefinition]) -> None:
        self.definitions = {item.metadata.name: item for item in definitions}
        self.references: dict[str, list[ReferenceDocument]] = {}

    async def discover_skills(self, tenant_id: str) -> SkillDiscoveryResult:
        return SkillDiscoveryResult(
            summaries=tuple(
                definition.to_summary()
                for definition in self.definitions.values()
                if definition.tenant_id == tenant_id
            )
        )

    async def activate_skill(self, summary: SkillSummary) -> SkillDefinition:
        return self.definitions[summary.name].model_copy(deep=True)

    async def load_references(self, skill: SkillDefinition) -> list[ReferenceDocument]:
        return list(self.references.get(skill.metadata.name, ()))

    async def list_resource_files(self, summary: SkillSummary) -> list[str]:
        del summary
        return []


class _FailingReferenceLoader(_SkillLoader):
    async def load_references(self, skill: SkillDefinition) -> list[ReferenceDocument]:
        del skill
        raise SkillReferenceLoadError(SkillReferenceLoadFailureReason.READ_FAILED)


async def _catalog(
    definitions: Sequence[SkillDefinition],
    *,
    tenant_id: str = "tenant-a",
) -> SkillCatalog:
    registry = TenantSkillRegistry(_SkillLoader(definitions))
    await registry.discover(tenant_id)
    return SkillCatalog(
        registry=registry,
        tenant_id=tenant_id,
        summaries=registry.get_summaries(tenant_id),
    )


def _flowengine_context(
    registry: TenantSkillRegistry | None,
    *,
    activated_skill_names: Sequence[str] = (),
) -> RunContext[AgentDeps]:
    deps = AgentDeps(
        registry=MagicMock(),
        providers=TenantProviders(),
        skill_registry=registry,
        tenant_id="tenant-a",
        activated_skill_names=list(activated_skill_names),
    )
    context = MagicMock(spec=RunContext)
    context.deps = deps
    return cast(RunContext[AgentDeps], context)


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_count", [20, 21])
async def test_summaries_preserve_all_specialist_declarations_in_order(
    declared_count: int,
) -> None:
    names = tuple(f"skill-{index:02d}" for index in range(declared_count))
    catalog = await _catalog(tuple(_skill(name) for name in reversed(names)))

    invocation = catalog.begin_invocation(
        specialist_skill_names=names,
    )

    assert [summary.name for summary in invocation.summaries] == list(names)
    assert all(
        "full-instructions-sentinel" not in summary.description
        for summary in invocation.summaries
    )
    if declared_count == 21:
        assert await invocation.activate("skill-20")


@pytest.mark.asyncio
async def test_multiple_activations_are_idempotent_ordered_and_do_not_expand_tools() -> (
    None
):
    catalog = await _catalog(
        (
            _skill(
                "filing-skill",
                required_tool_ids=("filing-reader",),
                allowed_tool_ids=("filing-reader", "news-reader"),
            ),
            _skill("market-skill", version=None),
            _skill("undeclared-skill"),
        ),
    )
    invocation = catalog.begin_invocation(
        specialist_skill_names=("market-skill", "filing-skill"),
    )

    market = await invocation.activate("market-skill")
    filing = await invocation.activate("filing-skill")
    activation_tool = invocation.activation_tool()

    assert market == "market-skill full-instructions-sentinel"
    assert filing == "filing-skill full-instructions-sentinel"
    assert await invocation.activate("market-skill") == market
    assert await activation_tool("market-skill") == ""
    with pytest.raises(ValueError, match="Skill is not eligible"):
        await invocation.activate("undeclared-skill")
    with pytest.raises(ValueError, match="Skill is not eligible"):
        await invocation.activate("missing-skill")


@pytest.mark.asyncio
async def test_declared_skills_are_eligible_and_allowed_tools_form_the_tool_ceiling() -> (
    None
):
    catalog = await _catalog(
        (
            _skill("requires-news", required_tool_ids=("news-reader",)),
            _skill("suggests-news", allowed_tool_ids=("news-reader",)),
        ),
    )

    invocation = catalog.begin_invocation(
        specialist_skill_names=("requires-news", "suggests-news"),
    )

    assert [summary.name for summary in invocation.summaries] == [
        "requires-news",
        "suggests-news",
    ]
    assert catalog.allowed_tool_ids_for(("requires-news", "suggests-news")) == (
        frozenset({"news-reader"})
    )


@pytest.mark.asyncio
async def test_activation_does_not_recompute_frozen_eligibility_from_live_metadata() -> (
    None
):
    definition = _skill("market-skill", instructions="CURRENT-INSTRUCTIONS")
    loader = _SkillLoader((definition,))
    registry = TenantSkillRegistry(loader)
    await registry.discover("tenant-a")
    catalog = SkillCatalog(
        registry=registry,
        tenant_id="tenant-a",
        summaries=registry.get_summaries("tenant-a"),
    )
    invocation = catalog.begin_invocation(
        specialist_skill_names=("market-skill",),
    )
    loader.definitions["market-skill"] = _skill(
        "market-skill",
        instructions="UPDATED-INSTRUCTIONS",
        required_tool_ids=("new-live-tool",),
    )

    activated = await invocation.activate("market-skill")

    assert activated == "UPDATED-INSTRUCTIONS"


@pytest.mark.asyncio
async def test_activation_rejects_a_definition_from_another_tenant() -> None:
    definition = _skill("market-skill")
    loader = _SkillLoader((definition,))
    registry = TenantSkillRegistry(loader)
    await registry.discover("tenant-a")
    invocation = SkillCatalog(
        registry=registry,
        tenant_id="tenant-a",
        summaries=registry.get_summaries("tenant-a"),
    ).begin_invocation(
        specialist_skill_names=("market-skill",),
    )
    loader.definitions["market-skill"] = definition.model_copy(
        update={"tenant_id": "tenant-b"}
    )

    with pytest.raises(ValueError, match="Skill is not eligible"):
        await invocation.activate("market-skill")


@pytest.mark.asyncio
async def test_activation_tool_discloses_only_instructions_once() -> None:
    invocation = (await _catalog((_skill("market-skill"),))).begin_invocation(
        specialist_skill_names=("market-skill",),
    )
    activate_skill = invocation.activation_tool()

    first_result = await activate_skill("market-skill")
    repeated_result = await activate_skill("market-skill")

    assert first_result == "market-skill full-instructions-sentinel"
    assert "content_hash" not in first_result
    assert repeated_result == ""


@pytest.mark.asyncio
async def test_reference_tool_reads_only_a_skill_activated_in_this_invocation() -> None:
    definition = _skill("market-skill")
    loader = _SkillLoader((definition,))
    loader.references["market-skill"] = [
        ReferenceDocument(
            filename="guide.md",
            content="CURRENT-REFERENCE",
            source_path="trusted-reference-path",
        )
    ]
    registry = TenantSkillRegistry(loader)
    await registry.discover("tenant-a")
    catalog = SkillCatalog(
        registry=registry,
        tenant_id="tenant-a",
        summaries=registry.get_summaries("tenant-a"),
    )
    activated_invocation = catalog.begin_invocation(
        specialist_skill_names=("market-skill",),
    )
    other_invocation = catalog.begin_invocation(
        specialist_skill_names=("market-skill",),
    )
    await activated_invocation.activate("market-skill")

    loaded = await activated_invocation.reference_tool()("market-skill")
    unavailable = await other_invocation.reference_tool()("market-skill")

    assert loaded.model_dump(mode="json") == {
        "status": "loaded",
        "skill_name": "market-skill",
        "references": [{"filename": "guide.md", "content": "CURRENT-REFERENCE"}],
    }
    assert unavailable.model_dump(mode="json") == {
        "status": "failed",
        "skill_name": "market-skill",
        "reason": "not-activated",
    }


@pytest.mark.asyncio
async def test_flowengine_and_specialist_reference_tools_share_the_same_contract() -> (
    None
):
    definition = _skill("market-skill")
    loader = _SkillLoader((definition,))
    loader.references["market-skill"] = [
        ReferenceDocument(
            filename="guide.md",
            content="CURRENT-REFERENCE",
            source_path="trusted-reference-path",
        )
    ]
    registry = TenantSkillRegistry(loader)
    await registry.discover("tenant-a")
    invocation = SkillCatalog(
        registry=registry,
        tenant_id="tenant-a",
        summaries=registry.get_summaries("tenant-a"),
    ).begin_invocation(
        specialist_skill_names=("market-skill",),
    )
    await invocation.activate("market-skill")
    ctx = _flowengine_context(
        registry,
        activated_skill_names=("market-skill",),
    )

    flowengine_result = await load_skill_references_tool(ctx, "market-skill")
    specialist_result = await invocation.reference_tool()("market-skill")

    assert flowengine_result == specialist_result
    assert specialist_result.model_dump(mode="json") == {
        "status": "loaded",
        "skill_name": "market-skill",
        "references": [{"filename": "guide.md", "content": "CURRENT-REFERENCE"}],
    }


@pytest.mark.asyncio
async def test_flowengine_reference_tool_requires_activation_in_its_current_run() -> (
    None
):
    definition = _skill("market-skill")
    loader = _SkillLoader((definition,))
    loader.references["market-skill"] = [
        ReferenceDocument(
            filename="guide.md",
            content="CURRENT-REFERENCE",
            source_path="trusted-reference-path",
        )
    ]
    registry = TenantSkillRegistry(loader)
    await registry.discover("tenant-a")
    await registry.activate("tenant-a", "market-skill")

    result = await load_skill_references_tool(
        _flowengine_context(registry),
        "market-skill",
    )

    assert result.model_dump(mode="json") == {
        "status": "failed",
        "skill_name": "market-skill",
        "reason": "not-activated",
    }


@pytest.mark.asyncio
async def test_flowengine_reference_tool_uses_common_failure_when_not_configured() -> (
    None
):
    result = await load_skill_references_tool(
        _flowengine_context(None),
        "market-skill",
    )

    assert result.model_dump(mode="json") == {
        "status": "failed",
        "skill_name": "market-skill",
        "reason": "not-configured",
    }


@pytest.mark.asyncio
async def test_reference_tool_returns_a_bounded_storage_failure() -> None:
    definition = _skill("market-skill")
    registry = TenantSkillRegistry(_FailingReferenceLoader((definition,)))
    await registry.discover("tenant-a")
    invocation = SkillCatalog(
        registry=registry,
        tenant_id="tenant-a",
        summaries=registry.get_summaries("tenant-a"),
    ).begin_invocation(
        specialist_skill_names=("market-skill",),
    )
    await invocation.activate("market-skill")

    result = await invocation.reference_tool()("market-skill")

    assert result.model_dump(mode="json") == {
        "status": "failed",
        "skill_name": "market-skill",
        "reason": "load-failed",
    }


@pytest.mark.asyncio
async def test_reference_tool_returns_a_successful_empty_reference_set() -> None:
    invocation = (await _catalog((_skill("market-skill"),))).begin_invocation(
        specialist_skill_names=("market-skill",),
    )
    await invocation.activate("market-skill")

    result = await invocation.reference_tool()("market-skill")

    assert result.model_dump(mode="json") == {
        "status": "loaded",
        "skill_name": "market-skill",
        "references": [],
    }
