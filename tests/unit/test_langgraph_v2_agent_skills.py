"""Specialist-owned progressive Skill activation contracts."""

from collections.abc import Sequence

import pytest

from app.langgraph_v2.agent_skills import SkillCatalog
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


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_count", [20, 21])
async def test_summaries_preserve_specialist_declaration_order_and_cap_at_twenty(
    declared_count: int,
) -> None:
    names = tuple(f"skill-{index:02d}" for index in range(declared_count))
    catalog = await _catalog(tuple(_skill(name) for name in reversed(names)))

    invocation = catalog.begin_invocation(
        specialist_skill_names=names,
        effective_tool_ids=frozenset({"filing-reader"}),
    )

    assert [summary.name for summary in invocation.summaries] == list(names[:20])
    assert all(
        "full-instructions-sentinel" not in summary.description
        for summary in invocation.summaries
    )
    if declared_count == 21:
        with pytest.raises(ValueError, match="Skill is not eligible"):
            await invocation.activate("skill-20")


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
        effective_tool_ids=frozenset({"filing-reader"}),
    )

    market = await invocation.activate("market-skill")
    filing = await invocation.activate("filing-skill")
    activation_tool = invocation.activation_tool()

    assert market.instructions == "market-skill full-instructions-sentinel"
    assert market.pin.version is None
    assert filing.pin.version == "2026.09"
    assert invocation.pins == (market.pin, filing.pin)
    assert invocation.effective_tool_ids == frozenset({"filing-reader"})
    assert await invocation.activate("market-skill") == market
    assert await activation_tool("market-skill") == ""
    assert invocation.pins == (market.pin, filing.pin)
    with pytest.raises(ValueError, match="Skill is not eligible"):
        await invocation.activate("undeclared-skill")
    with pytest.raises(ValueError, match="Skill is not eligible"):
        await invocation.activate("missing-skill")


@pytest.mark.asyncio
async def test_required_tools_filter_summaries_but_allowed_tools_do_not() -> None:
    catalog = await _catalog(
        (
            _skill("requires-news", required_tool_ids=("news-reader",)),
            _skill("suggests-news", allowed_tool_ids=("news-reader",)),
        ),
    )

    invocation = catalog.begin_invocation(
        specialist_skill_names=("requires-news", "suggests-news"),
        effective_tool_ids=frozenset({"filing-reader"}),
    )

    assert [summary.name for summary in invocation.summaries] == ["suggests-news"]
    with pytest.raises(ValueError, match="Skill is not eligible"):
        await invocation.activate("requires-news")


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
        effective_tool_ids=frozenset(),
    )
    loader.definitions["market-skill"] = _skill(
        "market-skill",
        instructions="UPDATED-INSTRUCTIONS",
        required_tool_ids=("new-live-tool",),
    )

    activated = await invocation.activate("market-skill")

    assert activated.instructions == "UPDATED-INSTRUCTIONS"


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
        effective_tool_ids=frozenset(),
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
        effective_tool_ids=frozenset(),
    )
    activate_skill = invocation.activation_tool()

    first_result = await activate_skill("market-skill")
    repeated_result = await activate_skill("market-skill")

    assert first_result == "market-skill full-instructions-sentinel"
    assert "content_hash" not in first_result
    assert repeated_result == ""
    assert [pin.name for pin in invocation.pins] == ["market-skill"]


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
        effective_tool_ids=frozenset(),
    )
    other_invocation = catalog.begin_invocation(
        specialist_skill_names=("market-skill",),
        effective_tool_ids=frozenset(),
    )
    await activated_invocation.activate("market-skill")

    loaded = await activated_invocation.reference_tool()("market-skill")
    unavailable = await other_invocation.reference_tool()("market-skill")

    assert loaded.model_dump(mode="json") == {
        "status": "loaded",
        "skill_name": "market-skill",
        "references": [
            {"filename": "guide.md", "content": "CURRENT-REFERENCE"}
        ],
    }
    assert unavailable.model_dump(mode="json") == {
        "status": "failed",
        "skill_name": "market-skill",
        "reason": "not-activated",
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
        effective_tool_ids=frozenset(),
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
        effective_tool_ids=frozenset(),
    )
    await invocation.activate("market-skill")

    result = await invocation.reference_tool()("market-skill")

    assert result.model_dump(mode="json") == {
        "status": "loaded",
        "skill_name": "market-skill",
        "references": [],
    }


@pytest.mark.asyncio
async def test_pin_hashes_activated_definition_but_excludes_references_and_storage_identity() -> (
    None
):
    original = _skill("filing-analysis", version=None)
    same_content_elsewhere = original.model_copy(
        update={
            "tenant_id": "tenant-b",
            "source_path": "gs://bucket/tenants/tenant-b/skills/renamed/SKILL.md",
            "references": [
                ReferenceDocument(
                    filename="guide.md",
                    content="REFERENCE-SENTINEL",
                    source_path="ignored",
                )
            ],
        }
    )
    changed = _skill(
        "filing-analysis",
        version=None,
        instructions="changed instructions",
    )

    original_invocation = (await _catalog((original,))).begin_invocation(
        specialist_skill_names=(original.metadata.name,),
        effective_tool_ids=frozenset(),
    )
    same_invocation = (
        await _catalog((same_content_elsewhere,), tenant_id="tenant-b")
    ).begin_invocation(
        specialist_skill_names=(same_content_elsewhere.metadata.name,),
        effective_tool_ids=frozenset(),
    )
    changed_invocation = (await _catalog((changed,))).begin_invocation(
        specialist_skill_names=(changed.metadata.name,),
        effective_tool_ids=frozenset(),
    )
    original_pin = (await original_invocation.activate(original.metadata.name)).pin
    same_pin = (
        await same_invocation.activate(same_content_elsewhere.metadata.name)
    ).pin
    changed_pin = (await changed_invocation.activate(changed.metadata.name)).pin

    assert original_pin.version is None
    assert original_pin.content_hash == same_pin.content_hash
    assert original_pin.content_hash != changed_pin.content_hash
