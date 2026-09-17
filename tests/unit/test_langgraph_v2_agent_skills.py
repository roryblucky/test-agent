"""Specialist-owned progressive Skill activation contracts."""

import pytest

from app.langgraph_v2.agent_skills import SkillCatalog
from app.skills.schema import ReferenceDocument, SkillDefinition, SkillMetadata


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


@pytest.mark.parametrize("declared_count", [20, 21])
def test_summaries_preserve_specialist_declaration_order_and_cap_at_twenty(
    declared_count: int,
) -> None:
    names = tuple(f"skill-{index:02d}" for index in range(declared_count))
    catalog = SkillCatalog(definitions=tuple(_skill(name) for name in reversed(names)))

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
            invocation.activate("skill-20")


def test_multiple_activations_are_idempotent_ordered_and_do_not_expand_tools() -> None:
    catalog = SkillCatalog(
        definitions=(
            _skill(
                "filing-skill",
                required_tool_ids=("filing-reader",),
                allowed_tool_ids=("filing-reader", "news-reader"),
            ),
            _skill("market-skill", version=None),
            _skill("undeclared-skill"),
        )
    )
    invocation = catalog.begin_invocation(
        specialist_skill_names=("market-skill", "filing-skill"),
        effective_tool_ids=frozenset({"filing-reader"}),
    )

    market = invocation.activate("market-skill")
    filing = invocation.activate("filing-skill")
    activation_tool = invocation.activation_tool()

    assert market.instructions == "market-skill full-instructions-sentinel"
    assert market.pin.version is None
    assert filing.pin.version == "2026.09"
    assert invocation.pins == (market.pin, filing.pin)
    assert invocation.effective_tool_ids == frozenset({"filing-reader"})
    assert invocation.activate("market-skill") == market
    assert activation_tool("market-skill") == ""
    assert invocation.pins == (market.pin, filing.pin)
    with pytest.raises(ValueError, match="Skill is not eligible"):
        invocation.activate("undeclared-skill")
    with pytest.raises(ValueError, match="Skill is not eligible"):
        invocation.activate("missing-skill")


def test_required_tools_filter_summaries_but_allowed_tools_do_not() -> None:
    catalog = SkillCatalog(
        definitions=(
            _skill("requires-news", required_tool_ids=("news-reader",)),
            _skill("suggests-news", allowed_tool_ids=("news-reader",)),
        )
    )

    invocation = catalog.begin_invocation(
        specialist_skill_names=("requires-news", "suggests-news"),
        effective_tool_ids=frozenset({"filing-reader"}),
    )

    assert [summary.name for summary in invocation.summaries] == ["suggests-news"]
    with pytest.raises(ValueError, match="Skill is not eligible"):
        invocation.activate("requires-news")


def test_activation_tool_discloses_only_instructions_once() -> None:
    invocation = SkillCatalog(definitions=(_skill("market-skill"),)).begin_invocation(
        specialist_skill_names=("market-skill",),
        effective_tool_ids=frozenset(),
    )
    activate_skill = invocation.activation_tool()

    first_result = activate_skill("market-skill")
    repeated_result = activate_skill("market-skill")

    assert first_result == "market-skill full-instructions-sentinel"
    assert "content_hash" not in first_result
    assert repeated_result == ""
    assert [pin.name for pin in invocation.pins] == ["market-skill"]


def test_pin_hashes_cached_definition_but_excludes_references_and_storage_identity() -> (
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

    original_pin = (
        SkillCatalog(definitions=(original,))
        .begin_invocation(
            specialist_skill_names=(original.metadata.name,),
            effective_tool_ids=frozenset(),
        )
        .activate(original.metadata.name)
        .pin
    )
    same_pin = (
        SkillCatalog(definitions=(same_content_elsewhere,))
        .begin_invocation(
            specialist_skill_names=(same_content_elsewhere.metadata.name,),
            effective_tool_ids=frozenset(),
        )
        .activate(same_content_elsewhere.metadata.name)
        .pin
    )
    changed_pin = (
        SkillCatalog(definitions=(changed,))
        .begin_invocation(
            specialist_skill_names=(changed.metadata.name,),
            effective_tool_ids=frozenset(),
        )
        .activate(changed.metadata.name)
        .pin
    )

    assert original_pin.version is None
    assert original_pin.content_hash == same_pin.content_hash
    assert original_pin.content_hash != changed_pin.content_hash
