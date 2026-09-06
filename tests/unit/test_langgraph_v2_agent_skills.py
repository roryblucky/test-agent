"""Specialist-owned progressive Skill activation contracts."""

import pytest

from app.langgraph_v2.agent_skills import SkillRegistration, SpecialistSkillRegistry


def _skill(
    name: str,
    *,
    required_tool_ids: frozenset[str] = frozenset(),
    allowed_tool_ids: frozenset[str] = frozenset(),
) -> SkillRegistration:
    return SkillRegistration(
        name=name,
        version="2026.09",
        description=f"{name} summary",
        instructions=f"{name} full-instructions-sentinel",
        required_tool_ids=required_tool_ids,
        allowed_tool_ids=allowed_tool_ids,
    )


@pytest.mark.parametrize("registered_count", [20, 21])
def test_effective_summaries_are_trusted_ordered_and_capped_at_twenty(
    registered_count: int,
) -> None:
    names = tuple(f"skill-{index:02d}" for index in range(registered_count))
    registry = SpecialistSkillRegistry(
        registrations=tuple(_skill(name) for name in names),
        tenant_eligible_names=frozenset(names),
        shared_skill_names=frozenset({"skill-00"}),
    )

    invocation = registry.begin_invocation(
        specialist_skill_names=frozenset(names[1:]),
        scope_skill_names=frozenset(names),
        effective_tool_ids=frozenset({"filing-reader"}),
    )

    assert [summary.name for summary in invocation.summaries] == list(names[:20])
    assert all("full-instructions-sentinel" not in summary.description for summary in invocation.summaries)
    if registered_count == 21:
        with pytest.raises(ValueError, match="Skill is not eligible"):
            invocation.activate("skill-20")


def test_activation_rechecks_eligibility_pins_content_and_cannot_expand_tools() -> None:
    registry = SpecialistSkillRegistry(
        registrations=(
            _skill(
                "shared-skill",
                required_tool_ids=frozenset({"filing-reader"}),
                allowed_tool_ids=frozenset({"filing-reader", "news-reader"}),
            ),
            _skill("market-skill"),
            _skill("other-specialist-skill"),
            _skill("cached-only-skill"),
        ),
        tenant_eligible_names=frozenset(
            {"shared-skill", "market-skill", "other-specialist-skill"}
        ),
        shared_skill_names=frozenset({"shared-skill"}),
    )
    invocation = registry.begin_invocation(
        specialist_skill_names=frozenset({"market-skill"}),
        scope_skill_names=frozenset({"shared-skill", "market-skill"}),
        effective_tool_ids=frozenset({"filing-reader"}),
    )

    activated = invocation.activate("shared-skill")

    assert activated.instructions == "shared-skill full-instructions-sentinel"
    assert activated.pin.name == "shared-skill"
    assert activated.pin.version == "2026.09"
    assert len(activated.pin.content_hash) == 64
    assert invocation.pins == (activated.pin,)
    assert invocation.effective_tool_ids == frozenset({"filing-reader"})
    assert invocation.activate("shared-skill") == activated
    assert invocation.pins == (activated.pin,)
    with pytest.raises(ValueError, match="Specialist may activate only one Skill"):
        invocation.activate("market-skill")

    for name in ("other-specialist-skill", "cached-only-skill", "missing-skill"):
        with pytest.raises(ValueError, match="Skill is not eligible"):
            invocation.activate(name)


def test_activation_rejects_required_tool_outside_frozen_tool_set() -> None:
    registry = SpecialistSkillRegistry(
        registrations=(
            _skill("requires-news", required_tool_ids=frozenset({"news-reader"})),
        ),
        tenant_eligible_names=frozenset({"requires-news"}),
        shared_skill_names=frozenset({"requires-news"}),
    )
    invocation = registry.begin_invocation(
        specialist_skill_names=frozenset(),
        scope_skill_names=frozenset({"requires-news"}),
        effective_tool_ids=frozenset({"filing-reader"}),
    )

    with pytest.raises(ValueError, match="Skill required Tool is not eligible"):
        invocation.activate("requires-news")
