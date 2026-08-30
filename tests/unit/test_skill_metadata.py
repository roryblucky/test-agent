"""Unit tests for skill metadata parsing."""

import pathlib

import pytest

from app.skills.loader import LocalSkillLoader
from app.skills.schema import SkillRiskLevel


def _write_skill(
    tmp_path: pathlib.Path, tenant_id: str, skill_name: str, content: str
) -> None:
    skill_dir = tmp_path / "tenants" / tenant_id / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


@pytest.mark.asyncio
async def test_skill_metadata_parses_required_tools_constraints_and_risk(
    tmp_path: pathlib.Path,
) -> None:
    """New Phase 4 skill metadata loads from SKILL.md frontmatter."""
    _write_skill(
        tmp_path,
        "tenant-a",
        "market-search",
        """---
name: market-search
description: Search market documents.
risk_level: medium
allowed-tools: search_documents rank_documents
required-tools:
  - search_documents
tool-constraints:
  search_documents:
    source_type: approved
    lookback_days: 7
---
Use approved sources.
""",
    )
    loader = LocalSkillLoader(tmp_path)
    summaries = await loader.discover_skills("tenant-a")
    skill = await loader.activate_skill(summaries[0])

    assert skill.metadata.risk_level == SkillRiskLevel.MEDIUM
    assert skill.metadata.allowed_tools == ["search_documents", "rank_documents"]
    assert skill.metadata.required_tools == ["search_documents"]
    assert skill.metadata.tool_constraints == {
        "search_documents": {
            "source_type": "approved",
            "lookback_days": 7,
        }
    }


@pytest.mark.asyncio
async def test_legacy_skill_metadata_defaults_still_load(
    tmp_path: pathlib.Path,
) -> None:
    """Old skills with only allowed-tools keep safe Phase 4 defaults."""
    _write_skill(
        tmp_path,
        "tenant-a",
        "legacy-search",
        """---
name: legacy-search
description: Legacy search skill.
allowed-tools: search_documents
---
Use search.
""",
    )
    loader = LocalSkillLoader(tmp_path)
    summaries = await loader.discover_skills("tenant-a")
    skill = await loader.activate_skill(summaries[0])

    assert skill.metadata.risk_level == SkillRiskLevel.LOW
    assert skill.metadata.allowed_tools == ["search_documents"]
    assert skill.metadata.required_tools == []
    assert skill.metadata.tool_constraints == {}
