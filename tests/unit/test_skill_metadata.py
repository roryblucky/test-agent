"""Unit tests for skill metadata parsing."""

import logging
import pathlib
from dataclasses import dataclass

import pytest

from app.skills.loader import (
    GCSBlobProtocol,
    GCSBucketProtocol,
    GCSSkillLoader,
    LocalSkillLoader,
)
from app.skills.registry import TenantSkillRegistry
from app.skills.schema import SkillRiskLevel


@dataclass
class _GCSBlob:
    name: str
    content: str | Exception

    def download_as_text(self) -> str:
        if isinstance(self.content, Exception):
            raise self.content
        return self.content


class _GCSBucket:
    def __init__(
        self,
        blobs: tuple[_GCSBlob, ...] = (),
        *,
        list_error: Exception | None = None,
    ) -> None:
        self.blobs = blobs
        self.list_error = list_error
        self.list_calls = 0

    def list_blobs(self, *, prefix: str) -> list[GCSBlobProtocol]:
        assert prefix.startswith("tenants/")
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return list(self.blobs)

    def blob(self, blob_name: str) -> GCSBlobProtocol:
        return next(blob for blob in self.blobs if blob.name == blob_name)


@dataclass
class _GCSClient:
    value: _GCSBucket

    def bucket(self, bucket_name: str) -> GCSBucketProtocol:
        assert bucket_name == "definitions"
        return self.value


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
    discovery = await loader.discover_skills("tenant-a")
    skill = await loader.activate_skill(discovery.summaries[0])

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
    discovery = await loader.discover_skills("tenant-a")
    skill = await loader.activate_skill(discovery.summaries[0])

    assert skill.metadata.risk_level == SkillRiskLevel.LOW
    assert skill.metadata.allowed_tools == ["search_documents"]
    assert skill.metadata.required_tools == []
    assert skill.metadata.tool_constraints == {}


@pytest.mark.asyncio
async def test_gcs_discovery_preserves_valid_siblings_and_bounds_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prefix = "tenants/tenant-a/skills"
    bucket = _GCSBucket(
        (
            _GCSBlob(
                f"{prefix}/valid/SKILL.md",
                """---
name: valid
description: Valid Skill.
---
VALID-INSTRUCTIONS
""",
            ),
            _GCSBlob(
                f"{prefix}/broken/SKILL.md",
                """---
name: broken
description: [SECRET-DOCUMENT-CONTENT
---
SECRET-INSTRUCTIONS
""",
            ),
            _GCSBlob(
                f"{prefix}/unreadable/SKILL.md",
                RuntimeError("SECRET-DOWNLOAD-ERROR"),
            ),
        )
    )
    registry = TenantSkillRegistry(
        GCSSkillLoader("definitions", client=_GCSClient(bucket))
    )
    caplog.set_level(logging.INFO)

    await registry.discover("tenant-a")

    assert [summary.name for summary in registry.get_summaries("tenant-a")] == ["valid"]
    assert [
        (failure.source_path, failure.reason)
        for failure in registry.get_discovery_failures("tenant-a")
    ] == [
        (
            "gs://definitions/tenants/tenant-a/skills/broken/SKILL.md",
            "invalid-definition",
        ),
        (
            "gs://definitions/tenants/tenant-a/skills/unreadable/SKILL.md",
            "unreadable-definition",
        ),
    ]
    assert "SECRET-DOCUMENT-CONTENT" not in caplog.text
    assert "SECRET-INSTRUCTIONS" not in caplog.text
    assert "SECRET-DOWNLOAD-ERROR" not in caplog.text


@pytest.mark.asyncio
async def test_gcs_discovery_reports_invalid_tenant_and_list_failure_without_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bucket = _GCSBucket(list_error=RuntimeError("SECRET-LIST-ERROR"))
    registry = TenantSkillRegistry(
        GCSSkillLoader("definitions", client=_GCSClient(bucket))
    )
    caplog.set_level(logging.INFO)

    await registry.discover("../tenant-b")
    await registry.discover("tenant-a")

    assert [
        failure.reason for failure in registry.get_discovery_failures("../tenant-b")
    ] == ["invalid-tenant-id"]
    assert [
        failure.reason for failure in registry.get_discovery_failures("tenant-a")
    ] == ["list-failed"]
    assert bucket.list_calls == 1
    assert "SECRET-LIST-ERROR" not in caplog.text
