"""Unit tests for skill metadata parsing."""

import logging
import pathlib
from dataclasses import dataclass
from typing import Protocol

import pytest

from app.skills.loader import (
    GCSBlobProtocol,
    GCSBucketProtocol,
    GCSSkillLoader,
    LocalSkillLoader,
    SkillReferenceLoadError,
    SkillReferenceLoadFailureReason,
)
from app.skills.registry import TenantSkillRegistry
from app.skills.schema import (
    ReferenceDocument,
    SkillDefinition,
    SkillMetadata,
    SkillRiskLevel,
)


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


def _gcs_skill() -> SkillDefinition:
    return SkillDefinition(
        metadata=SkillMetadata(
            name="market-analysis",
            description="Analyze a market.",
        ),
        instructions="ANALYZE-MARKET",
        tenant_id="tenant-a",
        source_path=(
            "gs://definitions/tenants/tenant-a/skills/market-analysis/SKILL.md"
        ),
    )


class _ReferenceContractHarness(Protocol):
    skill: SkillDefinition

    async def load(
        self, skill: SkillDefinition | None = None
    ) -> list[ReferenceDocument]: ...

    def write(self, filename: str, content: str) -> None: ...

    def write_nested(self, filename: str, content: str) -> None: ...

    def create_empty_directory(self) -> None: ...

    def fail_listing(self) -> None: ...

    def fail_file(self, filename: str) -> None: ...

    def foreign_skill(self) -> SkillDefinition: ...


class _GCSReferenceHarness:
    prefix = "tenants/tenant-a/skills/market-analysis/references/"

    def __init__(self) -> None:
        self.bucket = _GCSBucket()
        self.loader = GCSSkillLoader(
            "definitions",
            client=_GCSClient(self.bucket),
        )
        self.skill = _gcs_skill()

    async def load(
        self, skill: SkillDefinition | None = None
    ) -> list[ReferenceDocument]:
        return await self.loader.load_references(skill or self.skill)

    def _set_blob(self, name: str, content: str | Exception) -> None:
        retained = tuple(blob for blob in self.bucket.blobs if blob.name != name)
        self.bucket.blobs = (*retained, _GCSBlob(name, content))

    def write(self, filename: str, content: str) -> None:
        self._set_blob(f"{self.prefix}{filename}", content)

    def write_nested(self, filename: str, content: str) -> None:
        self._set_blob(f"{self.prefix}nested/{filename}", content)

    def create_empty_directory(self) -> None:
        self._set_blob(self.prefix, "")

    def fail_listing(self) -> None:
        self.bucket.list_error = RuntimeError("SECRET-LIST-ERROR")

    def fail_file(self, filename: str) -> None:
        self._set_blob(
            f"{self.prefix}{filename}",
            RuntimeError("SECRET-READ-ERROR"),
        )

    def foreign_skill(self) -> SkillDefinition:
        return self.skill.model_copy(
            update={
                "source_path": (
                    "gs://definitions/tenants/tenant-b/skills/"
                    "market-analysis/SKILL.md"
                )
            }
        )


class _LocalReferenceHarness:
    def __init__(self, tmp_path: pathlib.Path) -> None:
        _write_skill(
            tmp_path,
            "tenant-a",
            "market-analysis",
            """---
name: market-analysis
description: Analyze a market.
---
ANALYZE-MARKET
""",
        )
        skill_file = (
            tmp_path
            / "tenants"
            / "tenant-a"
            / "skills"
            / "market-analysis"
            / "SKILL.md"
        )
        self.references = skill_file.parent / "references"
        self.loader = LocalSkillLoader(tmp_path)
        self.skill = SkillDefinition(
            metadata=SkillMetadata(
                name="market-analysis",
                description="Analyze a market.",
            ),
            instructions="ANALYZE-MARKET",
            tenant_id="tenant-a",
            source_path=str(skill_file),
        )

    async def load(
        self, skill: SkillDefinition | None = None
    ) -> list[ReferenceDocument]:
        return await self.loader.load_references(skill or self.skill)

    def write(self, filename: str, content: str) -> None:
        self.references.mkdir(exist_ok=True)
        (self.references / filename).write_text(content, encoding="utf-8")

    def write_nested(self, filename: str, content: str) -> None:
        nested = self.references / "nested"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / filename).write_text(content, encoding="utf-8")

    def create_empty_directory(self) -> None:
        self.references.mkdir()

    def fail_listing(self) -> None:
        self.references.write_text("NOT-A-DIRECTORY", encoding="utf-8")

    def fail_file(self, filename: str) -> None:
        self.references.mkdir(exist_ok=True)
        (self.references / filename).write_bytes(b"\xff")

    def foreign_skill(self) -> SkillDefinition:
        return self.skill.model_copy(
            update={
                "source_path": self.skill.source_path.replace(
                    "tenants/tenant-a/",
                    "tenants/tenant-b/",
                )
            }
        )


def _reference_harness(
    adapter: str, tmp_path: pathlib.Path
) -> _ReferenceContractHarness:
    if adapter == "gcs":
        return _GCSReferenceHarness()
    if adapter == "local":
        return _LocalReferenceHarness(tmp_path)
    raise AssertionError(f"Unknown reference adapter: {adapter}")


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


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["local", "gcs"])
async def test_reference_loader_contract_reads_current_direct_files_in_order(
    adapter: str,
    tmp_path: pathlib.Path,
) -> None:
    harness = _reference_harness(adapter, tmp_path)
    harness.write("z-later.md", "LATER")
    harness.write_nested("ignored.md", "NESTED")
    harness.write("a-first.md", "FIRST-V1")

    initial = await harness.load()
    harness.write("a-first.md", "FIRST-V2")
    current = await harness.load()

    assert [(item.filename, item.content) for item in initial] == [
        ("a-first.md", "FIRST-V1"),
        ("z-later.md", "LATER"),
    ]
    assert [(item.filename, item.content) for item in current] == [
        ("a-first.md", "FIRST-V2"),
        ("z-later.md", "LATER"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["local", "gcs"])
async def test_reference_loader_contract_accepts_absent_and_empty_directories(
    adapter: str,
    tmp_path: pathlib.Path,
) -> None:
    harness = _reference_harness(adapter, tmp_path)

    absent = await harness.load()
    harness.create_empty_directory()
    empty = await harness.load()

    assert absent == []
    assert empty == []


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["local", "gcs"])
async def test_reference_loader_contract_reports_listing_failure(
    adapter: str,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = _reference_harness(adapter, tmp_path)
    harness.fail_listing()

    with pytest.raises(SkillReferenceLoadError) as raised:
        await harness.load()

    assert raised.value.reason is SkillReferenceLoadFailureReason.LIST_FAILED
    assert "SECRET-LIST-ERROR" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["local", "gcs"])
async def test_reference_loader_contract_reports_one_file_failure_without_partial_data(
    adapter: str,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = _reference_harness(adapter, tmp_path)
    harness.write("a-valid.md", "VALID")
    harness.fail_file("b-invalid.md")

    with pytest.raises(SkillReferenceLoadError) as raised:
        await harness.load()

    assert raised.value.reason is SkillReferenceLoadFailureReason.READ_FAILED
    assert "SECRET-READ-ERROR" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["local", "gcs"])
async def test_reference_loader_contract_rejects_cross_tenant_identity(
    adapter: str,
    tmp_path: pathlib.Path,
) -> None:
    harness = _reference_harness(adapter, tmp_path)

    with pytest.raises(ValueError, match="outside its Tenant"):
        await harness.load(harness.foreign_skill())


@pytest.mark.asyncio
async def test_local_reference_symlink_is_a_bounded_storage_failure(
    tmp_path: pathlib.Path,
) -> None:
    harness = _LocalReferenceHarness(tmp_path)
    target = tmp_path / "reference-target"
    target.mkdir()
    harness.references.symlink_to(target, target_is_directory=True)

    with pytest.raises(SkillReferenceLoadError) as raised:
        await harness.load()

    assert raised.value.reason is SkillReferenceLoadFailureReason.LIST_FAILED
