"""GCS-backed and local Skill loader following agentskills.io structure.

Directory layout (agentskills.io standard)::

    tenants/{tenant_id}/skills/{skill-name}/
    ├── SKILL.md          # Required: frontmatter + instructions
    ├── references/       # Optional: additional context documents
    │   ├── schema.md
    │   └── examples.md
    └── assets/           # Optional: templates, data (not loaded by default)

GCS paths::

    gs://{bucket}/tenants/{tenant_id}/skills/{skill-name}/SKILL.md
    gs://{bucket}/tenants/{tenant_id}/skills/{skill-name}/references/*.md

K8s adaptation: ``scripts/`` directory is NOT loaded or executed.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import yaml

from app.skills.schema import (
    ReferenceDocument,
    SkillDefinition,
    SkillMetadata,
    SkillSummary,
)

logger = logging.getLogger(__name__)


class _GCSBlob(Protocol):
    name: str

    def download_as_text(self) -> str: ...


class _GCSBucket(Protocol):
    def list_blobs(self, *, prefix: str) -> list[_GCSBlob]: ...

    def blob(self, blob_name: str) -> _GCSBlob: ...


class _GCSClient(Protocol):
    def bucket(self, bucket_name: str) -> _GCSBucket: ...


# ---------------------------------------------------------------------------
# Internal parser
# ---------------------------------------------------------------------------


def _parse_frontmatter_and_body(
    content: str, source_path: str
) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter and Markdown body from a SKILL.md file.

    Args:
        content: Raw file content.
        source_path: Path string for error messages.

    Returns:
        Tuple of (frontmatter_dict, instructions_body).

    Raises:
        ValueError: If the frontmatter delimiters are missing.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        raise ValueError(
            f"Invalid SKILL.md at {source_path}: "
            "expected YAML frontmatter between --- delimiters"
        )
    loaded: object = yaml.safe_load(match.group(1))
    if loaded is None:
        frontmatter: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        frontmatter = cast(dict[str, Any], loaded)
    else:
        raise ValueError(
            f"Invalid SKILL.md at {source_path}: frontmatter must be a mapping"
        )
    instructions = match.group(2).strip()
    return frontmatter, instructions


def _parse_skill_md(content: str, tenant_id: str, source_path: str) -> SkillDefinition:
    """Parse a complete SKILL.md file (Tier 2 Activation object).

    Args:
        content: Raw SKILL.md content.
        tenant_id: Owning tenant ID.
        source_path: GCS URI or local path for traceability.

    Returns:
        SkillDefinition with metadata + instructions (no references yet).
    """
    frontmatter, instructions = _parse_frontmatter_and_body(content, source_path)
    return SkillDefinition(
        metadata=SkillMetadata(**frontmatter),
        instructions=instructions,
        tenant_id=tenant_id,
        source_path=source_path,
    )


def _parse_skill_summary(
    content: str, tenant_id: str, source_path: str
) -> SkillSummary:
    """Parse only the frontmatter for Tier 1 Discovery (lightweight).

    Only reads name + description — ~30-50 tokens per skill.

    Args:
        content: Raw SKILL.md content.
        tenant_id: Owning tenant ID.
        source_path: GCS URI or local path.

    Returns:
        SkillSummary with name + description only.
    """
    frontmatter, _ = _parse_frontmatter_and_body(content, source_path)
    return SkillSummary(
        name=frontmatter.get("name", ""),
        description=frontmatter.get("description", ""),
        source_path=source_path,
        tenant_id=tenant_id,
    )


# ---------------------------------------------------------------------------
# Loader protocol
# ---------------------------------------------------------------------------


class SkillLoaderProtocol(Protocol):
    """Protocol for skill loaders (GCS, local, test doubles)."""

    async def discover_skills(self, tenant_id: str) -> list[SkillSummary]:
        """Tier 1: Load only name + description for all skills."""
        ...

    async def activate_skill(self, summary: SkillSummary) -> SkillDefinition:
        """Tier 2: Load full SKILL.md instructions for a specific skill."""
        ...

    async def load_references(self, skill: SkillDefinition) -> list[ReferenceDocument]:
        """Tier 3: Load all files from the skill's references/ directory."""
        ...

    async def list_resource_files(self, summary: SkillSummary) -> list[str]:
        """List available reference filenames for a skill without loading content.

        Returns filenames only (e.g. ['schema.md', 'examples.md']).
        Used to populate <skill_resources> in the activation response.
        No scripts/ support per K8s enterprise policy.
        """
        ...


# ---------------------------------------------------------------------------
# GCS implementation
# ---------------------------------------------------------------------------


class GCSSkillLoader:
    """Load skills from a GCP Storage bucket.

    Progressive disclosure:
    - ``discover_skills``  → lists SKILL.md blobs, reads frontmatter only
    - ``activate_skill``   → downloads and fully parses one SKILL.md
    - ``load_references``  → downloads all files under references/

    Usage::

        loader = GCSSkillLoader("my-config-bucket")
        summaries = await loader.discover_skills("acme")       # Tier 1
        skill = await loader.activate_skill(summaries[0])     # Tier 2
        refs = await loader.load_references(skill)            # Tier 3
    """

    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        self._client: _GCSClient | None = None

    def _get_client(self) -> _GCSClient:
        if self._client is None:
            storage_module = importlib.import_module("google.cloud.storage")
            client_type = cast(Callable[[], object], storage_module.Client)
            self._client = cast(_GCSClient, client_type())
        return self._client

    def _get_bucket(self) -> _GCSBucket:
        return self._get_client().bucket(self.bucket_name)

    async def discover_skills(self, tenant_id: str) -> list[SkillSummary]:
        """Tier 1: Scan GCS for SKILL.md files, parse frontmatter only."""
        bucket = self._get_bucket()
        prefix = f"tenants/{tenant_id}/skills/"
        summaries: list[SkillSummary] = []

        try:
            blobs = list(bucket.list_blobs(prefix=prefix))
            for blob in blobs:
                if blob.name.endswith("SKILL.md"):
                    source_path = f"gs://{self.bucket_name}/{blob.name}"
                    try:
                        content = blob.download_as_text()
                        summary = _parse_skill_summary(content, tenant_id, source_path)
                        summaries.append(summary)
                        logger.debug(f"[{tenant_id}] Discovered skill: {summary.name}")
                    except Exception:
                        logger.exception(
                            f"[{tenant_id}] Failed to parse skill summary at "
                            f"{source_path}"
                        )
        except Exception:
            logger.exception(
                f"[{tenant_id}] Failed to list skills from "
                f"gs://{self.bucket_name}/{prefix}"
            )

        logger.info(
            f"[{tenant_id}] Discovered {len(summaries)} skill(s): "
            f"{[s.name for s in summaries]}"
        )
        return summaries

    async def activate_skill(self, summary: SkillSummary) -> SkillDefinition:
        """Tier 2: Download and fully parse a specific SKILL.md."""
        # Derive the blob path from the source_path URI
        # source_path = gs://{bucket}/{blob_name}
        blob_name = summary.source_path.removeprefix(f"gs://{self.bucket_name}/")
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()

        skill = _parse_skill_md(content, summary.tenant_id, summary.source_path)
        logger.info(
            f"[{summary.tenant_id}] Activated skill: {skill.metadata.name} "
            f"(tools: {skill.metadata.allowed_tools})"
        )
        return skill

    async def load_references(self, skill: SkillDefinition) -> list[ReferenceDocument]:
        """Tier 3: Download all files from references/ for a skill.

        GCS path: ``tenants/{tenant_id}/skills/{name}/references/*``
        """
        # Derive the skill directory prefix from the SKILL.md path
        # source_path = gs://{bucket}/tenants/{tid}/skills/{name}/SKILL.md
        skill_dir = summary_source_to_prefix(skill.source_path, self.bucket_name)
        refs_prefix = f"{skill_dir}references/"

        bucket = self._get_bucket()
        documents: list[ReferenceDocument] = []

        try:
            blobs = list(bucket.list_blobs(prefix=refs_prefix))
            for blob in blobs:
                if blob.name == refs_prefix:
                    continue  # Skip the directory placeholder blob
                try:
                    content = blob.download_as_text()
                    filename = blob.name.split("/")[-1]
                    source_path = f"gs://{self.bucket_name}/{blob.name}"
                    documents.append(
                        ReferenceDocument(
                            filename=filename,
                            content=content,
                            source_path=source_path,
                        )
                    )
                    logger.debug(f"[{skill.tenant_id}] Loaded reference: {filename}")
                except Exception:
                    logger.exception(
                        f"[{skill.tenant_id}] Failed to load reference: {blob.name}"
                    )
        except Exception:
            logger.exception(
                f"[{skill.tenant_id}] Failed to list references at "
                f"gs://{self.bucket_name}/{refs_prefix}"
            )

        logger.info(
            f"[{skill.tenant_id}] Loaded {len(documents)} reference(s) "
            f"for skill '{skill.metadata.name}'"
        )
        return documents

    async def list_resource_files(self, summary: SkillSummary) -> list[str]:
        """List reference filenames for a skill without downloading content.

        Scans ``references/*`` only. No scripts/ per K8s policy.
        Returns bare filenames sorted alphabetically.
        """
        skill_dir = summary_source_to_prefix(summary.source_path, self.bucket_name)
        refs_prefix = f"{skill_dir}references/"
        bucket = self._get_bucket()
        filenames: list[str] = []

        try:
            blobs = list(bucket.list_blobs(prefix=refs_prefix))
            for blob in blobs:
                if blob.name == refs_prefix:
                    continue
                filenames.append(blob.name.split("/")[-1])
        except Exception:
            logger.debug(
                f"[{summary.tenant_id}] No references found at "
                f"gs://{self.bucket_name}/{refs_prefix}"
            )

        return sorted(filenames)


def summary_source_to_prefix(source_path: str, bucket_name: str) -> str:
    """Convert a SKILL.md GCS URI to its parent directory blob prefix.

    Example::
        "gs://bucket/tenants/acme/skills/vector-search/SKILL.md"
        → "tenants/acme/skills/vector-search/"
    """
    blob_name = source_path.removeprefix(f"gs://{bucket_name}/")
    # Drop the SKILL.md filename → parent dir
    parts = blob_name.rsplit("/", 1)
    return f"{parts[0]}/" if len(parts) > 1 else ""


# ---------------------------------------------------------------------------
# Local filesystem implementation (dev / testing)
# ---------------------------------------------------------------------------


class LocalSkillLoader:
    """Load skills from local filesystem (dev / testing).

    Mirrors the GCS loader's three-tier API exactly so the registry
    and agent handler work identically in both environments.

    Directory layout::

        {base_dir}/
          tenants/{tenant_id}/skills/{skill-name}/
            SKILL.md
            references/
              doc1.md
              doc2.txt
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def _tenant_skills_dir(self, tenant_id: str) -> Path:
        return self.base_dir / "tenants" / tenant_id / "skills"

    async def discover_skills(self, tenant_id: str) -> list[SkillSummary]:
        """Tier 1: Scan local directory, parse frontmatter only."""
        tenant_dir = self._tenant_skills_dir(tenant_id)
        summaries: list[SkillSummary] = []

        if not tenant_dir.exists():
            logger.warning(f"[{tenant_id}] Skill directory not found: {tenant_dir}")
            return summaries

        for skill_dir in sorted(tenant_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            try:
                content = skill_file.read_text(encoding="utf-8")
                summary = _parse_skill_summary(content, tenant_id, str(skill_file))
                summaries.append(summary)
                logger.debug(f"[{tenant_id}] Discovered skill: {summary.name}")
            except Exception:
                logger.exception(
                    f"[{tenant_id}] Failed to parse skill summary at {skill_file}"
                )

        logger.info(
            f"[{tenant_id}] Discovered {len(summaries)} local skill(s): "
            f"{[s.name for s in summaries]}"
        )
        return summaries

    async def activate_skill(self, summary: SkillSummary) -> SkillDefinition:
        """Tier 2: Read and fully parse a SKILL.md from local path."""
        skill_file = Path(summary.source_path)
        content = skill_file.read_text(encoding="utf-8")
        skill = _parse_skill_md(content, summary.tenant_id, str(skill_file))
        logger.info(f"[{summary.tenant_id}] Activated skill: {skill.metadata.name}")
        return skill

    async def load_references(self, skill: SkillDefinition) -> list[ReferenceDocument]:
        """Tier 3: Load all files from references/ subdirectory."""
        skill_file = Path(skill.source_path)
        refs_dir = skill_file.parent / "references"
        documents: list[ReferenceDocument] = []

        if not refs_dir.exists():
            return documents

        for ref_file in sorted(refs_dir.iterdir()):
            if not ref_file.is_file():
                continue
            try:
                content = ref_file.read_text(encoding="utf-8")
                documents.append(
                    ReferenceDocument(
                        filename=ref_file.name,
                        content=content,
                        source_path=str(ref_file),
                    )
                )
                logger.debug(f"[{skill.tenant_id}] Loaded reference: {ref_file.name}")
            except Exception:
                logger.exception(
                    f"[{skill.tenant_id}] Failed to load reference: {ref_file}"
                )

        logger.info(
            f"[{skill.tenant_id}] Loaded {len(documents)} reference(s) "
            f"for skill '{skill.metadata.name}'"
        )
        return documents

    async def list_resource_files(self, summary: SkillSummary) -> list[str]:
        """List reference filenames without loading content.

        Scans ``references/`` subdirectory only. No scripts/ per K8s policy.
        Returns bare filenames sorted alphabetically.
        """
        skill_file = Path(summary.source_path)
        refs_dir = skill_file.parent / "references"

        if not refs_dir.exists():
            return []

        return sorted(f.name for f in refs_dir.iterdir() if f.is_file())
