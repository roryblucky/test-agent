"""Pydantic models for the agentskills.io-compliant Skill system.

Follows the official agentskills.io specification:
  https://agentskills.io

Three-tier progressive disclosure model:
  Tier 1 - Discovery:   name + description only (~30-50 tokens)
  Tier 2 - Activation:  full SKILL.md instructions (loaded on demand)
  Tier 3 - References:  external reference documents (loaded on demand)

K8s/enterprise adaptation:
  - No ``scripts/`` support (no remote code execution)
  - ``allowed-tools`` references Python functions in BuiltInToolRegistry
  - ``references/`` loaded from GCS, not local filesystem
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import AliasChoices, BaseModel, Field, field_validator


class SkillRiskLevel(StrEnum):
    """Risk level declared by skill metadata."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SkillMetadata(BaseModel):
    """YAML frontmatter from a SKILL.md file.

    Follows the official agentskills.io frontmatter specification.
    All official fields are preserved; enterprise-specific fields are
    prefixed with ``x-`` or added as extension fields.

    Official fields
    ---------------
    name          Required. Lowercase alphanumeric and hyphens, 1-64 chars.
    description   Required. What the skill does AND when to trigger it.
                  Max 1024 chars. This is the Discovery tier text.
    license       Optional. License name or bundled file reference.
    compatibility Optional. Max 500 chars. Environment requirements.
    metadata      Optional. Arbitrary key-value pairs (author, version…).
    allowed-tools Optional. Space-separated list of pre-approved tools.
    required-tools Optional. Space-separated list of tools required by the skill.

    Extension fields (enterprise / K8s)
    ------------------------------------
    risk_level             Skill risk level for runtime policy.
    tool-constraints       Declarative per-tool constraints.
    redirect               Whether tool results bypass LLM (ToolOutput).
    redirect-output-schema Pydantic model name in OUTPUT_MODEL_REGISTRY.
    """

    # ---- Official agentskills.io fields ----
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=1024)
    license: str | None = None
    compatibility: Annotated[str | None, Field(max_length=500)] = None
    # Official spec uses ``metadata`` as a dict of arbitrary kv pairs
    skill_metadata: Annotated[
        dict[str, Any],
        Field(
            validation_alias=AliasChoices("skill_metadata", "metadata"),
            serialization_alias="metadata",
        ),
    ] = Field(default_factory=dict)

    # Official spec: space-separated string or list accepted
    allowed_tools: Annotated[
        list[str],
        Field(
            validation_alias=AliasChoices("allowed_tools", "allowed-tools"),
            serialization_alias="allowed-tools",
        ),
    ] = Field(default_factory=list)

    # ---- Enterprise extension fields ----
    risk_level: Annotated[
        SkillRiskLevel,
        Field(
            validation_alias=AliasChoices("risk_level", "risk-level"),
            serialization_alias="risk_level",
        ),
    ] = SkillRiskLevel.LOW
    required_tools: Annotated[
        list[str],
        Field(
            validation_alias=AliasChoices("required_tools", "required-tools"),
            serialization_alias="required-tools",
        ),
    ] = Field(default_factory=list)
    tool_constraints: Annotated[
        dict[str, dict[str, Any]],
        Field(
            validation_alias=AliasChoices("tool_constraints", "tool-constraints"),
            serialization_alias="tool-constraints",
        ),
    ] = Field(default_factory=dict)
    # If True, ToolOutput is used to return results directly (redirect=True)
    redirect: bool = False
    # Maps to a Pydantic model class in OUTPUT_MODEL_REGISTRY
    redirect_output_schema: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices(
                "redirect_output_schema", "redirect-output-schema"
            ),
            serialization_alias="redirect-output-schema",
        ),
    ] = None

    model_config = {"populate_by_name": True}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Warn on invalid name but load anyway (lenient per agentskills.io spec).

        The spec says: "Name doesn't match the parent directory name → warn,
        load anyway. Name exceeds 64 characters → warn, load anyway."
        Only log warnings; never skip a skill due to name formatting issues.
        """
        import logging
        import re

        _log = logging.getLogger(__name__)
        if len(v) > 64:
            _log.warning(
                f"Skill name '{v}' exceeds 64 characters (spec max). Loading anyway."
            )
        if not re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$|^[a-z0-9]$", v):
            _log.warning(
                f"Skill name '{v}' contains invalid characters per agentskills.io spec "
                "(expected lowercase alphanumeric + hyphens, no leading/trailing hyphens). "
                "Loading anyway for cross-client compatibility."
            )
        return v

    @field_validator("allowed_tools", "required_tools", mode="before")
    @classmethod
    def parse_tool_names(cls, v: Any) -> list[str]:
        """Accept both space-separated string and list."""
        if isinstance(v, str):
            return v.split()
        return v or []

    @field_validator("tool_constraints", mode="before")
    @classmethod
    def parse_tool_constraints(cls, v: Any) -> dict[str, dict[str, Any]]:
        """Normalize missing tool constraints to an empty mapping."""
        return v or {}


class SkillSummary(BaseModel):
    """Tier 1 Discovery object — minimal metadata for skill routing.

    Only ``name`` and ``description`` are loaded at startup.
    This keeps the agent's context window lean when many skills exist.

    ~30-50 tokens per skill, suitable for inclusion in system prompt
    as a capability index.
    """

    name: str
    description: str
    source_path: str  # GCS URI or local path (for lazy full loading)
    tenant_id: str


class ReferenceDocument(BaseModel):
    """A single document from the ``references/`` directory.

    Tier 3 — only loaded during execution when the agent needs
    additional context beyond the SKILL.md instructions.
    """

    filename: str
    content: str
    source_path: str  # Full GCS URI or local path


class SkillDefinition(BaseModel):
    """Tier 2 Activation object — fully loaded skill with instructions.

    Contains the complete SKILL.md content including instructions.
    References are NOT pre-loaded; use ``references`` field only after
    Tier 3 loading.
    """

    metadata: SkillMetadata
    instructions: str  # Markdown body from SKILL.md
    tenant_id: str
    source_path: str  # gs://bucket/path or local path

    # Tier 3: populated lazily by the registry on demand
    references: list[ReferenceDocument] = Field(default_factory=list[ReferenceDocument])

    def to_summary(self) -> SkillSummary:
        """Downgrade to a Tier 1 summary for discovery."""
        return SkillSummary(
            name=self.metadata.name,
            description=self.metadata.description,
            source_path=self.source_path,
            tenant_id=self.tenant_id,
        )
