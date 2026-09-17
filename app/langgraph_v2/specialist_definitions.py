"""Startup loading for Tenant-authored Specialist Definitions."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.agents.specialist import create_bound_specialist_actor
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import (
    AgentToolRegistry,
    SpecialistActor,
    SpecialistActorFactory,
    SpecialistCatalog,
    SpecialistRegistration,
    SpecialistTool,
)
from app.langgraph_v2.agent_evidence import SpecialistToolCapture
from app.langgraph_v2.agent_skills import SkillCatalog as RuntimeSkillCatalog
from app.langgraph_v2.agent_skills import SkillInvocation
from app.markdown import parse_frontmatter_and_body
from app.skills.loader import parse_skill_definition
from app.skills.schema import SkillDefinition

logger = logging.getLogger(__name__)

MAX_SPECIALIST_INSTRUCTION_CHARACTERS = 30_000


@dataclass(frozen=True)
class SpecialistSourceDocument:
    """One complete `.agent.md` document owned by a trusted Tenant path."""

    tenant_id: str
    source_identity: str
    content: str


@dataclass(frozen=True)
class SpecialistSourceFailure:
    """One bounded storage failure without document content."""

    tenant_id: str
    source_identity: str
    reason: str


@dataclass(frozen=True)
class SpecialistSourceLoad:
    """Storage result consumed by the catalog implementation."""

    documents: tuple[SpecialistSourceDocument, ...] = ()
    failures: tuple[SpecialistSourceFailure, ...] = ()


@dataclass(frozen=True)
class SkillSourceDocument:
    """One complete `SKILL.md` document owned by a trusted Tenant path."""

    tenant_id: str
    source_identity: str
    content: str


@dataclass(frozen=True)
class SkillSourceFailure:
    """One bounded Skill storage failure without document content."""

    tenant_id: str
    source_identity: str
    reason: str


@dataclass(frozen=True)
class SkillSourceLoad:
    """Skill storage result consumed by the common startup catalog loader."""

    documents: tuple[SkillSourceDocument, ...] = ()
    failures: tuple[SkillSourceFailure, ...] = ()


class SpecialistDefinitionLoader(Protocol):
    """Return Tenant-isolated source documents from one storage adapter."""

    async def load_specialists(self, tenant_id: str) -> SpecialistSourceLoad:
        """Load complete Specialist documents for exactly one known Tenant."""
        ...

    async def load_skills(self, tenant_id: str) -> SkillSourceLoad:
        """Load complete Skill documents for exactly one known Tenant."""
        ...


class TenantModelRegistryProvider(Protocol):
    """Startup-owned Tenant list and approved model registries."""

    @property
    def tenant_ids(self) -> list[str]:
        """Return the trusted Tenant identities configured for this process."""
        ...

    def get_model_registry(self, tenant_id: str, /) -> ModelRegistry:
        """Return the approved model registry for one configured Tenant."""
        ...


@dataclass(frozen=True)
class LocalSpecialistDefinitionLoader:
    """Read the shared Tenant definition tree from a local development root."""

    root: Path

    async def load_specialists(self, tenant_id: str) -> SpecialistSourceLoad:
        """Read complete `.agent.md` documents below one Tenant directory."""
        if (
            not tenant_id
            or Path(tenant_id).parts != (tenant_id,)
            or tenant_id in {".", ".."}
        ):
            return SpecialistSourceLoad(
                failures=(
                    SpecialistSourceFailure(
                        tenant_id=tenant_id,
                        source_identity="tenant-definition-root",
                        reason="invalid-tenant-id",
                    ),
                )
            )
        tenant_directory = self.root / "tenants" / tenant_id
        directory = tenant_directory / "agents"
        if tenant_directory.is_symlink() or directory.is_symlink():
            return SpecialistSourceLoad(
                failures=(
                    SpecialistSourceFailure(
                        tenant_id=tenant_id,
                        source_identity=f"tenants/{tenant_id}/agents",
                        reason="symlink-not-allowed",
                    ),
                )
            )
        if not directory.exists():
            return SpecialistSourceLoad()
        documents: list[SpecialistSourceDocument] = []
        failures: list[SpecialistSourceFailure] = []
        for path in sorted(directory.glob("*.agent.md")):
            source_identity = f"tenants/{tenant_id}/agents/{path.name}"
            if path.is_symlink():
                failures.append(
                    SpecialistSourceFailure(
                        tenant_id=tenant_id,
                        source_identity=source_identity,
                        reason="symlink-not-allowed",
                    )
                )
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                failures.append(
                    SpecialistSourceFailure(
                        tenant_id=tenant_id,
                        source_identity=source_identity,
                        reason="read-failed",
                    )
                )
                continue
            documents.append(
                SpecialistSourceDocument(
                    tenant_id=tenant_id,
                    source_identity=source_identity,
                    content=content,
                )
            )
        return SpecialistSourceLoad(
            documents=tuple(documents),
            failures=tuple(failures),
        )

    async def load_skills(self, tenant_id: str) -> SkillSourceLoad:
        """Read `SKILL.md` files without opening references or other resources."""
        if (
            not tenant_id
            or Path(tenant_id).parts != (tenant_id,)
            or tenant_id in {".", ".."}
        ):
            return SkillSourceLoad(
                failures=(
                    SkillSourceFailure(
                        tenant_id=tenant_id,
                        source_identity="tenant-definition-root",
                        reason="invalid-tenant-id",
                    ),
                )
            )
        tenant_directory = self.root / "tenants" / tenant_id
        directory = tenant_directory / "skills"
        if tenant_directory.is_symlink() or directory.is_symlink():
            return SkillSourceLoad(
                failures=(
                    SkillSourceFailure(
                        tenant_id=tenant_id,
                        source_identity=f"tenants/{tenant_id}/skills",
                        reason="symlink-not-allowed",
                    ),
                )
            )
        if not directory.exists():
            return SkillSourceLoad()
        documents: list[SkillSourceDocument] = []
        failures: list[SkillSourceFailure] = []
        for skill_directory in sorted(directory.iterdir()):
            if not skill_directory.is_dir():
                continue
            source_identity = (
                f"tenants/{tenant_id}/skills/{skill_directory.name}/SKILL.md"
            )
            skill_path = skill_directory / "SKILL.md"
            if skill_directory.is_symlink() or skill_path.is_symlink():
                failures.append(
                    SkillSourceFailure(
                        tenant_id=tenant_id,
                        source_identity=source_identity,
                        reason="symlink-not-allowed",
                    )
                )
                continue
            if not skill_path.is_file():
                continue
            try:
                content = skill_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                failures.append(
                    SkillSourceFailure(
                        tenant_id=tenant_id,
                        source_identity=source_identity,
                        reason="read-failed",
                    )
                )
                continue
            documents.append(
                SkillSourceDocument(
                    tenant_id=tenant_id,
                    source_identity=source_identity,
                    content=content,
                )
            )
        return SkillSourceLoad(
            documents=tuple(documents),
            failures=tuple(failures),
        )


async def load_local_specialist_catalogs(
    tenant_manager: TenantModelRegistryProvider,
    *,
    root: Path,
) -> dict[str, SpecialistCatalog]:
    """Build one immutable local Specialist Catalog per configured Tenant."""
    loader = LocalSpecialistDefinitionLoader(root)
    catalogs: dict[str, SpecialistCatalog] = {}
    for tenant_id in tenant_manager.tenant_ids:
        catalogs[tenant_id] = await build_specialist_catalog(
            loader,
            tenant_id=tenant_id,
            model_registry=tenant_manager.get_model_registry(tenant_id),
        )
    return catalogs


class _SpecialistMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    model_profile: str = Field(alias="model-profile", min_length=1)
    skills: tuple[str, ...]

    @field_validator("id", "description", "model_profile")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("skills")
    @classmethod
    def _unique_skill_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value) or len(set(value)) != len(value):
            raise ValueError("Skill declarations must be unique and non-blank")
        return value


@dataclass(frozen=True)
class _ParsedSpecialistDefinition:
    metadata: _SpecialistMetadata
    instructions: str
    definition_pin: str


async def build_specialist_catalog(
    loader: SpecialistDefinitionLoader,
    *,
    tenant_id: str,
    model_registry: ModelRegistry,
    tool_registry: AgentToolRegistry | None = None,
    tenant_allowed_tool_ids: frozenset[str] = frozenset(),
) -> SpecialistCatalog:
    """Load, validate, and index one immutable Tenant Specialist Catalog."""
    effective_tool_registry = tool_registry or AgentToolRegistry()
    skill_load = await loader.load_skills(tenant_id)
    skill_definitions: list[SkillDefinition] = []
    seen_skill_names: set[str] = set()
    skipped_skills = len(skill_load.failures)
    for failure in skill_load.failures:
        reason = failure.reason if failure.tenant_id == tenant_id else "tenant-mismatch"
        _log_skill_skip(tenant_id, failure.source_identity, reason)
    for document in skill_load.documents:
        if document.tenant_id != tenant_id:
            skipped_skills += 1
            _log_skill_skip(tenant_id, document.source_identity, "tenant-mismatch")
            continue
        try:
            definition = parse_skill_definition(
                document.content,
                tenant_id,
                document.source_identity,
            )
            if definition.metadata.name in seen_skill_names:
                raise ValueError("duplicate-name")
            declared_tool_ids = set(definition.metadata.required_tools) | set(
                definition.metadata.allowed_tools
            )
            if not declared_tool_ids <= effective_tool_registry.registered_ids:
                raise ValueError("unknown-tool")
            RuntimeSkillCatalog(definitions=(definition,))
        except (ValueError, ValidationError) as error:
            skipped_skills += 1
            _log_skill_skip(
                tenant_id,
                document.source_identity,
                _skill_error_reason(error),
            )
            continue
        seen_skill_names.add(definition.metadata.name)
        skill_definitions.append(definition)
    skill_catalog = RuntimeSkillCatalog(definitions=tuple(skill_definitions))
    logger.info(
        "Specialist definitions loaded tenant=%s kind=skill loaded=%d skipped=%d",
        tenant_id,
        len(skill_definitions),
        skipped_skills,
    )

    loaded = await loader.load_specialists(tenant_id)
    registrations: list[SpecialistRegistration] = []
    seen_ids: set[str] = set()
    skipped = len(loaded.failures)
    for failure in loaded.failures:
        reason = failure.reason if failure.tenant_id == tenant_id else "tenant-mismatch"
        _log_skip(tenant_id, failure.source_identity, reason)
    for document in loaded.documents:
        if document.tenant_id != tenant_id:
            skipped += 1
            _log_skip(tenant_id, document.source_identity, "tenant-mismatch")
            continue
        try:
            definition = _parse_definition(document)
        except (ValueError, ValidationError) as error:
            skipped += 1
            _log_skip(
                tenant_id,
                document.source_identity,
                _definition_error_reason(error),
            )
            continue
        if definition.metadata.id in seen_ids:
            skipped += 1
            _log_skip(tenant_id, document.source_identity, "duplicate-id")
            continue
        try:
            model_registry.get_model(definition.metadata.model_profile)
        except KeyError:
            skipped += 1
            _log_skip(tenant_id, document.source_identity, "unknown-model-profile")
            continue
        if not set(definition.metadata.skills) <= skill_catalog.names:
            skipped += 1
            _log_skip(tenant_id, document.source_identity, "invalid-skill")
            continue
        seen_ids.add(definition.metadata.id)
        registrations.append(
            SpecialistRegistration(
                id=definition.metadata.id,
                description=definition.metadata.description,
                actor_factory=_markdown_actor_factory(
                    model_registry=model_registry,
                    model_profile=definition.metadata.model_profile,
                    instructions=definition.instructions,
                ),
                skill_names=definition.metadata.skills,
                definition_pin=definition.definition_pin,
            )
        )
    logger.info(
        "Specialist definitions loaded tenant=%s kind=specialist loaded=%d skipped=%d",
        tenant_id,
        len(registrations),
        skipped,
    )
    return SpecialistCatalog(
        registrations=tuple(registrations),
        tool_registry=effective_tool_registry,
        tenant_allowed_tool_ids=tenant_allowed_tool_ids,
        skill_catalog=skill_catalog,
    )


def _parse_definition(
    document: SpecialistSourceDocument,
) -> _ParsedSpecialistDefinition:
    frontmatter, instructions = parse_frontmatter_and_body(
        document.content,
        source_identity=document.source_identity,
        document_name=".agent.md",
    )
    metadata = _SpecialistMetadata.model_validate(frontmatter)
    if not instructions or len(instructions) > MAX_SPECIALIST_INSTRUCTION_CHARACTERS:
        raise ValueError("invalid-instructions")
    canonical = json.dumps(
        {
            "description": metadata.description,
            "id": metadata.id,
            "instructions": instructions,
            "model-profile": metadata.model_profile,
            "skills": list(metadata.skills),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _ParsedSpecialistDefinition(
        metadata=metadata,
        instructions=instructions,
        definition_pin=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _markdown_actor_factory(
    *,
    model_registry: ModelRegistry,
    model_profile: str,
    instructions: str,
) -> SpecialistActorFactory:
    def build(
        tools: tuple[SpecialistTool, ...],
        tool_capture: SpecialistToolCapture,
        skill_invocation: SkillInvocation | None,
    ) -> SpecialistActor:
        return create_bound_specialist_actor(
            model_registry,
            model_name=model_profile,
            tools=tools,
            tool_capture=tool_capture,
            skill_invocation=skill_invocation,
            tenant_instructions=instructions,
        )

    return build


def _definition_error_reason(error: ValueError | ValidationError) -> str:
    if str(error) == "invalid-instructions":
        return "invalid-instructions"
    return "invalid-metadata"


def _skill_error_reason(error: ValueError | ValidationError) -> str:
    if str(error) == "unknown-tool":
        return "unknown-tool"
    if str(error) == "duplicate-name":
        return "duplicate-name"
    return "invalid-metadata"


def _log_skip(tenant_id: str, source_identity: str, reason: str) -> None:
    logger.warning(
        "Specialist definition skipped tenant=%s source=%s kind=specialist reason=%s",
        tenant_id,
        source_identity,
        reason,
    )


def _log_skill_skip(tenant_id: str, source_identity: str, reason: str) -> None:
    logger.warning(
        "Specialist definition skipped tenant=%s source=%s kind=skill reason=%s",
        tenant_id,
        source_identity,
        reason,
    )
