"""Startup loading for Tenant-authored Specialist and Skill definitions."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
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
from app.markdown import is_safe_path_segment, parse_frontmatter_and_body
from app.skills.loader import LocalSkillLoader
from app.skills.registry import TenantSkillRegistry
from app.skills.schema import SkillSummary

logger = logging.getLogger(__name__)

MAX_SPECIALIST_INSTRUCTION_CHARACTERS = 30_000


@dataclass(frozen=True)
class TenantSourceDocument:
    """One complete definition document owned by a trusted Tenant path."""

    tenant_id: str
    source_identity: str
    content: str


@dataclass(frozen=True)
class TenantSourceFailure:
    """One bounded storage failure without document content."""

    tenant_id: str
    source_identity: str
    reason: str


@dataclass(frozen=True)
class TenantSourceLoad:
    """Storage result consumed by the catalog implementation."""

    documents: tuple[TenantSourceDocument, ...] = ()
    failures: tuple[TenantSourceFailure, ...] = ()


class TenantDefinitionLoader(Protocol):
    """Return Tenant-isolated Specialist documents from one storage adapter."""

    async def load_specialists(self, tenant_id: str) -> TenantSourceLoad:
        """Load complete Specialist documents for exactly one known Tenant."""
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
class _LocalDefinitionFile:
    path: Path
    source_identity: str
    guarded_paths: tuple[Path, ...]


@dataclass(frozen=True)
class LocalTenantDefinitionLoader:
    """Read the shared Tenant definition tree from a local development root."""

    root: Path

    async def load_specialists(self, tenant_id: str) -> TenantSourceLoad:
        """Read complete `.agent.md` documents below one Tenant directory."""
        directory, failure = self._definition_directory(tenant_id, "agents")
        if failure is not None:
            return TenantSourceLoad(failures=(failure,))
        if not directory.exists():
            return TenantSourceLoad()
        return self._read_documents(
            tenant_id,
            tuple(
                _LocalDefinitionFile(
                    path=path,
                    source_identity=f"tenants/{tenant_id}/agents/{path.name}",
                    guarded_paths=(path,),
                )
                for path in sorted(directory.glob("*.agent.md"))
            ),
        )

    def _definition_directory(
        self,
        tenant_id: str,
        kind: str,
    ) -> tuple[Path, TenantSourceFailure | None]:
        tenant_directory = self.root / "tenants" / tenant_id
        directory = tenant_directory / kind
        if not is_safe_path_segment(tenant_id):
            return directory, TenantSourceFailure(
                tenant_id=tenant_id,
                source_identity="tenant-definition-root",
                reason="invalid-tenant-id",
            )
        if tenant_directory.is_symlink() or directory.is_symlink():
            return directory, TenantSourceFailure(
                tenant_id=tenant_id,
                source_identity=f"tenants/{tenant_id}/{kind}",
                reason="symlink-not-allowed",
            )
        return directory, None

    def _read_documents(
        self,
        tenant_id: str,
        candidates: Sequence[_LocalDefinitionFile],
    ) -> TenantSourceLoad:
        documents: list[TenantSourceDocument] = []
        failures: list[TenantSourceFailure] = []
        for candidate in candidates:
            if any(path.is_symlink() for path in candidate.guarded_paths):
                failures.append(
                    TenantSourceFailure(
                        tenant_id=tenant_id,
                        source_identity=candidate.source_identity,
                        reason="symlink-not-allowed",
                    )
                )
                continue
            try:
                content = candidate.path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                failures.append(
                    TenantSourceFailure(
                        tenant_id=tenant_id,
                        source_identity=candidate.source_identity,
                        reason="read-failed",
                    )
                )
                continue
            documents.append(
                TenantSourceDocument(
                    tenant_id=tenant_id,
                    source_identity=candidate.source_identity,
                    content=content,
                )
            )
        return TenantSourceLoad(
            documents=tuple(documents),
            failures=tuple(failures),
        )


async def load_local_specialist_catalogs(
    tenant_manager: TenantModelRegistryProvider,
    *,
    root: Path,
    tool_registry: AgentToolRegistry,
    tenant_allowed_tool_ids: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, SpecialistCatalog]:
    """Build one immutable local Specialist Catalog per configured Tenant."""
    loader = LocalTenantDefinitionLoader(root)
    skill_registry = TenantSkillRegistry(LocalSkillLoader(root))
    tenant_tool_policy = tenant_allowed_tool_ids or {}
    catalogs: dict[str, SpecialistCatalog] = {}
    for tenant_id in tenant_manager.tenant_ids:
        catalogs[tenant_id] = await build_specialist_catalog(
            loader,
            tenant_id=tenant_id,
            model_registry=tenant_manager.get_model_registry(tenant_id),
            skill_registry=skill_registry,
            tool_registry=tool_registry,
            tenant_allowed_tool_ids=tenant_tool_policy.get(tenant_id, frozenset()),
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


async def build_specialist_catalog(
    loader: TenantDefinitionLoader,
    *,
    tenant_id: str,
    model_registry: ModelRegistry,
    skill_registry: TenantSkillRegistry | None = None,
    tool_registry: AgentToolRegistry | None = None,
    tenant_allowed_tool_ids: frozenset[str] = frozenset(),
) -> SpecialistCatalog:
    """Load, validate, and index one immutable Tenant Specialist Catalog."""
    effective_tool_registry = tool_registry or AgentToolRegistry()
    if skill_registry is not None:
        await skill_registry.discover(tenant_id)
        discovered_skills = skill_registry.get_summaries(tenant_id)
        discovery_failures = skill_registry.get_discovery_failures(tenant_id)
    else:
        discovered_skills = []
        discovery_failures = ()
    skill_summaries: list[SkillSummary] = []
    seen_skill_names: set[str] = set()
    skipped_skills = len(discovery_failures)
    for summary in discovered_skills:
        if summary.tenant_id != tenant_id:
            skipped_skills += 1
            _log_skill_skip(tenant_id, summary.source_path, "tenant-mismatch")
            continue
        try:
            if summary.name in seen_skill_names:
                raise ValueError("duplicate-name")
            declared_tool_ids = set(summary.allowed_tools)
            if not declared_tool_ids <= effective_tool_registry.registered_ids:
                raise ValueError("unknown-tool")
        except ValueError as error:
            skipped_skills += 1
            _log_skill_skip(
                tenant_id,
                summary.source_path,
                _skill_error_reason(error),
            )
            continue
        seen_skill_names.add(summary.name)
        skill_summaries.append(summary)
    skill_catalog = (
        RuntimeSkillCatalog(
            registry=skill_registry,
            tenant_id=tenant_id,
            summaries=tuple(skill_summaries),
        )
        if skill_registry is not None
        else None
    )
    logger.info(
        "Skill summaries discovered tenant=%s kind=skill loaded=%d skipped=%d",
        tenant_id,
        len(skill_summaries),
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
        skill_names: frozenset[str] = (
            skill_catalog.names if skill_catalog is not None else frozenset()
        )
        if not set(definition.metadata.skills) <= skill_names:
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
    document: TenantSourceDocument,
) -> _ParsedSpecialistDefinition:
    frontmatter, instructions = parse_frontmatter_and_body(
        document.content,
        source_identity=document.source_identity,
        document_name=".agent.md",
    )
    metadata = _SpecialistMetadata.model_validate(frontmatter)
    if not instructions or len(instructions) > MAX_SPECIALIST_INSTRUCTION_CHARACTERS:
        raise ValueError("invalid-instructions")
    return _ParsedSpecialistDefinition(
        metadata=metadata,
        instructions=instructions,
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
        "Skill definition skipped tenant=%s source=%s kind=skill reason=%s",
        tenant_id,
        source_identity,
        reason,
    )
