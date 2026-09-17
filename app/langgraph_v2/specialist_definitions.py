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
from app.langgraph_v2.agent_skills import SkillInvocation
from app.markdown import parse_frontmatter_and_body

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


class SpecialistDefinitionLoader(Protocol):
    """Return Tenant-isolated source documents from one storage adapter."""

    async def load_specialists(self, tenant_id: str) -> SpecialistSourceLoad:
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
        directory = self.root / "tenants" / tenant_id / "agents"
        if not directory.exists():
            return SpecialistSourceLoad()
        documents: list[SpecialistSourceDocument] = []
        failures: list[SpecialistSourceFailure] = []
        for path in sorted(directory.glob("*.agent.md")):
            source_identity = f"tenants/{tenant_id}/agents/{path.name}"
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
        if definition.metadata.skills:
            skipped += 1
            _log_skip(tenant_id, document.source_identity, "skills-unavailable")
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
        tool_registry=tool_registry or AgentToolRegistry(),
        tenant_allowed_tool_ids=tenant_allowed_tool_ids,
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


def _log_skip(tenant_id: str, source_identity: str, reason: str) -> None:
    logger.warning(
        "Specialist definition skipped tenant=%s source=%s kind=specialist reason=%s",
        tenant_id,
        source_identity,
        reason,
    )
