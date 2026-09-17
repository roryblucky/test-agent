"""Tenant Specialist Definition Loader and Catalog contract coverage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from app.agents.specialist import SPECIALIST_INSTRUCTIONS, SPECIALIST_SECURITY_GUARDS
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import (
    AcceptedBatch,
    AgentToolRegistry,
    BatchContribution,
    DispatchBatch,
    EvidenceToolRegistration,
    SpecialistAttempt,
    SpecialistCatalog,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistResult,
    SpecialistTaskInput,
    SpecialistTool,
    TaskProposal,
    TaskSpecialistDefinitionPin,
    TaskSucceeded,
    accept_initial_dispatch,
    execute_specialist,
    promote_batch,
)
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    EvidenceInvocationContext,
    SpecialistToolCapture,
)
from app.langgraph_v2.agent_scope import SpecialistDescriptor
from app.langgraph_v2.agent_skills import SkillInvocation
from app.langgraph_v2.specialist_definitions import (
    LocalSpecialistDefinitionLoader,
    SpecialistSourceDocument,
    SpecialistSourceLoad,
    build_specialist_catalog,
    load_local_specialist_catalogs,
)


class _ModelRegistry:
    def __init__(self, *approved_profiles: str) -> None:
        self.approved_profiles = frozenset(approved_profiles)

    def get_model(self, name: str) -> object:
        if name not in self.approved_profiles:
            raise KeyError(name)
        return object()


class _ExecutableModelRegistry(_ModelRegistry):
    def __init__(self) -> None:
        super().__init__("specialist")
        self.instructions: list[str] = []
        self.tools: list[tuple[SpecialistTool, ...]] = []

    def create_agent(self, name: str, **kwargs: Any) -> object:
        assert name == "specialist"
        instructions = kwargs.get("instructions")
        assert isinstance(instructions, str)
        self.instructions.append(instructions)
        tools = kwargs.get("tools")
        assert isinstance(tools, tuple)
        self.tools.append(cast(tuple[SpecialistTool, ...], tools))
        return Agent(
            TestModel(
                call_tools=[],
                custom_output_args={
                    "summary": "Markdown market finding",
                    "evidence_ids": [],
                },
            ),
            **kwargs,
        )


class _TenantManager:
    def __init__(self, registries: dict[str, _ModelRegistry]) -> None:
        self.registries = registries

    @property
    def tenant_ids(self) -> list[str]:
        return list(self.registries)

    def get_model_registry(self, tenant_id: str) -> ModelRegistry:
        return cast(ModelRegistry, self.registries[tenant_id])


class _ContractActor:
    def __init__(
        self,
        *,
        inputs: list[SpecialistTaskInput],
    ) -> None:
        self.inputs = inputs

    async def run(
        self,
        input: SpecialistTaskInput,
        **_kwargs: object,
    ) -> SpecialistAttempt:
        self.inputs.append(input)
        return SpecialistAttempt(
            finding=SpecialistFindingDraft(summary="Markdown market finding")
        )


def _write_agent(
    root: Path,
    *,
    tenant_id: str,
    filename: str,
    content: str,
) -> None:
    directory = root / "tenants" / tenant_id / "agents"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(content, encoding="utf-8")


def _agent_document(
    *,
    specialist_id: str = "market-data",
    description: str = "Analyze market data.",
    model_profile: str = "specialist",
    skills: str = "[]",
    instructions: str = "Use only the assigned Task context.",
    extra_frontmatter: str = "",
) -> str:
    return f"""---
id: {specialist_id}
description: {description}
model-profile: {model_profile}
skills: {skills}
{extra_frontmatter}---
{instructions}
"""


@pytest.mark.asyncio
async def test_local_catalog_keeps_valid_sibling_and_logs_bounded_invalid_entries(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="market-data.agent.md",
        content=_agent_document(instructions="VALID-INSTRUCTIONS-SENTINEL"),
    )
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="unknown-model.agent.md",
        content=_agent_document(
            specialist_id="unknown-model", model_profile="not-approved"
        ),
    )
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="has-skills.agent.md",
        content=_agent_document(specialist_id="has-skills", skills="[filing-analysis]"),
    )
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="extra-field.agent.md",
        content=_agent_document(
            specialist_id="extra-field", extra_frontmatter="role: analyst\n"
        ),
    )
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="too-long.agent.md",
        content=_agent_document(specialist_id="too-long", instructions="x" * 30_001),
    )
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="malformed-yaml.agent.md",
        content="""---
id: [not-closed
---
Instructions
""",
    )
    invalid_encoding = tmp_path / "tenants" / "tenant-a" / "agents" / "invalid.agent.md"
    invalid_encoding.write_bytes(b"\xff")
    _write_agent(
        tmp_path,
        tenant_id="tenant-b",
        filename="foreign.agent.md",
        content=_agent_document(specialist_id="foreign"),
    )
    caplog.set_level(logging.INFO)

    catalog = await build_specialist_catalog(
        LocalSpecialistDefinitionLoader(tmp_path),
        tenant_id="tenant-a",
        model_registry=cast(ModelRegistry, _ModelRegistry("specialist")),
    )

    assert catalog.descriptors == (
        SpecialistDescriptor(id="market-data", description="Analyze market data."),
    )
    with pytest.raises(ValueError, match="Specialist is not eligible"):
        catalog.resolve("foreign")
    assert "tenant=tenant-a" in caplog.text
    assert "kind=specialist" in caplog.text
    assert "reason=unknown-model-profile" in caplog.text
    assert "reason=skills-unavailable" in caplog.text
    assert "reason=invalid-metadata" in caplog.text
    assert "reason=invalid-instructions" in caplog.text
    assert "reason=read-failed" in caplog.text
    assert "loaded=1 skipped=6" in caplog.text
    assert "VALID-INSTRUCTIONS-SENTINEL" not in caplog.text


@pytest.mark.asyncio
async def test_markdown_specialist_uses_instruction_precedence_and_persists_pin(
    tmp_path: Path,
) -> None:
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="market-data.agent.md",
        content=_agent_document(instructions="TENANT-INSTRUCTIONS-SENTINEL"),
    )
    registry = _ExecutableModelRegistry()
    catalog = await build_specialist_catalog(
        LocalSpecialistDefinitionLoader(tmp_path),
        tenant_id="tenant-a",
        model_registry=cast(ModelRegistry, registry),
    )
    batch = accept_initial_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Assess the market.",
                ),
            ),
        ),
        request_id="request-1",
        specialist_catalog=catalog,
    )

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        specialist_catalog=catalog,
        context=EvidenceInvocationContext(
            tenant_id="tenant-a",
            request_id="request-1",
            task_id=batch.tasks[0].id,
        ),
    )
    accepted = promote_batch(batch, {contribution.task_id: contribution})

    registration = catalog.resolve("market-data")
    assert len(registration.definition_pin) == 64
    assert contribution.specialist_definition_pin == registration.definition_pin
    assert accepted.specialist_definition_pins == (
        TaskSpecialistDefinitionPin(
            task_id=batch.tasks[0].id,
            pin=registration.definition_pin,
        ),
    )
    instructions = registry.instructions[0]
    assert instructions.index(SPECIALIST_INSTRUCTIONS.strip()) < instructions.index(
        SPECIALIST_SECURITY_GUARDS.strip()
    )
    assert instructions.index(SPECIALIST_SECURITY_GUARDS.strip()) < instructions.index(
        "TENANT-INSTRUCTIONS-SENTINEL"
    )
    assert isinstance(contribution.outcome, TaskSucceeded)
    assert contribution.outcome.result == SpecialistResult(
        summary="Markdown market finding"
    )


@pytest.mark.asyncio
async def test_catalog_snapshot_stays_fixed_until_a_new_startup_load(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tenants" / "tenant-a" / "agents" / "market-data.agent.md"
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename=path.name,
        content=_agent_document(
            description="First description.", instructions="FIRST-INSTRUCTIONS"
        ),
    )
    registry = _ExecutableModelRegistry()
    first = await build_specialist_catalog(
        LocalSpecialistDefinitionLoader(tmp_path),
        tenant_id="tenant-a",
        model_registry=cast(ModelRegistry, registry),
    )
    first_registration = first.resolve("market-data")

    path.write_text(
        _agent_document(
            description="Second description.", instructions="SECOND-INSTRUCTIONS"
        ),
        encoding="utf-8",
    )
    first.bind_actor(
        first_registration,
        context=EvidenceInvocationContext(
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
        ),
    )
    second = await build_specialist_catalog(
        LocalSpecialistDefinitionLoader(tmp_path),
        tenant_id="tenant-a",
        model_registry=cast(ModelRegistry, registry),
    )
    second_registration = second.resolve("market-data")
    second.bind_actor(
        second_registration,
        context=EvidenceInvocationContext(
            tenant_id="tenant-a",
            request_id="request-2",
            task_id="task-2",
        ),
    )

    assert first.descriptors == (
        SpecialistDescriptor(id="market-data", description="First description."),
    )
    assert second.descriptors == (
        SpecialistDescriptor(id="market-data", description="Second description."),
    )
    assert first_registration.definition_pin != second_registration.definition_pin
    assert "FIRST-INSTRUCTIONS" in registry.instructions[0]
    assert "SECOND-INSTRUCTIONS" not in registry.instructions[0]
    assert "SECOND-INSTRUCTIONS" in registry.instructions[1]


@pytest.mark.asyncio
async def test_definition_pin_is_canonical_and_excludes_storage_identity(
    tmp_path: Path,
) -> None:
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="first-name.agent.md",
        content=_agent_document(),
    )
    _write_agent(
        tmp_path,
        tenant_id="tenant-b",
        filename="different-name.agent.md",
        content="""---
skills: []
model-profile: specialist
description: Analyze market data.
id: market-data
---
Use only the assigned Task context.
""",
    )
    registry = cast(ModelRegistry, _ModelRegistry("specialist"))

    first = await build_specialist_catalog(
        LocalSpecialistDefinitionLoader(tmp_path),
        tenant_id="tenant-a",
        model_registry=registry,
    )
    second = await build_specialist_catalog(
        LocalSpecialistDefinitionLoader(tmp_path),
        tenant_id="tenant-b",
        model_registry=registry,
    )

    assert (
        first.resolve("market-data").definition_pin
        == second.resolve("market-data").definition_pin
    )


def test_definition_pin_fields_are_backward_compatible_with_old_checkpoint_data() -> (
    None
):
    accepted = AcceptedBatch.model_validate({"id": "batch-1", "outcomes": []})

    assert accepted.specialist_definition_pins == ()


@pytest.mark.asyncio
async def test_catalog_rejects_a_loader_document_owned_by_another_tenant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ForeignTenantLoader:
        async def load_specialists(self, tenant_id: str) -> SpecialistSourceLoad:
            assert tenant_id == "tenant-a"
            return SpecialistSourceLoad(
                documents=(
                    SpecialistSourceDocument(
                        tenant_id="tenant-b",
                        source_identity="tenants/tenant-b/agents/foreign.agent.md",
                        content=_agent_document(specialist_id="foreign"),
                    ),
                )
            )

    catalog = await build_specialist_catalog(
        ForeignTenantLoader(),
        tenant_id="tenant-a",
        model_registry=cast(ModelRegistry, _ModelRegistry("specialist")),
    )

    assert catalog.descriptors == ()
    assert "tenant=tenant-a" in caplog.text
    assert "reason=tenant-mismatch" in caplog.text


@pytest.mark.asyncio
async def test_local_loader_rejects_a_tenant_id_that_can_escape_its_prefix(
    tmp_path: Path,
) -> None:
    escaped_directory = tmp_path / "tenant-b" / "agents"
    escaped_directory.mkdir(parents=True)
    (escaped_directory / "foreign.agent.md").write_text(
        _agent_document(specialist_id="foreign"),
        encoding="utf-8",
    )

    loaded = await LocalSpecialistDefinitionLoader(tmp_path).load_specialists(
        "../tenant-b"
    )

    assert loaded.documents == ()
    assert len(loaded.failures) == 1
    assert loaded.failures[0].reason == "invalid-tenant-id"


@pytest.mark.asyncio
async def test_local_loader_rejects_a_cross_tenant_definition_symlink(
    tmp_path: Path,
) -> None:
    _write_agent(
        tmp_path,
        tenant_id="tenant-b",
        filename="foreign.agent.md",
        content=_agent_document(specialist_id="foreign"),
    )
    tenant_a_directory = tmp_path / "tenants" / "tenant-a" / "agents"
    tenant_a_directory.mkdir(parents=True)
    (tenant_a_directory / "foreign.agent.md").symlink_to(
        tmp_path / "tenants" / "tenant-b" / "agents" / "foreign.agent.md"
    )

    loaded = await LocalSpecialistDefinitionLoader(tmp_path).load_specialists(
        "tenant-a"
    )

    assert loaded.documents == ()
    assert len(loaded.failures) == 1
    assert loaded.failures[0].reason == "symlink-not-allowed"


@pytest.mark.asyncio
async def test_startup_loads_only_known_tenants_with_their_own_model_registries(
    tmp_path: Path,
) -> None:
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="tenant-a-specialist.agent.md",
        content=_agent_document(
            specialist_id="tenant-a-specialist",
            model_profile="tenant-a-model",
        ),
    )
    _write_agent(
        tmp_path,
        tenant_id="tenant-b",
        filename="tenant-b-specialist.agent.md",
        content=_agent_document(
            specialist_id="tenant-b-specialist",
            model_profile="tenant-b-model",
        ),
    )
    _write_agent(
        tmp_path,
        tenant_id="storage-only",
        filename="untrusted-tenant.agent.md",
        content=_agent_document(specialist_id="untrusted-tenant"),
    )
    manager = _TenantManager(
        {
            "tenant-a": _ModelRegistry("tenant-a-model"),
            "tenant-b": _ModelRegistry("tenant-b-model"),
        }
    )

    catalogs = await load_local_specialist_catalogs(manager, root=tmp_path)

    assert set(catalogs) == {"tenant-a", "tenant-b"}
    assert [item.id for item in catalogs["tenant-a"].descriptors] == [
        "tenant-a-specialist"
    ]
    assert [item.id for item in catalogs["tenant-b"].descriptors] == [
        "tenant-b-specialist"
    ]


@pytest.mark.asyncio
async def test_code_and_markdown_adapters_share_the_execution_contract(
    tmp_path: Path,
) -> None:
    async def provider(_source: str, _query: str) -> EvidenceEnvelope:
        raise AssertionError("The contract fixture must not call the business Tool")

    tool_registry = AgentToolRegistry(
        evidence_registrations=(
            EvidenceToolRegistration(
                id="filing-reader",
                provider=provider,
                allowed_sources=frozenset({"sec"}),
            ),
        )
    )
    effective_tool_ids = frozenset({"filing-reader"})
    code_inputs: list[SpecialistTaskInput] = []
    code_tool_names: list[tuple[str, ...]] = []

    def code_factory(
        tools: tuple[SpecialistTool, ...],
        tool_capture: SpecialistToolCapture,
        skill_invocation: SkillInvocation | None,
    ) -> _ContractActor:
        del tool_capture, skill_invocation
        code_tool_names.append(tuple(tool.__name__ for tool in tools))
        return _ContractActor(inputs=code_inputs)

    code_catalog = SpecialistCatalog(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                description="Analyze market data.",
                actor_factory=code_factory,
            ),
        ),
        tool_registry=tool_registry,
        tenant_allowed_tool_ids=effective_tool_ids,
    )
    _write_agent(
        tmp_path,
        tenant_id="tenant-a",
        filename="market-data.agent.md",
        content=_agent_document(),
    )
    markdown_registry = _ExecutableModelRegistry()
    markdown_catalog = await build_specialist_catalog(
        LocalSpecialistDefinitionLoader(tmp_path),
        tenant_id="tenant-a",
        model_registry=cast(ModelRegistry, markdown_registry),
        tool_registry=tool_registry,
        tenant_allowed_tool_ids=effective_tool_ids,
    )
    context = EvidenceInvocationContext(
        tenant_id="tenant-a",
        request_id="request-contract",
        task_id="unused-before-dispatch",
        allowed_tool_ids=effective_tool_ids,
        allowed_sources=frozenset({"sec"}),
    )
    contributions: list[BatchContribution] = []
    for catalog in (code_catalog, markdown_catalog):
        batch = accept_initial_dispatch(
            DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Assess the market.",
                    ),
                ),
            ),
            request_id="request-contract",
            specialist_catalog=catalog,
        )
        task = batch.tasks[0]
        contributions.append(
            await execute_specialist(
                task,
                batch_id=batch.id,
                specialist_catalog=catalog,
                context=context.model_copy(update={"task_id": task.id}),
            )
        )

    assert len(code_inputs) == 1
    assert code_inputs[0].task_id == contributions[0].task_id
    assert code_inputs[0].objective == "Assess the market."
    assert code_inputs[0].context_results == ()
    assert code_tool_names == [("filing-reader",)]
    assert tuple(tool.__name__ for tool in markdown_registry.tools[0]) == (
        "filing-reader",
    )
    assert contributions[0].outcome == contributions[1].outcome
    assert all(
        len(item.specialist_definition_pin or "") == 64 for item in contributions
    )
