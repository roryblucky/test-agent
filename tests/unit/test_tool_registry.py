"""Unit tests for built-in tool registry metadata and policy resolution."""

from pydantic import BaseModel

from app.agents import tools
from app.agents.tool_registry import (
    BuiltInToolRegistry,
    SearchDocumentsInput,
    TextToolOutput,
    ToolDefinition,
    ToolRiskLevel,
)
from app.config.models import FlowConfig, LLMConfig, TenantConfig, ToolRuntimeConfig
from app.skills.schema import SkillDefinition, SkillMetadata


def _tenant(tool_runtime_config: ToolRuntimeConfig | None = None) -> TenantConfig:
    return TenantConfig(
        kmsAppName="Tool Test App",
        applicationId="tool-test",
        adGroups=["group1"],
        llm_config=LLMConfig(models={}),
        flow_config=FlowConfig(),
        tool_runtime_config=tool_runtime_config,
    )


def _skill(name: str, allowed_tools: list[str]) -> SkillDefinition:
    return SkillDefinition(
        metadata=SkillMetadata(
            name=name,
            description=f"{name} test skill",
            allowed_tools=allowed_tools,
        ),
        instructions="Use the allowed tools.",
        tenant_id="tool-test",
        source_path=f"/skills/{name}/SKILL.md",
    )


def test_get_tools_keeps_legacy_callable_interface() -> None:
    """Existing callers still receive function objects from get_tools."""
    resolved = BuiltInToolRegistry.get_tools(
        "tool-test",
        ["search_documents", "missing_tool", "search_documents"],
    )

    assert resolved == [
        tools.search_documents_tool,
        tools.search_documents_tool,
    ]


def test_registered_tool_has_typed_metadata() -> None:
    """Built-in tools expose metadata and input/output schemas."""
    definition = BuiltInToolRegistry.get_definitions(
        "tool-test",
        ["search_documents"],
    )[0]

    assert definition.name == "search_documents"
    assert definition.function is tools.search_documents_tool
    assert definition.risk_level == ToolRiskLevel.LOW
    assert definition.input_schema is SearchDocumentsInput
    assert definition.output_schema is TextToolOutput


def test_resolve_tools_uses_tenant_allowlist() -> None:
    """toolRuntimeConfig.enabledTools filters requested tools."""
    tenant = _tenant(
        ToolRuntimeConfig(
            enabledTools=["rank_documents"],
        )
    )

    result = BuiltInToolRegistry.resolve_tool_definitions(
        tenant,
        requested_names=["search_documents", "rank_documents"],
    )

    assert [definition.name for definition in result.definitions] == [
        "rank_documents"
    ]
    assert result.blocked_tool_names == ["search_documents"]
    assert result.missing_tool_names == []


def test_resolve_tools_intersects_skill_allowed_tools() -> None:
    """Runtime toolset is tenant allowlist intersected with skill allowed tools."""
    tenant = _tenant(
        ToolRuntimeConfig(
            enabledTools=["search_documents", "rank_documents"],
        )
    )
    skill = _skill("ranking-skill", ["rank_documents"])

    result = BuiltInToolRegistry.resolve_tool_definitions(
        tenant,
        skill_defs=[skill],
        requested_names=["search_documents", "rank_documents"],
    )

    assert [definition.name for definition in result.definitions] == [
        "rank_documents"
    ]
    assert result.blocked_tool_names == ["search_documents"]


def test_resolve_tools_defaults_to_legacy_requested_names_without_runtime_config() -> None:
    """Absent toolRuntimeConfig preserves the old requested-name behavior."""
    tenant = _tenant()

    resolved = BuiltInToolRegistry.resolve_tools(
        tenant,
        requested_names=["search_documents"],
    )

    assert resolved == [tools.search_documents_tool]


def test_high_risk_tool_requires_confirmation_by_default(
    monkeypatch,
) -> None:
    """High-risk tools marked for confirmation are blocked without approval."""

    class DangerousInput(BaseModel):
        target: str

    class DangerousOutput(BaseModel):
        status: str

    async def dangerous_tool(target: str) -> str:
        return f"deleted {target}"

    definition = ToolDefinition(
        name="dangerous_delete",
        function=dangerous_tool,
        description="Dangerous destructive test tool.",
        risk_level=ToolRiskLevel.HIGH,
        requires_confirmation=True,
        input_schema=DangerousInput,
        output_schema=DangerousOutput,
    )
    monkeypatch.setattr(
        BuiltInToolRegistry,
        "_definitions",
        {
            **BuiltInToolRegistry._definitions,
            "dangerous_delete": definition,
        },
    )
    tenant = _tenant(
        ToolRuntimeConfig(
            enabledTools=["dangerous_delete"],
        )
    )

    result = BuiltInToolRegistry.resolve_tool_definitions(
        tenant,
        requested_names=["dangerous_delete"],
    )

    assert result.definitions == []
    assert result.blocked_tool_names == ["dangerous_delete"]
