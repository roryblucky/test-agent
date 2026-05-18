"""Multi-tenant built-in tool registry.

Skills and tenant configs reference tool names from this registry. The
registry stores typed metadata alongside each callable so platform policy can
resolve allowed tools without hiding behavior in loose dictionaries.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agents import tools
from app.api.schemas import QuestionAnswerSelector
from app.models.workflow import ToolObservation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.config.models import TenantConfig
    from app.skills.schema import SkillDefinition

logger = logging.getLogger(__name__)


class ToolRiskLevel(StrEnum):
    """Risk level used by runtime policy."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolCallable(Protocol):
    """Protocol for registered tool callables."""

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool."""
        ...


class SearchDocumentsInput(BaseModel):
    """Input schema for ``search_documents``."""

    query: str
    filter_expr: str | None = None


class RankDocumentsInput(BaseModel):
    """Input schema for ``rank_documents``."""

    query: str
    document_texts: list[str]


class DecomposeQuestionInput(BaseModel):
    """Input schema for ``decompose_question``."""

    complex_question: str


class AnalyzeSectionInput(BaseModel):
    """Input schema for ``analyze_section``."""

    question: str
    context: str


class PlanAndReasonInput(BaseModel):
    """Input schema for ``plan_and_reason``."""

    reasoning: str


class GetUserClassificationInput(BaseModel):
    """Input schema for ``get_user_classification``."""

    response: str
    quick_questions: list[QuestionAnswerSelector] | None = None


class TextToolOutput(BaseModel):
    """Legacy text output schema for existing built-in tools."""

    text: str


class QuestionListToolOutput(BaseModel):
    """Output schema for question decomposition."""

    questions: list[str]


class ToolDefinition(BaseModel):
    """Registered tool metadata and implementation."""

    name: str
    function: Callable[..., Any] = Field(exclude=True)
    description: str
    domains: list[str] = Field(default_factory=list)
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    requires_confirmation: bool = False
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    provider_key: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolResolutionResult(BaseModel):
    """Result of tenant/skill-aware tool resolution."""

    definitions: list[ToolDefinition] = Field(default_factory=list)
    missing_tool_names: list[str] = Field(default_factory=list)
    blocked_tool_names: list[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def functions(self) -> list[Callable[..., Any]]:
        """Return resolved callables for Pydantic AI toolsets."""
        return [definition.function for definition in self.definitions]


class BuiltInToolRegistry:
    """Registry for built-in python function tools.

    Usage::

        # Get a list of resolved tool functions for the agent
        tool_fns = BuiltInToolRegistry.get_tools("app123", ["search_documents", "rank_documents"])
    """

    # Map of tool names to typed definitions.
    # Pydantic AI auto-extracts schema from type hints + docstrings.
    # Skill system tools (activate_skill, load_skill_references) are intentionally
    # NOT here — they are registered by SkillsCapability.get_toolset() instead.
    _definitions: dict[str, ToolDefinition] = {
        # Domain tools (RAG operations)
        "search_documents": ToolDefinition(
            name="search_documents",
            function=tools.search_documents_tool,
            description="Search tenant knowledge sources for relevant documents.",
            domains=["knowledge"],
            risk_level=ToolRiskLevel.LOW,
            input_schema=SearchDocumentsInput,
            output_schema=ToolObservation,
        ),
        "rank_documents": ToolDefinition(
            name="rank_documents",
            function=tools.rank_documents_tool,
            description="Rank document text snippets by relevance to a query.",
            domains=["knowledge"],
            risk_level=ToolRiskLevel.LOW,
            input_schema=RankDocumentsInput,
            output_schema=ToolObservation,
        ),
        "decompose_question": ToolDefinition(
            name="decompose_question",
            function=tools.decompose_question_tool,
            description="Break a complex question into focused sub-questions.",
            domains=["reasoning"],
            risk_level=ToolRiskLevel.LOW,
            input_schema=DecomposeQuestionInput,
            output_schema=ToolObservation,
        ),
        "analyze_section": ToolDefinition(
            name="analyze_section",
            function=tools.analyze_section_tool,
            description="Analyze provided context to answer a focused question.",
            domains=["knowledge", "reasoning"],
            risk_level=ToolRiskLevel.LOW,
            input_schema=AnalyzeSectionInput,
            output_schema=ToolObservation,
        ),
        "plan_and_reason": ToolDefinition(
            name="plan_and_reason",
            function=tools.plan_and_reason_tool,
            description="Record a lightweight planning note for agent orchestration.",
            domains=["reasoning"],
            risk_level=ToolRiskLevel.LOW,
            input_schema=PlanAndReasonInput,
            output_schema=ToolObservation,
        ),
        "get_user_classification": ToolDefinition(
            name="get_user_classification",
            function=tools.get_user_classification_tool,
            description="Ask the user to clarify intent with structured options.",
            domains=["clarification"],
            risk_level=ToolRiskLevel.LOW,
            input_schema=GetUserClassificationInput,
            output_schema=ToolObservation,
        ),
    }

    @classmethod
    def get_tools(
        cls, application_id: str, allowed_tool_names: list[str]
    ) -> list[Callable[..., Any]]:
        """Return a list of tool functions for the given names.

        These are passed directly to ``Agent(tools=[...])``.
        Pydantic AI auto-generates the function-calling schema from
        each function's signature.

        Args:
            application_id: Tenant ID for logging.
            allowed_tool_names: List of tool names to resolve.

        Returns:
            List of callable tool functions.
        """
        logger.info(
            f"[{application_id}] Resolving built-in tools: {allowed_tool_names}"
        )

        return [
            definition.function
            for definition in cls.get_definitions(application_id, allowed_tool_names)
        ]

    @classmethod
    def get_definitions(
        cls, application_id: str, tool_names: list[str]
    ) -> list[ToolDefinition]:
        """Return registered tool definitions for the given names."""
        resolved: list[ToolDefinition] = []
        for name in tool_names:
            definition = cls._definitions.get(name)
            if definition is None:
                logger.warning(
                    f"[{application_id}] Tool '{name}' not found in registry. "
                    "Skipping."
                )
                continue
            resolved.append(definition)

        return resolved

    @classmethod
    def resolve_tool_definitions(
        cls,
        tenant_config: TenantConfig,
        skill_defs: Sequence[SkillDefinition] | None = None,
        requested_names: Sequence[str] | None = None,
    ) -> ToolResolutionResult:
        """Resolve tools allowed by tenant config and activated skills.

        Compatibility rule:
        if ``toolRuntimeConfig`` is absent, ``requested_names`` behaves exactly
        like the legacy ``get_tools`` allowlist.
        """
        requested = cls._requested_tool_names(tenant_config, requested_names)
        tenant_allowed = cls._tenant_allowed_names(tenant_config, requested)
        skill_allowed = cls._skill_allowed_names(skill_defs)

        candidate_names = tenant_allowed
        if skill_allowed is not None:
            candidate_names = [
                name for name in candidate_names if name in skill_allowed
            ]

        result = ToolResolutionResult()
        registered_names = set(cls._definitions)

        for name in cls._dedupe(requested):
            if name not in registered_names:
                result.missing_tool_names.append(name)
                continue
            if name not in candidate_names:
                result.blocked_tool_names.append(name)

        require_high_risk_confirmation = True
        if tenant_config.tool_runtime_config is not None:
            require_high_risk_confirmation = (
                tenant_config.tool_runtime_config.require_confirmation_for_high_risk
            )

        for name in candidate_names:
            definition = cls._definitions.get(name)
            if definition is None:
                continue
            if (
                definition.requires_confirmation
                and definition.risk_level == ToolRiskLevel.HIGH
                and require_high_risk_confirmation
            ):
                result.blocked_tool_names.append(name)
                continue
            result.definitions.append(definition)

        if result.blocked_tool_names:
            logger.info(
                "[%s] Blocked built-in tools by tenant/skill policy: %s",
                tenant_config.application_id,
                result.blocked_tool_names,
            )
        if result.missing_tool_names:
            logger.warning(
                "[%s] Requested built-in tools missing from registry: %s",
                tenant_config.application_id,
                result.missing_tool_names,
            )

        return result

    @classmethod
    def resolve_tools(
        cls,
        tenant_config: TenantConfig,
        skill_defs: Sequence[SkillDefinition] | None = None,
        requested_names: Sequence[str] | None = None,
    ) -> list[Callable[..., Any]]:
        """Return tenant/skill-resolved tool functions."""
        return cls.resolve_tool_definitions(
            tenant_config,
            skill_defs=skill_defs,
            requested_names=requested_names,
        ).functions

    @classmethod
    def get_all_tool_names(cls) -> list[str]:
        """Return all registered tool names (for introspection)."""
        return list(cls._definitions)

    @classmethod
    def register_tool(
        cls,
        definition: ToolDefinition | None = None,
        *,
        name: str | None = None,
        func: Callable[..., Any] | None = None,
        description: str | None = None,
        domains: list[str] | None = None,
        risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
        requires_confirmation: bool = False,
        input_schema: type[BaseModel] | None = None,
        output_schema: type[BaseModel] | None = None,
        provider_key: str | None = None,
    ) -> None:
        """Dynamically register a new tool at runtime.

        Useful for plugins or test fixtures.
        """
        if definition is None:
            if name is None or func is None:
                raise ValueError("register_tool requires a ToolDefinition or name+func")
            definition = ToolDefinition(
                name=name,
                function=func,
                description=description or func.__doc__ or name,
                domains=domains or [],
                risk_level=risk_level,
                requires_confirmation=requires_confirmation,
                input_schema=input_schema or BaseModel,
                output_schema=output_schema or ToolObservation,
                provider_key=provider_key,
            )

        cls._definitions[definition.name] = definition
        logger.info(f"Registered new tool: {definition.name}")

    @classmethod
    def _requested_tool_names(
        cls,
        tenant_config: TenantConfig,
        requested_names: Sequence[str] | None,
    ) -> list[str]:
        if requested_names is not None:
            return cls._dedupe(requested_names)
        if tenant_config.tool_runtime_config is not None:
            return cls._dedupe(tenant_config.tool_runtime_config.enabled_tools)
        return cls.get_all_tool_names()

    @classmethod
    def _tenant_allowed_names(
        cls,
        tenant_config: TenantConfig,
        requested_names: Sequence[str],
    ) -> list[str]:
        if tenant_config.tool_runtime_config is None:
            return cls._dedupe(requested_names)

        enabled = tenant_config.tool_runtime_config.enabled_tools
        if not enabled:
            return []

        enabled_set = set(enabled)
        return [name for name in cls._dedupe(requested_names) if name in enabled_set]

    @staticmethod
    def _skill_allowed_names(
        skill_defs: Sequence[SkillDefinition] | None,
    ) -> set[str] | None:
        if not skill_defs:
            return None

        allowed: set[str] = set()
        for skill in skill_defs:
            allowed.update(skill.metadata.allowed_tools)
        return allowed

    @staticmethod
    def _dedupe(names: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(names))
