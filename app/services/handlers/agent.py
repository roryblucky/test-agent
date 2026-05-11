"""Agent Handler — dynamic Pydantic AI orchestration with Skill capability.

Skills are managed by SkillsCapability (AbstractCapability), which owns:

- Tier 1 (Discovery): XML catalog injected via get_instructions callable
- Tier 2 (Activation): activate_skill tool registered via get_toolset
- Tier 3 (References): load_skill_references tool registered via get_toolset
- Per-run state: for_run injects registry/tenant into ctx.deps

Domain tools are registered via BuiltInToolRegistry as a FunctionToolset.
MCP servers are registered as MCP capabilities.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import Agent, RunContext, ToolOutput
from pydantic_ai.capabilities import MCP
from pydantic_ai.toolsets import FunctionToolset

from app.agents.agent_deps import AgentDeps
from app.agents.tool_registry import BuiltInToolRegistry
from app.config.models import AgentConfig, FlowStep, TenantConfig
from app.config.prompts import load_prompt
from app.core.model_registry import ModelRegistry
from app.prompts.builder import LayeredPromptBuilder
from app.services.flow_context import FlowContext
from app.services.handlers.base import StepHandler
from app.services.tenant_manager import TenantProviders
from app.skills.capability import SkillsCapability
from app.skills.registry import TenantSkillRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output model registry for ToolOutput (redirect=True) skills
# ---------------------------------------------------------------------------

# Import all structured output models that skills can redirect to.
# Add new models here as you define new redirect-capable skills.
from pydantic import BaseModel  # noqa: E402


class SQLResult(BaseModel):
    """Structured SQL query result for direct return."""

    query: str
    rows: list[dict[str, Any]]
    row_count: int


class SearchResult(BaseModel):
    """Structured search result for direct return."""

    answer: str
    sources: list[str]


# Registry: maps redirect_output_schema string → Pydantic model class
OUTPUT_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "SQLResult": SQLResult,
    "SearchResult": SearchResult,
}


# ---------------------------------------------------------------------------
# AgentHandler
# ---------------------------------------------------------------------------


class AgentHandler(StepHandler):
    """Executes a dynamic agent assembled via tenant config + skills."""

    def __init__(
        self,
        registry: ModelRegistry,
        providers: TenantProviders,
        cfg: TenantConfig,
        skill_registry: TenantSkillRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self.cfg = cfg
        self.app_id = cfg.application_id
        self.skill_registry = skill_registry

        self._agent_cache: dict[str, Agent] = {}

    def warmup(self, steps: list[FlowStep]) -> None:
        """Pre-build agents for all configured agent steps.

        Fully synchronous — only assembles Pydantic AI Agent objects.
        The skill registry needs only Tier 1 discovery (summaries) to be
        complete before this runs, which is cheap and synchronous-friendly.
        Full skill content (Tier 2) is loaded lazily at runtime by the
        ``activate_skill`` tool when the LLM decides it's needed.
        """
        for step in steps:
            if step.type == "agent" and step.agent_config:
                key = step.mode or "default"
                self._agent_cache[key] = self._build_tenant_agent(
                    step.agent_config
                )

    def _build_tenant_agent(
        self, agent_config: AgentConfig
    ) -> Agent[Any, Any]:
        """Dynamically assemble a Pydantic AI Agent using Pydantic AI capabilities.

        Architecture (pydantic-ai 1.93+):

        capabilities = [
            SkillsCapability(registry, tenant_id),   # agentskills.io Tier 1/2/3
            MCP(url=...).prefix_tools(name),         # per MCP server
        ]
        tools = FunctionToolset([...domain tools...])  # RAG operations
        Agent(capabilities=capabilities, toolsets=[tools])

        SkillsCapability owns the full skill lifecycle:
        - get_instructions: XML catalog in system prompt (Tier 1)
        - get_toolset: activate_skill + load_skill_references tools (Tier 2/3)
        - for_run: injects registry/tenant/activated_skill_names into deps
        """
        capabilities: list[Any] = []

        # --- 1. Skills capability (Tier 1 + 2 + 3) ---
        if self.skill_registry:
            capabilities.append(
                SkillsCapability(
                    registry=self.skill_registry,
                    tenant_id=self.app_id,
                )
            )

        # --- 2. MCP capabilities ---
        for mcp_name, mcp_cfg in agent_config.mcp_servers.items():
            if not mcp_cfg.url:
                continue
            mcp_cap = MCP(
                url=mcp_cfg.url,
                id=mcp_name,
                allowed_tools=mcp_cfg.allowed_tools,  # None = allow all
            )
            capabilities.append(mcp_cap)

        # --- 3. Domain tools (FunctionToolset) ---
        builtin_tool_names = set(agent_config.built_in_tools)
        if agent_config.enable_todo and "plan_and_reason" in builtin_tool_names:
            logger.warning(
                "Both enableTodo and plan_and_reason configured. "
                "Dropping plan_and_reason to prevent cognitive overload."
            )
            builtin_tool_names.discard("plan_and_reason")

        domain_toolset = FunctionToolset(
            BuiltInToolRegistry.get_tools(self.app_id, list(builtin_tool_names))
        )

        # --- 4. System prompt (Identity + Guardrails + Context layers) ---
        # NOTE: Tier 1 skill discovery catalog is injected by SkillsCapability
        # via get_instructions, so we pass no discovery_index here.
        legacy_prompt: str | None = None
        if agent_config.prompt_type:
            legacy_prompt = load_prompt(agent_config.prompt_type, self.app_id)

        system_prompt = LayeredPromptBuilder.build(
            extra_instructions=legacy_prompt,
        )

        # --- 5. ToolOutput for redirect-capable skills ---
        output_types: list[Any] = [str]
        if self.skill_registry:
            for summary in self.skill_registry.get_summaries(self.app_id):
                if summary.name not in (agent_config.skills or []):
                    continue
                skill = self.skill_registry.get_activated_skill(self.app_id, summary.name)
                if skill and skill.metadata.redirect and skill.metadata.redirect_output_schema:
                    model_cls = OUTPUT_MODEL_REGISTRY.get(skill.metadata.redirect_output_schema)
                    if model_cls:
                        output_types.append(
                            ToolOutput(
                                model_cls,
                                name=f"return_{skill.metadata.name}_result",
                                description=(
                                    f"Directly return {skill.metadata.name} "
                                    "results without further LLM processing"
                                ),
                            )
                        )

        # --- 6. Assemble the Agent ---
        model = self.registry.get_model(agent_config.llm_type)
        return Agent(
            model=model,
            deps_type=AgentDeps,
            output_type=output_types if len(output_types) > 1 else str,
            instructions=system_prompt,
            toolsets=[domain_toolset],
            capabilities=capabilities,
        )

    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Run the configured agent."""
        effective_query = ctx.refined_query or ctx.query

        if ctx.emitter:
            await ctx.emitter.emit_step_start("agent:orchestration")

        key = step.mode or "default"
        agent = self._agent_cache.get(key)
        if not agent:
            raise ValueError(
                f"Agent for mode {key!r} not initialized. "
                "Ensure agentConfig is provided in the step."
            )

        deps = AgentDeps(
            registry=self.registry,
            providers=self.providers,
            emitter=ctx.emitter,
            skill_registry=self.skill_registry,
            tenant_id=self.app_id,
        )

        async with agent.run_stream(
            effective_query,
            deps=deps,
            message_history=ctx.message_history or None,
        ) as stream:
            previous_text = ""
            async for chunk in stream.stream_output(debounce_by=0.01):
                # Check for stop signal
                if ctx.emitter and ctx.emitter.is_cancelled:
                    break
                if isinstance(chunk, str):
                    new_text = chunk[len(previous_text) :]
                    if new_text and ctx.emitter:
                        await ctx.emitter.emit_token(new_text)
                    previous_text = chunk

            output = await stream.get_output()
            ctx.add_usage(stream.usage())
            ctx.new_messages = stream.new_messages()

        ctx.llm_response = output if isinstance(output, str) else str(output)

        # Handle stop signal
        if ctx.emitter and ctx.emitter.is_cancelled:
            await ctx.emitter.emit_stopped(previous_text or str(output))
            from app.services.events import GenerationCancelledError

            raise GenerationCancelledError("Stopped during agent:orchestration")

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "agent:orchestration",
                {"output_length": len(str(output))},
            )

        return ctx
