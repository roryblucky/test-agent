"""Agent Handler for dynamic Pydantic AI orchestration."""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import MCP, PrepareTools
from pydantic_ai.tools import ToolDefinition

from app.agents.agent_deps import AgentDeps
from app.agents.tool_registry import BuiltInToolsCapability
from app.config.models import AgentConfig, FlowStep, TenantConfig
from app.config.prompts import load_prompt
from app.core.model_registry import ModelRegistry
from app.services.flow_context import FlowContext
from app.services.handlers.base import StepHandler
from app.services.tenant_manager import TenantProviders

try:
    from pydantic_ai_skills import SkillsCapability
except ImportError:
    SkillsCapability = None

try:
    from pydantic_ai_todo import TodoCapability
except ImportError:
    TodoCapability = None

logger = logging.getLogger(__name__)


class AgentHandler(StepHandler):
    """Executes a dynamic agent assembled via configuration."""

    def __init__(
        self,
        registry: ModelRegistry,
        providers: TenantProviders,
        cfg: TenantConfig,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self.cfg = cfg
        self.app_id = cfg.application_id
        
        self._agent_cache: dict[str, Agent] = {}

    def warmup(self, steps: list[FlowStep]) -> None:
        """Pre-build agents for all configured agent steps."""
        for step in steps:
            if step.type == "agent" and step.agent_config:
                key = step.mode or "default"
                self._agent_cache[key] = self._build_tenant_agent(step.agent_config)

    def _build_tenant_agent(self, agent_config: AgentConfig) -> Agent[Any, Any]:
        """Dynamically assemble the Pydantic AI Agent."""
        capabilities = []
        
        # 1. Prompt Loading (System Prompt)
        system_prompt = "You are a helpful assistant."
        if agent_config.prompt_type:
            system_prompt = load_prompt(agent_config.prompt_type, self.app_id)

        # 2. Todo Capability
        if agent_config.enable_todo:
            if TodoCapability:
                capabilities.append(TodoCapability(enable_subtasks=True))
            else:
                logger.error("TodoCapability requested but pydantic-ai-todo not installed.")

        # 3. Agent Skills Capability
        if agent_config.skills:
            if SkillsCapability:
                capabilities.append(SkillsCapability(directories=agent_config.skills))
            else:
                logger.error("SkillsCapability requested but pydantic-ai-skills not installed.")

        # 4. Built-in Tools Capability
        if agent_config.built_in_tools:
            allowed_tools = list(agent_config.built_in_tools)
            # Mutual exclusion: Deep Agent (Todo) conflicts with Lightweight planning (plan_and_reason)
            if agent_config.enable_todo and "plan_and_reason" in allowed_tools:
                logger.warning(
                    "Both enableTodo and plan_and_reason are configured. "
                    "Dropping plan_and_reason to prevent cognitive overload."
                )
                allowed_tools.remove("plan_and_reason")

            if allowed_tools:
                capabilities.append(
                    BuiltInToolsCapability(
                        application_id=self.app_id,
                        allowed_tool_names=allowed_tools,
                    )
                )

        # 5. MCP Servers
        exact_allowed_mcp_tools: set[str] = set()
        allow_all_mcp_prefixes: list[str] = []
        has_mcp_allowed_lists = False
        
        for name, mcp_cfg in agent_config.mcp_servers.items():
            if mcp_cfg.allowed_tools is not None:
                has_mcp_allowed_lists = True
                for t in mcp_cfg.allowed_tools:
                    exact_allowed_mcp_tools.add(f"{name}_{t}")
            else:
                allow_all_mcp_prefixes.append(f"{name}_")
                
            if mcp_cfg.url:
                capabilities.append(MCP(url=mcp_cfg.url).prefix_tools(name))
            elif mcp_cfg.command:
                logger.warning(f"MCP command transport for {name} not yet standard via capabilities.")

        exact_allowed_skill_tools: set[str] = set()

        # Fail-Fast Collision Check
        built_in_set = set(allowed_tools) if allowed_tools else set()
        mcp_builtin_overlap = exact_allowed_mcp_tools.intersection(built_in_set)
        if mcp_builtin_overlap:
            raise ValueError(f"Tool name collision detected between MCP and Built-in tools: {mcp_builtin_overlap}")
            
        skill_builtin_overlap = exact_allowed_skill_tools.intersection(built_in_set)
        if skill_builtin_overlap:
             raise ValueError(f"Tool name collision detected between Skill and Built-in tools: {skill_builtin_overlap}")

        # 6. Global Tool Filter
        async def filter_tools(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            filtered = []
            for td in tool_defs:
                if td.name in built_in_set:
                    filtered.append(td)
                elif td.name.startswith("add_todo") or td.name.startswith("read_todos"): 
                    filtered.append(td)
                elif has_mcp_allowed_lists and td.name in exact_allowed_mcp_tools:
                    filtered.append(td)
                elif allow_all_mcp_prefixes and td.name.startswith(tuple(allow_all_mcp_prefixes)):
                    filtered.append(td)
                elif td.name in exact_allowed_skill_tools:
                    filtered.append(td)
                else:
                    logger.debug(f"Filtered out unallowed or unknown tool: {td.name}")
            return filtered
            
        capabilities.append(PrepareTools(filter_tools))

        model = self.registry.get_model(agent_config.llm_type)
        return Agent(
            model=model,
            output_type=str,
            system_prompt=system_prompt,
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
            raise ValueError(f"Agent for mode {key!r} not initialized. Ensure agentConfig is provided in the step.")

        deps = AgentDeps(
            registry=self.registry,
            providers=self.providers,
            emitter=ctx.emitter,
        )

        async with agent.run_stream(
            effective_query, deps=deps, message_history=ctx.message_history or None
        ) as stream:
            previous_text = ""
            async for chunk in stream.stream_output(debounce_by=0.01):
                # Check for stop signal
                if ctx.emitter and ctx.emitter.is_cancelled:
                    break
                if isinstance(chunk, str):
                    new_text = chunk[len(previous_text):]
                    if new_text and ctx.emitter:
                        await ctx.emitter.emit_token(new_text)
                    previous_text = chunk

            output = await stream.get_output()
            ctx.add_usage(stream.usage())
            ctx.new_messages = stream.new_messages()

        ctx.llm_response = output

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
