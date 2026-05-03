"""Config-driven Agent Orchestration.

Dynamically assembles Pydantic AI Agents based on TenantConfig, leveraging
Capabilities for Skills, MCPs, local tools, and tool filtering.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import MCP, PrepareTools
from pydantic_ai.tools import ToolDefinition

from app.agents.agent_deps import AgentDeps
from app.agents.tool_registry import BuiltInToolsCapability
from app.api.schemas import ClarificationRequest
from app.config.models import TenantConfig
from app.config.prompts import load_prompt
from app.core.model_registry import ModelRegistry
from app.services.events import EventEmitter
from app.services.flow_context import FlowContext
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


def create_mcp_filter_capability(allowed_tools: list[str] | None) -> PrepareTools | None:
    """Create a PrepareTools capability to filter MCP tools based on allowed_tools list."""
    if allowed_tools is None:
        return None

    allowed_set = set(allowed_tools)

    async def filter_tools(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        filtered = []
        for td in tool_defs:
            if td.name in allowed_set:
                filtered.append(td)
            else:
                logger.debug(f"Filtered out unallowed MCP tool: {td.name}")
        return filtered

    return PrepareTools(filter_tools)


class ConfigDrivenOrchestrator:
    """Assembles and runs an agent purely from TenantConfig."""

    def __init__(
        self,
        config: TenantConfig,
        registry: ModelRegistry,
        providers: TenantProviders,
    ) -> None:
        self.config = config
        self.registry = registry
        self.providers = providers
        
        self.agent_config = config.flow_config.agent_config
        if not self.agent_config:
            raise ValueError("ConfigDrivenOrchestrator requires agentConfig.")
            
        self.app_id = config.application_id

    def _build_agent(self) -> Agent[Any, Any]:
        """Dynamically assemble the Pydantic AI Agent."""
        capabilities = []
        
        # 1. Prompt Loading (System Prompt)
        system_prompt = "You are a helpful assistant."
        if self.agent_config.prompt_type:
            system_prompt = load_prompt(self.agent_config.prompt_type, self.app_id)

        # 2. Todo Capability
        if self.agent_config.enable_todo:
            if TodoCapability:
                capabilities.append(TodoCapability(enable_subtasks=True))
            else:
                logger.error("TodoCapability requested but pydantic-ai-todo not installed.")

        # 3. Agent Skills Capability
        if self.agent_config.skills:
            if SkillsCapability:
                capabilities.append(SkillsCapability(directories=self.agent_config.skills))
            else:
                logger.error("SkillsCapability requested but pydantic-ai-skills not installed.")

        # 4. Built-in Tools Capability
        if self.agent_config.built_in_tools:
            allowed_tools = list(self.agent_config.built_in_tools)
            # Mutual exclusion: Deep Agent (Todo) conflicts with Lightweight planning (plan_and_reason)
            if self.agent_config.enable_todo and "plan_and_reason" in allowed_tools:
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
        # For tool collision checking across MCP servers
        all_allowed_mcp_tools = set()
        has_allowed_lists = False
        
        for name, mcp_cfg in self.agent_config.mcp_servers.items():
            if mcp_cfg.allowed_tools is not None:
                has_allowed_lists = True
                all_allowed_mcp_tools.update(mcp_cfg.allowed_tools)
                
            if mcp_cfg.url:
                capabilities.append(MCP(url=mcp_cfg.url).prefix_tools(name))
            elif mcp_cfg.command:
                # Assuming the builtin MCP capability handles stdio via command soon,
                # otherwise we fallback to custom instantiation if needed.
                # For now, pydantic-ai.capabilities.MCP uses `url` to denote sse/http.
                # We log warning if command is used but not yet supported directly by `MCP()`.
                # If supported, it might be something like `MCP(command=...)`
                logger.warning(f"MCP command transport for {name} not yet standard via capabilities.")

        # 6. Global MCP Filter (if any allowed_tools restrictions exist)
        # Note: We filter at the global level because PrepareTools acts globally on the agent's tool defs.
        # Since we use `.prefix_tools(name)`, the tool names become `name_toolname`.
        # We need to refine the filter to handle prefixes if the user configured `allowedTools: ["A"]`.
        # Here we do a simplified check for exact matches or prefix matching.
        if has_allowed_lists:
            async def filter_tools(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
                filtered = []
                for td in tool_defs:
                    # Strip prefix to check against allowed names if using PrefixTools
                    # E.g., `cioView_A` -> `A`
                    base_name = td.name
                    if "_" in td.name:
                        base_name = td.name.split("_", 1)[1]
                        
                    if base_name in all_allowed_mcp_tools or td.name in all_allowed_mcp_tools:
                        filtered.append(td)
                    # We also must allow built-in tools and skill tools to pass through
                    elif self.agent_config.built_in_tools and td.name in self.agent_config.built_in_tools:
                        filtered.append(td)
                    elif td.name.startswith("add_todo") or td.name.startswith("read_todos"): 
                        # Allow Todo tools (should really use metadata to tag safe tools)
                        filtered.append(td)
                    else:
                        # Simple heuristic: if it's an MCP tool not in the allowed list, drop it.
                        pass
                        
                return filtered
            capabilities.append(PrepareTools(filter_tools))

        # Check for immediate collisions between built-in tools and requested MCP tools
        if all_allowed_mcp_tools and self.agent_config.built_in_tools:
            overlap = all_allowed_mcp_tools.intersection(self.agent_config.built_in_tools)
            if overlap:
                raise ValueError(f"Tool name collision detected (Fail-Fast): {overlap}")

        model = self.registry.get_model(self.agent_config.llm_type)
        agent = Agent(
            model=model,
            result_type=str | ClarificationRequest,
            system_prompt=system_prompt,
            capabilities=capabilities,
        )
        return agent

    async def execute(
        self,
        query: str,
        emitter: EventEmitter | None = None,
        session_id: str | None = None,
        message_history: list | None = None,
    ) -> FlowContext:
        """Run the configuration-driven agent."""
        ctx = FlowContext(
            query=query,
            emitter=emitter,
            session_id=session_id,
            message_history=message_history or [],
        )

        agent = self._build_agent()
        
        if ctx.emitter:
            await ctx.emitter.emit_step_start("config_driven_agent")

        deps = AgentDeps(
            registry=self.registry,
            providers=self.providers,
            emitter=ctx.emitter,
        )

        async with agent.run_stream(
            query, deps=deps, message_history=ctx.message_history or None
        ) as stream:
            # 动态判断：如果大模型选择输出纯文本(str)，则逐字流式打字
            if stream.is_text:
                async for chunk in stream.stream_text(delta=True):
                    if ctx.emitter:
                        await ctx.emitter.emit_token(chunk)
            
            output = await stream.get_output()
            ctx.add_usage(stream.usage())
            # 捕获本次运行新产生的对话记录
            ctx.new_messages = stream.new_messages()

        if isinstance(output, ClarificationRequest):
            ctx.clarification_request = output
            if ctx.emitter:
                await ctx.emitter.emit_step_completed(
                    "config_driven_agent",
                    {"clarification": True},
                )
        else:
            ctx.llm_response = output
            if ctx.emitter:
                await ctx.emitter.emit_step_completed(
                    "config_driven_agent",
                    {"output_length": len(str(output))},
                )
            
        return ctx
