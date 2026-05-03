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
        exact_allowed_mcp_tools: set[str] = set()
        allow_all_mcp_prefixes: list[str] = []
        has_mcp_allowed_lists = False
        
        for name, mcp_cfg in self.agent_config.mcp_servers.items():
            if mcp_cfg.allowed_tools is not None:
                has_mcp_allowed_lists = True
                for t in mcp_cfg.allowed_tools:
                    # Because we use .prefix_tools(name), the actual registered tool name is prefixed
                    exact_allowed_mcp_tools.add(f"{name}_{t}")
            else:
                # If allowed_tools is None, we allow ALL tools from this server
                allow_all_mcp_prefixes.append(f"{name}_")
                
            if mcp_cfg.url:
                capabilities.append(MCP(url=mcp_cfg.url).prefix_tools(name))
            elif mcp_cfg.command:
                logger.warning(f"MCP command transport for {name} not yet standard via capabilities.")

        # If Skill allowed_tools is implemented in the future, you would populate it here:
        exact_allowed_skill_tools: set[str] = set()
        # allow_all_skill_prefixes = ["skill_"] # if you prefix skill tools

        # Fail-Fast Collision Check
        # Now we can precisely check for collisions because we know the EXACT registered names
        built_in_set = set(allowed_tools) if allowed_tools else set()
        mcp_builtin_overlap = exact_allowed_mcp_tools.intersection(built_in_set)
        if mcp_builtin_overlap:
            raise ValueError(f"Tool name collision detected between MCP and Built-in tools: {mcp_builtin_overlap}")
            
        skill_builtin_overlap = exact_allowed_skill_tools.intersection(built_in_set)
        if skill_builtin_overlap:
             raise ValueError(f"Tool name collision detected between Skill and Built-in tools: {skill_builtin_overlap}")

        # 6. Global Tool Filter
        # Note: We filter at the global level because PrepareTools acts globally on the agent's tool defs.
        # We must use exact tool names or precise prefixes to distinguish sources, since ToolDefinition 
        # itself does not store "source" metadata.
        async def filter_tools(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            filtered = []
            for td in tool_defs:
                # 1. Check Built-in tools
                if td.name in built_in_set:
                    filtered.append(td)
                # 2. Check Todo tools (always allowed if enabled)
                elif td.name.startswith("add_todo") or td.name.startswith("read_todos"): 
                    filtered.append(td)
                # 3. Check exact allowed MCP tools
                elif has_mcp_allowed_lists and td.name in exact_allowed_mcp_tools:
                    filtered.append(td)
                # 4. Check MCP servers that allow all tools
                elif allow_all_mcp_prefixes and td.name.startswith(tuple(allow_all_mcp_prefixes)):
                    filtered.append(td)
                # 5. Check exact allowed Skill tools (Future)
                elif td.name in exact_allowed_skill_tools:
                    filtered.append(td)
                # 6. Check Skill sources that allow all tools (if any prefix or logic applies)
                # elif td.name.startswith(tuple(allow_all_skill_prefixes)):
                #     filtered.append(td)
                else:
                    logger.debug(f"Filtered out unallowed or unknown tool: {td.name}")
                    
            return filtered
            
        capabilities.append(PrepareTools(filter_tools))

        model = self.registry.get_model(self.agent_config.llm_type)
        agent = Agent(
            model=model,
            output_type=str | ClarificationRequest,
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
            # Pydantic AI 1.x 移除了 is_text 和对 Union 类型调用 stream_text 的支持
            # 我们通过 stream_output 流式获取解析结果，如果是字符串则计算差值并打字
            previous_text = ""
            async for chunk in stream.stream_output(debounce_by=0.01):
                if isinstance(chunk, str):
                    new_text = chunk[len(previous_text):]
                    if new_text and ctx.emitter:
                        await ctx.emitter.emit_token(new_text)
                    previous_text = chunk
            
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
