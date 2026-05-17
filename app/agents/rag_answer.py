"""RAG answer generation agent.

Uses a *pro* model to synthesise an answer from retrieved context.

Architecture (informed by pydantic-ai best practices):

- **Static ``instructions``** (identity, guardrails, tenant/domain contracts)
  are set at Agent build time.  API providers can cache this prefix.
- **Dynamic ``@agent.instructions``** appends per-request reference data
  (documents, evidence).  This is *not* retained in ``message_history``
  for subsequent runs, avoiding stale context accumulation.

We use ``instructions`` (not ``system_prompt``) because pydantic-ai
recommends it by default, and because ``system_prompt`` content gets
baked into ``message_history`` — which would cause retrieved documents
from earlier turns to persist and pollute subsequent requests.
"""

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import AbstractToolset

from app.config.models import MCPServerConfig
from app.core.mcp import build_mcp_toolsets
from app.core.model_registry import ModelRegistry


@dataclass
class RAGAgentDeps:
    """Dependencies for the RAG agent.

    ``reference_data`` carries per-request document/evidence context.
    It is injected as a dynamic instruction so the static
    ``instructions`` prefix can be cached by the API provider,
    and so it does NOT persist in message_history.
    """

    reference_data: str


def create_rag_answer_agent(
    registry: ModelRegistry,
    model_name: str = "pro",
    instructions: str | None = None,
    mcp_configs: list[MCPServerConfig] | None = None,
) -> Agent[RAGAgentDeps, str]:
    """Create a RAG answer agent with cacheable static instructions.

    Args:
        registry: Model registry for resolving model names.
        model_name: Named model from ``llmConfig.models``.
        instructions: Static system prompt (identity + guardrails + contracts).
            Set at build time so API providers can cache this prefix.
        mcp_configs: Optional MCP server configurations.
    """
    from app.agents.history_processors import filter_thinking, trim_history

    # Build MCP toolsets if provided
    toolsets: list[AbstractToolset] = build_mcp_toolsets(mcp_configs or [])

    agent = registry.create_agent(
        model_name,
        output_type=str,
        deps_type=RAGAgentDeps,
        instructions=instructions,
        toolsets=toolsets,
        history_processors=[trim_history(20), filter_thinking()],
    )

    # Dynamic instruction: appends per-request reference data AFTER
    # the static instructions.  Using @agent.instructions (not
    # system_prompt) so it:
    # 1. Is NOT retained in message_history for subsequent runs.
    # 2. Is automatically placed after static instructions for
    #    smart cache boundary placement (Anthropic).
    @agent.instructions
    def inject_reference_data(ctx: RunContext[RAGAgentDeps]) -> str:
        data = ctx.deps.reference_data
        if data:
            return f"<reference_data>\n{data}\n</reference_data>"
        return ""

    return agent
