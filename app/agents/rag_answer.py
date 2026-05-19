"""RAG answer generation agent.

Uses a *pro* model to synthesise an answer from retrieved context.

Architecture:

- **Static ``instructions``** (identity, guardrails, tenant/domain contracts)
  are set at Agent build time.  API providers can cache this prefix.
- Per-request reference data (documents, evidence) is sent as the answer
  runtime prompt by the handler, then sanitized out of persisted history.
"""

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.toolsets import AbstractToolset

from app.config.models import MCPServerConfig
from app.core.mcp import build_mcp_toolsets
from app.core.model_registry import ModelRegistry


@dataclass
class RAGAgentDeps:
    """Dependencies for the RAG agent."""


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

    return agent
