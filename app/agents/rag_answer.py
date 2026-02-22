"""RAG answer generation agent.

Uses a *pro* model to synthesise an answer from retrieved context.
"""

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import AbstractToolset

from app.config.models import MCPServerConfig
from app.core.mcp import build_mcp_toolsets
from app.core.model_registry import ModelRegistry


@dataclass
class RAGAgentDeps:
    """Dependencies for the RAG agent."""

    system_prompt: str
    # Add other dependencies here if needed (e.g., specific user context)


def rag_system_prompt(ctx: RunContext[RAGAgentDeps]) -> str:
    """Dynamic system prompt for the RAG agent."""
    return ctx.deps.system_prompt


def create_rag_answer_agent(
    registry: ModelRegistry,
    model_name: str = "pro",
    mcp_configs: list[MCPServerConfig] | None = None,
) -> Agent[RAGAgentDeps, str]:
    """Create a RAG answer agent that produces a text response."""
    from app.agents.history_processors import filter_thinking, trim_history

    # Build MCP toolsets if provided
    toolsets: list[AbstractToolset] = build_mcp_toolsets(mcp_configs or [])

    return registry.create_agent(
        model_name,
        output_type=str,
        deps_type=RAGAgentDeps,
        system_prompt=rag_system_prompt,
        toolsets=toolsets,
        history_processors=[trim_history(20), filter_thinking()],
    )
