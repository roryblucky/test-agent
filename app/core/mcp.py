"""Shared MCP utilities."""

from __future__ import annotations

from pydantic_ai.toolsets import AbstractToolset

from app.config.models import MCPServerConfig


def build_mcp_toolsets(
    configs: list[MCPServerConfig],
) -> list[AbstractToolset]:
    """Convert MCP server configs to pydantic-ai toolset instances."""
    if not configs:
        return []

    toolsets: list[AbstractToolset] = []
    for cfg in configs:
        if cfg.url:
            # SSE transport (legacy) vs Streamable HTTP
            if cfg.url.rstrip("/").endswith("/sse"):
                from pydantic_ai.mcp import MCPServerSSE

                toolsets.append(MCPServerSSE(cfg.url, tool_prefix=cfg.name))
            else:
                from pydantic_ai.mcp import MCPServerStreamableHTTP

                toolsets.append(MCPServerStreamableHTTP(cfg.url, tool_prefix=cfg.name))
        elif cfg.command:
            from pydantic_ai.mcp import MCPServerStdio

            toolsets.append(
                MCPServerStdio(
                    cfg.command,
                    args=cfg.args,
                    env=cfg.env,
                    tool_prefix=cfg.name,
                )
            )
    return toolsets
