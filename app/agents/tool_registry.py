"""Multi-tenant Built-in Tool Registry.

Manages available internal function tools and provides them to the agent
as a Capability, scoped to the specific tenant's application ID.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from app.agents import tools

logger = logging.getLogger(__name__)


class BuiltInToolRegistry:
    """Registry for all safe, built-in python function tools."""

    # Map of tool names to their implementations
    # In a real app, this could be auto-discovered or loaded via decorators
    _all_tools: dict[str, Callable[..., Any]] = {
        "search_documents": tools.search_documents_tool,
        "rank_documents": tools.rank_documents_tool,
        "decompose_question": tools.decompose_question_tool,
        "analyze_section": tools.analyze_section_tool,
        "plan_and_reason": tools.plan_and_reason_tool, 
    }

    @classmethod
    def get_toolset(
        cls, application_id: str, allowed_tool_names: list[str]
    ) -> AgentToolset[Any]:
        """Create a toolset containing only the allowed tools for this tenant."""
        toolset = FunctionToolset()
        
        logger.info(f"[{application_id}] Assembling built-in tools: {allowed_tool_names}")

        for name in allowed_tool_names:
            if name in cls._all_tools:
                # Register the tool with the toolset
                toolset.tool(name=name)(cls._all_tools[name])
            else:
                logger.warning(
                    f"[{application_id}] Tool '{name}' not found in registry. "
                    "Skipping."
                )

        return toolset


@dataclass
class BuiltInToolsCapability(AbstractCapability[Any]):
    """Capability to inject tenant-specific built-in tools into an Agent."""

    application_id: str
    allowed_tool_names: list[str]

    def get_toolset(self) -> AgentToolset[Any] | None:
        """Return the assembled AgentToolset for this capability."""
        if not self.allowed_tool_names:
            return None
        return BuiltInToolRegistry.get_toolset(
            self.application_id, self.allowed_tool_names
        )
