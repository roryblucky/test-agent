"""Dependencies injected into Pydantic AI RunContext for Agent tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.model_registry import ModelRegistry
from app.services.events import EventEmitter
from app.services.tenant_manager import TenantProviders

if TYPE_CHECKING:
    from app.services.flow_context import FlowContext
    from app.skills.registry import TenantSkillRegistry


@dataclass
class AgentDeps:
    """Dependencies container tailored for the ConfigDrivenOrchestrator's tools.

    The ``skill_registry`` and ``tenant_id`` fields are required for the
    ``activate_skill`` and ``load_references`` tools to work correctly —
    they allow the LLM-driven agent to dynamically load skill content at
    runtime (Tier 2 and Tier 3 of the agentskills.io progressive disclosure).
    """

    registry: ModelRegistry
    providers: TenantProviders
    emitter: EventEmitter | None = None

    # Skill system — injected so activate_skill / load_references tools work
    skill_registry: TenantSkillRegistry | None = None
    tenant_id: str = ""
    # Tracks which skills have already been activated in the current run
    activated_skill_names: list[str] = field(default_factory=list[str])
    # Built-in tools available to this agent after tenant/runtime policy filtering
    available_tool_names: list[str] = field(default_factory=list[str])
    # Optional workflow execution context for tools that write evidence/audit data
    flow_context: FlowContext | None = None
