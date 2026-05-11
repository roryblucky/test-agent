"""Dependencies injected into Pydantic AI RunContext for Agent tools."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.model_registry import ModelRegistry
from app.services.events import EventEmitter
from app.services.tenant_manager import TenantProviders


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
    skill_registry: object | None = None   # TenantSkillRegistry (avoid circular import)
    tenant_id: str = ""
    # Tracks which skills have already been activated in the current run
    activated_skill_names: list[str] = field(default_factory=list)
