"""Dependencies injected into Pydantic AI RunContext for Agent tools."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.model_registry import ModelRegistry
from app.services.events import EventEmitter
from app.services.tenant_manager import TenantProviders


@dataclass
class AgentDeps:
    """Dependencies container tailored for the ConfigDrivenOrchestrator's tools."""

    registry: ModelRegistry
    providers: TenantProviders
    emitter: EventEmitter | None = None
