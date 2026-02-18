"""Base interface for flow step handlers."""

from __future__ import annotations

from typing import Protocol

from app.config.models import FlowStep
from app.services.flow_context import FlowContext


class StepHandler(Protocol):
    """Protocol for a flow step handler."""

    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Execute a single flow step."""
        ...
