"""Handler for memory persistence step."""

from __future__ import annotations

from app.config.models import FlowStep
from app.services.flow_context import FlowContext


class MemoryHandler:
    """Handles memory persistence."""

    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Persist session / long-term memory."""
        # Currently a no-op placeholder.
        if ctx.emitter:
            await ctx.emitter.emit_step_completed("memory", {"mode": step.mode})
        return ctx
