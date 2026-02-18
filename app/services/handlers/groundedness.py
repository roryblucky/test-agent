"""Handler for groundedness check step."""

from __future__ import annotations

from app.config.models import FlowStep
from app.core.telemetry import trace_span
from app.providers.base import BaseGroundednessProvider
from app.services.flow_context import FlowContext


class GroundednessHandler:
    """Handles groundedness checking."""

    def __init__(self, provider: BaseGroundednessProvider | None):
        self.provider = provider

    @trace_span("groundedness")
    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Run groundedness check."""
        if not self.provider:
            raise ValueError(
                "Flow step 'groundedness' requires 'groundednessConfig' in tenant config"
            )
        if ctx.llm_response is None:
            raise ValueError("Groundedness step requires a prior LLM response")
        context_docs = ctx.ranked_documents or ctx.documents
        ctx.groundedness_result = await self.provider.check(
            ctx.llm_response, context_docs
        )
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "groundedness",
                {
                    "is_grounded": ctx.groundedness_result.is_grounded,
                    "score": ctx.groundedness_result.score,
                },
            )
        return ctx
