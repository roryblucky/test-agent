"""Handler for document ranking step."""

from __future__ import annotations

from app.config.models import FlowStep
from app.core.telemetry import trace_span
from app.providers.base import BaseRankerProvider
from app.services.flow_context import FlowContext


class RankingHandler:
    """Handles document ranking."""

    def __init__(self, provider: BaseRankerProvider | None):
        self.provider = provider

    @trace_span("ranking")
    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Run document ranking."""
        if not self.provider:
            raise ValueError(
                "Flow step 'ranking' requires 'rankingConfig' in tenant config"
            )
        effective_query = ctx.refined_query or ctx.query
        ctx.ranked_documents = await self.provider.rank(effective_query, ctx.documents)
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "ranking",
                {
                    "document_count": len(ctx.ranked_documents),
                    "documents": [
                        {"id": d.id, "score": d.score} for d in ctx.ranked_documents
                    ],
                },
            )
        return ctx
