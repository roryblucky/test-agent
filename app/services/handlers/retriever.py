"""Handler for document retrieval step."""

from __future__ import annotations

from app.config.models import FlowStep
from app.core.resilience import safe_execute
from app.core.telemetry import trace_span
from app.providers.base import BaseRetrieverProvider
from app.services.flow_context import FlowContext


class RetrieverHandler:
    """Handles document retrieval."""

    def __init__(self, provider: BaseRetrieverProvider | None):
        self.provider = provider

    @trace_span("retriever")
    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Run document retrieval."""
        if not self.provider:
            raise ValueError(
                "Flow step 'retriever' requires 'retrieverConfig' in tenant config"
            )
        effective_query = ctx.refined_query or ctx.query

        # Use safe_execute for retrieval
        ctx.documents = await safe_execute(
            self.provider.retrieve,
            effective_query,
            self.provider.config.top_k,
        )
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "retriever",
                {
                    "document_count": len(ctx.documents),
                    "documents": [
                        {"id": d.id, "score": d.score} for d in ctx.documents
                    ],
                },
            )
        return ctx
