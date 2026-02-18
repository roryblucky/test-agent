"""Handler for content moderation step."""

from __future__ import annotations

from app.config.models import FlowStep
from app.providers.base import BaseModerationProvider
from app.services.exceptions import ContentFlaggedError
from app.services.flow_context import FlowContext


class ModerationHandler:
    """Handles content moderation (pre/post)."""

    def __init__(self, provider: BaseModerationProvider | None):
        self.provider = provider

    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Run content moderation."""
        if not self.provider:
            raise ValueError(
                "Flow step 'moderation' requires 'moderationConfig' in tenant config"
            )

        mode = step.mode or "pre"
        step_name = f"moderation:{mode}"

        if mode == "pre":
            # Check the user's input query
            result = await self.provider.check(ctx.query)
            if result.is_flagged:
                raise ContentFlaggedError(result)
            ctx.moderation_result = result
        elif mode == "post":
            # Check the AI-generated answer
            if ctx.llm_response is None:
                raise ValueError("Moderation 'post' requires a prior LLM response")
            result = await self.provider.check(ctx.llm_response)
            if result.is_flagged:
                # Replace the answer with a safe message instead of raising
                ctx.llm_response = (
                    "The generated response was flagged by content moderation "
                    "and has been removed."
                )
            ctx.metadata["post_moderation"] = {
                "is_flagged": result.is_flagged,
            }
        else:
            raise ValueError(f"Unknown moderation mode: {mode!r}")

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                step_name,
                {"is_flagged": result.is_flagged, "mode": mode},
            )
        return ctx
