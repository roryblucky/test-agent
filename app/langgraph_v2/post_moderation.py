"""Replay-safe output moderation phase for the v2 linear graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.langgraph_v2.output_assessments import (
    build_output_assessment_scope,
    record_output_assessment,
)
from app.langgraph_v2.phase_results import PhaseExecutionContext
from app.langgraph_v2.pre_moderation import ModerationDecision, ModerationProvider
from app.langgraph_v2.run_events import EventInput


async def run_post_moderation(
    state: Mapping[str, Any],
    *,
    context: PhaseExecutionContext,
    provider: ModerationProvider,
) -> tuple[list[EventInput], ModerationDecision | None, str | None]:
    """Assess the generated answer without changing its publication state."""
    assessment_scope = build_output_assessment_scope(
        tenant_id=context.tenant_id,
        conversation_id=(
            state.get("conversation_id")
            if isinstance(state.get("conversation_id"), str)
            else None
        ),
        turn_id=context.current_turn_id or state.get("turn_id"),
    )

    answer = state.get("answer")
    if not isinstance(answer, str) or not answer:
        message = "Post-moderation requires a generated answer."
        failed_result = {"failed": True, "error": message}
        await record_output_assessment(
            context.output_assessment_audit,
            scope=assessment_scope,
            assessment_type="post_moderation",
            result=failed_result,
        )
        return (
            [
                EventInput(
                    event_key="phase:post_moderation:step_start:1",
                    type="step_start",
                    step="moderation:post",
                ),
                EventInput(
                    event_key="phase:post_moderation:error:1",
                    type="step_completed",
                    step="moderation:post",
                    data=failed_result,
                ),
            ],
            None,
            message,
        )
    try:
        decision = await provider.check(answer)
    except Exception as exc:
        message = str(exc) or "Post-moderation failed."
        failed_result = {"failed": True, "error": message}
        await record_output_assessment(
            context.output_assessment_audit,
            scope=assessment_scope,
            assessment_type="post_moderation",
            result=failed_result,
        )
        return (
            [
                EventInput(
                    event_key="phase:post_moderation:step_start:1",
                    type="step_start",
                    step="moderation:post",
                ),
                EventInput(
                    event_key="phase:post_moderation:error:1",
                    type="step_completed",
                    step="moderation:post",
                    data=failed_result,
                ),
            ],
            None,
            message,
        )
    await record_output_assessment(
        context.output_assessment_audit,
        scope=assessment_scope,
        assessment_type="post_moderation",
        result={"decision": decision.model_dump(exclude_none=True)},
    )
    return (
        [
            EventInput(
                event_key="phase:post_moderation:step_start:1",
                type="step_start",
                step="moderation:post",
            ),
            EventInput(
                event_key="phase:post_moderation:step_completed:1",
                type="step_completed",
                step="moderation:post",
                data={"is_flagged": decision.is_flagged, "mode": "post"},
            ),
        ],
        decision,
        None,
    )
