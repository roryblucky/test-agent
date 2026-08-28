"""Replay-safe output moderation phase for the v2 linear graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultInput
from app.langgraph_v2.pre_moderation import ModerationDecision, ModerationProvider
from app.langgraph_v2.run_events import EventInput, EventRecord

SAFE_MODERATION_MESSAGE = (
    "The generated response was flagged by content moderation and has been removed."
)


async def run_post_moderation(
    state: Mapping[str, Any],
    *,
    context: PhaseExecutionContext,
    provider: ModerationProvider,
) -> tuple[list[EventRecord], ModerationDecision | None, str | None, bool, str | None]:
    """Moderate the generated answer and journal only its safe publication state."""

    async def invoke() -> PhaseResultInput:
        answer = state.get("answer")
        if not isinstance(answer, str) or not answer:
            message = "Post-moderation requires a generated answer."
            return PhaseResultInput(
                phase_name="post_moderation",
                normalized_result={"failed": True, "error": message},
                events=(
                    EventInput(
                        event_key="phase:post_moderation:step_start:1",
                        type="step_start",
                        step="moderation:post",
                    ),
                    EventInput(
                        event_key="phase:post_moderation:error:1",
                        type="error",
                        data=message,
                    ),
                ),
                terminal_status="failed",
            )
        try:
            decision = await provider.check(answer)
        except Exception as exc:
            message = str(exc) or "Post-moderation failed."
            return PhaseResultInput(
                phase_name="post_moderation",
                normalized_result={"failed": True, "error": message},
                events=(
                    EventInput(
                        event_key="phase:post_moderation:step_start:1",
                        type="step_start",
                        step="moderation:post",
                    ),
                    EventInput(
                        event_key="phase:post_moderation:error:1",
                        type="error",
                        data=message,
                    ),
                ),
                terminal_status="failed",
            )
        safe_answer = SAFE_MODERATION_MESSAGE if decision.is_flagged else answer
        return PhaseResultInput(
            phase_name="post_moderation",
            normalized_result={
                "decision": decision.model_dump(exclude_none=True),
                "safe_answer": safe_answer,
            },
            events=(
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
            ),
        )

    result = await context.repository.get_or_invoke(
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        owner_instance_id=context.owner_instance_id,
        execution_epoch=context.execution_epoch,
        phase_name="post_moderation",
        invoke=invoke,
    )
    if result.normalized_result.get("failed") is True:
        return (
            list(result.events),
            None,
            None,
            True,
            str(result.normalized_result["error"]),
        )
    decision = ModerationDecision.model_validate(result.normalized_result["decision"])
    return (
        list(result.events),
        decision,
        str(result.normalized_result["safe_answer"]),
        False,
        None,
    )
