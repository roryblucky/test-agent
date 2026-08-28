"""Clean-room pre-moderation actor used by the v2 linear graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultInput
from app.langgraph_v2.run_events import EventInput


class ModerationDecision(BaseModel):
    """Normalized moderation output safe to store in the phase journal."""

    is_flagged: bool
    categories: dict[str, float] = Field(default_factory=dict)
    reason: str | None = None


class ModerationProvider(Protocol):
    """Provider seam for checking the original user query."""

    async def check(self, text: str) -> ModerationDecision:
        """Return a normalized moderation decision."""
        ...


class MockModerationProvider:
    """Deterministic POC provider with an explicit blocked-word policy."""

    async def check(self, text: str) -> ModerationDecision:
        """Flag queries containing a small deterministic blocked-word set."""
        blocked = next(
            (
                word
                for word in ("blocked", "unsafe", "forbidden")
                if word in text.lower()
            ),
            None,
        )
        if blocked is None:
            return ModerationDecision(is_flagged=False)
        return ModerationDecision(
            is_flagged=True,
            categories={"policy": 1.0},
            reason=f"query contains blocked term: {blocked}",
        )


async def run_pre_moderation(
    state: Mapping[str, Any],
    *,
    context: PhaseExecutionContext,
    provider: ModerationProvider,
) -> tuple[list[dict[str, Any]], bool, ModerationDecision]:
    """Journal and return the pre-moderation Events and halt decision."""

    async def invoke() -> PhaseResultInput:
        decision = await provider.check(state["query"])
        events = [
            EventInput(
                event_key="phase:pre_moderation:step_start:1",
                type="step_start",
                step="pre_moderation",
            ),
            EventInput(
                event_key="phase:pre_moderation:step_completed:1",
                type="step_completed",
                step="pre_moderation",
                data=decision.model_dump(exclude_none=True),
            ),
        ]
        if decision.is_flagged:
            events.append(
                EventInput(
                    event_key="phase:pre_moderation:error:1",
                    type="error",
                    step="pre_moderation",
                    data={
                        "message": "Your query was flagged by content moderation.",
                        "moderation": decision.model_dump(exclude_none=True),
                    },
                )
            )
        return PhaseResultInput(
            phase_name="pre_moderation",
            normalized_result=decision.model_dump(exclude_none=True),
            events=tuple(events),
        )

    result = await context.repository.get_or_invoke(
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        owner_instance_id=context.owner_instance_id,
        execution_epoch=context.execution_epoch,
        phase_name="pre_moderation",
        invoke=invoke,
    )
    decision = ModerationDecision.model_validate(result.normalized_result)
    return (
        [
            {
                "event_key": event.event_key,
                "type": event.type,
                "step": event.step,
                "data": event.data,
                "sequence": event.sequence,
            }
            for event in result.events
        ],
        decision.is_flagged,
        decision,
    )
