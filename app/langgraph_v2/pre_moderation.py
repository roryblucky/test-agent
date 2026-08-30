"""Clean-room pre-moderation actor used by the v2 linear graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.langgraph_v2.contracts import LiveStreamEvent


class ModerationDecision(BaseModel):
    """Normalized moderation output carried in Graph State."""

    is_flagged: bool
    categories: dict[str, float] = Field(default_factory=dict)
    reason: str | None = None


def moderation_error_message(decision: ModerationDecision) -> str:
    """Match the legacy ContentFlaggedError message."""
    detail = decision.reason or decision.categories
    return f"Content flagged by moderation: {detail}"


def pre_moderation_events(decision: ModerationDecision) -> tuple[LiveStreamEvent, ...]:
    """Build the stable stream Events for one moderation decision."""
    events = [
        LiveStreamEvent(
            type="step_start",
            step="moderation:pre",
        )
    ]
    if not decision.is_flagged:
        events.append(
            LiveStreamEvent(
                type="step_completed",
                step="moderation:pre",
                data={"is_flagged": False, "mode": "pre"},
            )
        )
    else:
        events.append(
            LiveStreamEvent(
                type="error",
                data=moderation_error_message(decision),
                checkpoint_terminal=True,
            )
        )
    return tuple(events)


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
    provider: ModerationProvider,
) -> tuple[tuple[LiveStreamEvent, ...], bool, ModerationDecision]:
    """Return the pre-moderation State update without an application journal."""
    try:
        decision = await provider.check(state["query"])
    except Exception as exc:
        message = str(exc) or "Moderation failed."
        decision = ModerationDecision(is_flagged=True, reason=message)
        return (
            (
                LiveStreamEvent(
                    type="step_start",
                    step="moderation:pre",
                ),
                LiveStreamEvent(
                    type="error",
                    data=message,
                    checkpoint_terminal=True,
                ),
            ),
            True,
            decision,
        )
    return pre_moderation_events(decision), decision.is_flagged, decision
