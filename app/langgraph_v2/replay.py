"""Tenant-scoped replay and loss-tolerant following of persisted v2 Events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID

from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.observability import observe, safe_span_attribute
from app.langgraph_v2.run_events import (
    EventNotFound,
    EventRecord,
    RunEventRepository,
)

FOLLOW_POLL_INTERVAL_SECONDS = 0.25


class PersistedEventReplay:
    """Read one immutable Event snapshot without starting or following a Run."""

    def __init__(self, repository: RunEventRepository) -> None:
        self._repository = repository

    async def snapshot(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        after_sequence: int,
    ) -> list[EventRecord]:
        """Return only events published after the requested Run-local sequence."""
        with observe(
            "replay.snapshot",
            run_id=run_id,
            attributes={"replay.mode": "snapshot"},
        ):
            return await self._repository.list_events_after(
                tenant_id,
                run_id,
                after_sequence=after_sequence,
            )


class PersistedEventFollower:
    """Replay then follow one Run through durable Event sequence reconciliation."""

    def __init__(
        self,
        repository: RunEventRepository,
        wakeups: LiveEventWakeups,
        *,
        poll_interval_seconds: float = FOLLOW_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._repository = repository
        self._wakeups = wakeups
        self._poll_interval_seconds = poll_interval_seconds

    async def follow(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        after_sequence: int,
        expected_execution_epoch: int | None = None,
    ) -> AsyncIterator[EventRecord]:
        """Yield each Event once, then close as soon as the Run is not running."""
        with observe(
            "replay.follow",
            run_id=run_id,
            execution_epoch=expected_execution_epoch,
            attributes={"replay.mode": "live"},
        ):
            cursor = after_sequence
            attached_run = await self._repository.get_run(tenant_id, run_id)
            followed_epoch = (
                attached_run.execution_epoch
                if expected_execution_epoch is None
                else expected_execution_epoch
            )
            async with self._wakeups.subscribe(tenant_id, run_id) as subscription:
                while True:
                    # Read the local generation before PostgreSQL so a producer cannot
                    # signal in the replay-to-wait handoff without waking this follower.
                    generation = subscription.generation
                    run_before = await self._repository.get_run(tenant_id, run_id)
                    events = await self._repository.list_events_after(
                        tenant_id,
                        run_id,
                        after_sequence=cursor,
                    )
                    run_after = await self._repository.get_run(tenant_id, run_id)

                    if (
                        run_before.execution_epoch != followed_epoch
                        or run_after.execution_epoch != followed_epoch
                    ):
                        # Re-read after observing the epoch transition.  Its durable
                        # interruption Event is now visible, but no Event from the
                        # replacement epoch may escape through this old follower.
                        events = await self._repository.list_events_after(
                            tenant_id,
                            run_id,
                            after_sequence=cursor,
                        )
                        for event in await _events_through_interruption(
                            self._repository,
                            tenant_id=tenant_id,
                            run_id=run_id,
                            events=events,
                            followed_epoch=followed_epoch,
                        ):
                            yield event
                        return

                    for event in events:
                        cursor = event.sequence
                        yield event

                    if run_after.status != "running":
                        # Status and its terminal Event are committed atomically, but
                        # the commit may land between the Event and status reads.
                        final_events = await self._repository.list_events_after(
                            tenant_id,
                            run_id,
                            after_sequence=cursor,
                        )
                        final_run = await self._repository.get_run(tenant_id, run_id)
                        if final_run.execution_epoch != followed_epoch:
                            final_events = await self._repository.list_events_after(
                                tenant_id,
                                run_id,
                                after_sequence=cursor,
                            )
                            final_events = await _events_through_interruption(
                                self._repository,
                                tenant_id=tenant_id,
                                run_id=run_id,
                                events=final_events,
                                followed_epoch=followed_epoch,
                            )
                        for event in final_events:
                            yield event
                        return

                    if run_after.expires_at <= datetime.now(UTC):
                        with observe(
                            "recovery.interrupt_expired",
                            run_id=run_id,
                            execution_epoch=followed_epoch,
                            attributes={"recovery.operation": "interrupt_expired"},
                        ) as recovery_span:
                            interrupted = await self._repository.interrupt_expired_run(
                                tenant_id=tenant_id,
                                run_id=run_id,
                                observed_execution_epoch=followed_epoch,
                            )
                            if interrupted is not None:
                                safe_span_attribute(
                                    recovery_span,
                                    "run.status",
                                    "interrupted",
                                )
                        if interrupted is None:
                            # PostgreSQL is authoritative for lease expiry.  If the
                            # application clock reached expiry first (or another CAS
                            # won), retain bounded polling instead of spinning.
                            await subscription.wait(
                                generation,
                                timeout_seconds=self._poll_interval_seconds,
                            )
                        continue
                    if events:
                        continue
                    await subscription.wait(
                        generation,
                        timeout_seconds=self._poll_interval_seconds,
                    )


async def _events_through_interruption(
    repository: RunEventRepository,
    *,
    tenant_id: str,
    run_id: UUID,
    events: list[EventRecord],
    followed_epoch: int,
) -> list[EventRecord]:
    """Return the old epoch through its boundary, independent of the cursor."""
    interruption_key = f"lifecycle:interrupted:{followed_epoch}"
    try:
        interruption = await repository.get_event(tenant_id, run_id, interruption_key)
    except EventNotFound:
        return []
    before_boundary = [
        event for event in events if event.sequence <= interruption.sequence
    ]
    if all(event.event_key != interruption_key for event in before_boundary):
        before_boundary.append(interruption)
    return before_boundary
