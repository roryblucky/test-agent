"""Tenant-scoped replay and loss-tolerant following of persisted v2 Events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID

from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.run_events import EventRecord, RunEventRepository

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
    ) -> AsyncIterator[EventRecord]:
        """Yield each Event once, then close as soon as the Run is not running."""
        cursor = after_sequence
        attached_run = await self._repository.get_run(tenant_id, run_id)
        followed_epoch = attached_run.execution_epoch
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
                    for event in _events_through_interruption(events, followed_epoch):
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
                        final_events = _events_through_interruption(
                            final_events, followed_epoch
                        )
                    for event in final_events:
                        yield event
                    return

                if run_after.expires_at <= datetime.now(UTC):
                    await self._repository.interrupt_expired_run(
                        tenant_id=tenant_id,
                        run_id=run_id,
                        observed_execution_epoch=followed_epoch,
                    )
                    continue
                if events:
                    continue
                await subscription.wait(
                    generation,
                    timeout_seconds=self._poll_interval_seconds,
                )


def _events_through_interruption(
    events: list[EventRecord], followed_epoch: int
) -> list[EventRecord]:
    """Keep the old epoch's boundary Event and fence all later Events."""
    interruption_key = f"lifecycle:interrupted:{followed_epoch}"
    for index, event in enumerate(events):
        if event.event_key == interruption_key:
            return events[: index + 1]
    return []
