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
        followed_epoch: int | None = None
        while True:
            # Read the local generation before PostgreSQL so a producer cannot
            # signal in the replay-to-wait handoff without waking this follower.
            generation = self._wakeups.generation(tenant_id, run_id)
            events = await self._repository.list_events_after(
                tenant_id,
                run_id,
                after_sequence=cursor,
            )
            for event in events:
                cursor = event.sequence
                yield event
            if events:
                # A producer can commit the next Event while the caller is
                # consuming this batch.  Reconcile its sequence before looking
                # at terminal state, or that boundary Event could be skipped.
                continue

            run = await self._repository.get_run(tenant_id, run_id)
            if followed_epoch is None:
                followed_epoch = run.execution_epoch
            elif run.execution_epoch != followed_epoch:
                # An explicit recovery owns the next epoch.  This follower
                # observes only the epoch it attached to and never competes.
                return
            if run.status != "running":
                return
            if run.expires_at <= datetime.now(UTC):
                await self._repository.interrupt_expired_run(
                    tenant_id=tenant_id,
                    run_id=run_id,
                    observed_execution_epoch=run.execution_epoch,
                )
                continue
            await self._wakeups.wait(
                tenant_id,
                run_id,
                generation,
                timeout_seconds=self._poll_interval_seconds,
            )
