"""Unit coverage for bounded persisted Event following loops."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.replay import PersistedEventFollower
from app.langgraph_v2.run_events import EventRecord, RunEventRepository, RunRecord


class ClockSkewRepository(RunEventRepository):
    """Expose an app-expired lease whose database CAS keeps rejecting expiry."""

    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.run = RunRecord(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            status="running",
            owner_instance_id="owner",
            execution_epoch=1,
            heartbeat_at=now,
            expires_at=now - timedelta(seconds=1),
            created_at=now,
        )
        self.cas_attempts = 0

    async def get_run(self, tenant_id: str, run_id: UUID) -> RunRecord:
        return self.run

    async def list_events_after(
        self, tenant_id: str, run_id: UUID, *, after_sequence: int
    ) -> list[EventRecord]:
        return []

    async def interrupt_expired_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        observed_execution_epoch: int,
    ) -> EventRecord | None:
        self.cas_attempts += 1
        return None


@pytest.mark.asyncio
async def test_failed_expiry_cas_keeps_bounded_polling_latency() -> None:
    repository = ClockSkewRepository()
    follower = PersistedEventFollower(
        repository,
        LiveEventWakeups(),
        poll_interval_seconds=0.02,
    ).follow(
        tenant_id="tenant-a",
        run_id=repository.run.run_id,
        after_sequence=0,
    )
    pending = asyncio.ensure_future(anext(follower))

    await asyncio.sleep(0.055)
    pending.cancel()
    with suppress(asyncio.CancelledError):
        await pending

    assert 1 <= repository.cas_attempts <= 4
