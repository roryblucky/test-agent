"""Owner-local observation tests for durable cancellation intent."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.langgraph_v2.cancellation import CancellationObserver
from app.langgraph_v2.live_events import LiveEventWakeups


class _IntentRepository:
    def __init__(self, *, requested_after: int) -> None:
        self.calls = 0
        self.requested_after = requested_after
        self.requested = asyncio.Event()

    async def is_requested(self, **kwargs: object) -> bool:
        del kwargs
        self.calls += 1
        if self.calls >= self.requested_after:
            self.requested.set()
            return True
        return False


@pytest.mark.asyncio
async def test_observer_polls_without_redis_until_postgres_confirms_intent() -> None:
    repository = _IntentRepository(requested_after=2)
    wakeups = LiveEventWakeups(instance_id="owner")
    observer = CancellationObserver(
        repository,  # type: ignore[arg-type]
        wakeups,
        tenant_id="tenant-a",
        run_id=uuid4(),
        poll_interval_seconds=0.01,
    )

    await observer.start()
    await asyncio.wait_for(repository.requested.wait(), timeout=1)
    await observer.close()

    assert repository.calls == 2


@pytest.mark.asyncio
async def test_owner_wakeup_triggers_authoritative_check_before_poll_timeout() -> None:
    repository = _IntentRepository(requested_after=2)
    wakeups = LiveEventWakeups(instance_id="owner")
    run_id = uuid4()
    observer = CancellationObserver(
        repository,  # type: ignore[arg-type]
        wakeups,
        tenant_id="tenant-a",
        run_id=run_id,
        poll_interval_seconds=10,
    )

    await observer.start()
    while repository.calls == 0:
        await asyncio.sleep(0)
    await wakeups.publish_cancellation("tenant-a", run_id, owner_instance_id="owner")
    await asyncio.wait_for(repository.requested.wait(), timeout=1)
    await observer.close()

    assert repository.calls == 2
