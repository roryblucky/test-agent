"""Unit coverage for loss-tolerant live-event wakeups."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.langgraph_v2.live_events import LiveEventWakeups


@pytest.mark.asyncio
async def test_local_notification_wakes_waiter_for_its_run() -> None:
    wakeups = LiveEventWakeups()
    run_id = uuid4()
    generation = wakeups.generation("tenant-a", run_id)
    waiter = asyncio.create_task(
        wakeups.wait("tenant-a", run_id, generation, timeout_seconds=1)
    )

    await asyncio.sleep(0)
    await wakeups.publish("tenant-a", run_id)

    assert await waiter > generation


@pytest.mark.asyncio
async def test_wait_returns_after_bounded_poll_timeout_without_notification() -> None:
    wakeups = LiveEventWakeups()
    run_id = uuid4()
    generation = wakeups.generation("tenant-a", run_id)

    assert (
        await wakeups.wait("tenant-a", run_id, generation, timeout_seconds=0.001)
        == generation
    )


@pytest.mark.asyncio
async def test_wait_does_not_miss_notification_before_subscription() -> None:
    wakeups = LiveEventWakeups()
    run_id = uuid4()
    generation = wakeups.generation("tenant-a", run_id)
    await wakeups.publish("tenant-a", run_id)

    assert (
        await wakeups.wait("tenant-a", run_id, generation, timeout_seconds=1)
        == generation + 1
    )
