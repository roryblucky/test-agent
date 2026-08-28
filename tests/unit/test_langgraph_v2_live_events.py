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
    async with wakeups.subscribe("tenant-a", run_id) as subscription:
        generation = subscription.generation
        waiter = asyncio.create_task(subscription.wait(generation, timeout_seconds=1))

        await asyncio.sleep(0)
        await wakeups.publish("tenant-a", run_id)

        assert await waiter > generation


@pytest.mark.asyncio
async def test_wait_returns_after_bounded_poll_timeout_without_notification() -> None:
    wakeups = LiveEventWakeups()
    run_id = uuid4()
    async with wakeups.subscribe("tenant-a", run_id) as subscription:
        generation = subscription.generation

        assert await subscription.wait(generation, timeout_seconds=0.001) == generation


@pytest.mark.asyncio
async def test_wait_does_not_miss_notification_before_subscription() -> None:
    wakeups = LiveEventWakeups()
    run_id = uuid4()
    async with wakeups.subscribe("tenant-a", run_id) as subscription:
        generation = subscription.generation
        await wakeups.publish("tenant-a", run_id)

        assert await subscription.wait(generation, timeout_seconds=1) == generation + 1


@pytest.mark.asyncio
async def test_inactive_runs_do_not_retain_local_wakeup_state() -> None:
    wakeups = LiveEventWakeups()
    run_id = uuid4()

    async with wakeups.subscribe("tenant-a", run_id):
        assert len(wakeups._slots) == 1

    await wakeups.publish("tenant-a", run_id)
    assert wakeups._slots == {}
