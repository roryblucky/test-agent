"""Unit coverage for loss-tolerant live-event wakeups."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

import app.langgraph_v2.live_events as live_events_module
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


@pytest.mark.asyncio
async def test_redis_pubsub_message_wakes_matching_local_subscriber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(
        live_events_module.aioredis,
        "from_url",
        lambda *args, **kwargs: redis,
    )
    wakeups = LiveEventWakeups(redis_url="redis://test")
    run_id = uuid4()

    async with wakeups.subscribe("tenant-a", run_id) as subscription:
        redis.pubsub_instance.message = {
            "type": "message",
            "data": ('{"tenant_id":"tenant-a","run_id":"' + str(run_id) + '"}'),
        }
        generation = subscription.generation
        await wakeups.start()

        assert await subscription.wait(generation, timeout_seconds=1) == generation + 1

    await wakeups.close()
    assert redis.pubsub_instance.subscribed_channel == "langgraph_v2:run-events"
    assert redis.closed is True


class FakePubSub:
    def __init__(self) -> None:
        self.message: dict[str, str] | None = None
        self.subscribed_channel: str | None = None

    async def subscribe(self, channel: str) -> None:
        self.subscribed_channel = channel

    async def listen(self):
        if self.message is not None:
            yield self.message
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        pass


class FakeRedis:
    def __init__(self) -> None:
        self.pubsub_instance = FakePubSub()
        self.closed = False

    def pubsub(self) -> FakePubSub:
        return self.pubsub_instance

    async def aclose(self) -> None:
        self.closed = True
