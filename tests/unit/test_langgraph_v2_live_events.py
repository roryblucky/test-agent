"""Unit coverage for loss-tolerant live-event wakeups."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
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
async def test_redis_publish_on_one_instance_wakes_another_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeRedisBroker()
    monkeypatch.setattr(
        live_events_module.aioredis,
        "from_url",
        lambda *args, **kwargs: broker.client(),
    )
    producer = LiveEventWakeups(redis_url="redis://test")
    consumer = LiveEventWakeups(redis_url="redis://test")
    run_id = uuid4()

    async with consumer.subscribe("tenant-a", run_id) as subscription:
        await consumer.start()
        await asyncio.wait_for(broker.subscribed.wait(), timeout=1)
        generation = subscription.generation
        await producer.publish("tenant-a", run_id)

        assert await subscription.wait(generation, timeout_seconds=1) == generation + 1

    await producer.close()
    await consumer.close()
    assert broker.published_channels == ["langgraph_v2:run-events"]
    assert all(client.closed for client in broker.clients)


@pytest.mark.asyncio
async def test_cancellation_wakeup_is_addressed_only_to_owning_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeRedisBroker()
    monkeypatch.setattr(
        live_events_module.aioredis,
        "from_url",
        lambda *args, **kwargs: broker.client(),
    )
    requester = LiveEventWakeups(redis_url="redis://test", instance_id="requester")
    owner = LiveEventWakeups(redis_url="redis://test", instance_id="owner")
    bystander = LiveEventWakeups(redis_url="redis://test", instance_id="bystander")
    run_id = uuid4()

    async with (
        owner.subscribe_cancellation("tenant-a", run_id) as owner_subscription,
        bystander.subscribe_cancellation("tenant-a", run_id) as bystander_subscription,
    ):
        await owner.start()
        await bystander.start()
        await asyncio.wait_for(broker.subscribed.wait(), timeout=1)
        await asyncio.sleep(0)
        owner_generation = owner_subscription.generation
        bystander_generation = bystander_subscription.generation

        await requester.publish_cancellation(
            "tenant-a",
            run_id,
            owner_instance_id="owner",
        )

        assert (
            await owner_subscription.wait(owner_generation, timeout_seconds=1)
            == owner_generation + 1
        )
        assert (
            await bystander_subscription.wait(
                bystander_generation, timeout_seconds=0.001
            )
            == bystander_generation
        )

    await requester.close()
    await owner.close()
    await bystander.close()
    assert broker.published_channels[-1] == "langgraph_v2:run-cancellations"


class FakePubSub:
    def __init__(self, broker: FakeRedisBroker) -> None:
        self._broker = broker
        self._messages: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        self.subscribed_channels: set[str] = set()

    async def subscribe(self, *channels: str) -> None:
        self.subscribed_channels.update(channels)
        self._broker.pubsubs.append(self)
        self._broker.subscribed.set()

    async def listen(self) -> AsyncIterator[dict[str, str]]:
        while True:
            yield await self._messages.get()

    async def aclose(self) -> None:
        pass


class FakeRedis:
    def __init__(self, broker: FakeRedisBroker) -> None:
        self._broker = broker
        self.closed = False

    def pubsub(self) -> FakePubSub:
        return FakePubSub(self._broker)

    async def publish(self, channel: str, data: str) -> None:
        self._broker.published_channels.append(channel)
        for pubsub in self._broker.pubsubs:
            if channel in pubsub.subscribed_channels:
                await pubsub._messages.put({"type": "message", "data": data})

    async def aclose(self) -> None:
        self.closed = True


class FakeRedisBroker:
    def __init__(self) -> None:
        self.clients: list[FakeRedis] = []
        self.pubsubs: list[FakePubSub] = []
        self.published_channels: list[str] = []
        self.subscribed = asyncio.Event()

    def client(self) -> FakeRedis:
        client = FakeRedis(self)
        self.clients.append(client)
        return client
