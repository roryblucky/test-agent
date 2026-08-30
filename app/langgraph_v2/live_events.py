"""Best-effort wakeups for durable, PostgreSQL-backed Run Event followers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis

_CHANNEL = "langgraph_v2:run-events"
_CANCELLATION_CHANNEL = "langgraph_v2:run-cancellations"


@dataclass
class _WakeupSlot:
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    generation: int = 0
    subscribers: int = 0


class LiveEventSubscription:
    """One bounded, Run-scoped local wakeup subscription."""

    def __init__(self, slot: _WakeupSlot) -> None:
        self._slot = slot

    @property
    def generation(self) -> int:
        """Return the current best-effort signal generation."""
        return self._slot.generation

    async def wait(self, observed_generation: int, *, timeout_seconds: float) -> int:
        """Wait for a signal, or return after bounded polling latency."""
        async with self._slot.condition:
            if self._slot.generation != observed_generation:
                return self._slot.generation
            try:
                await asyncio.wait_for(
                    self._slot.condition.wait(), timeout=timeout_seconds
                )
            except TimeoutError:
                pass
            return self._slot.generation


class LiveEventWakeups:
    """Wake local followers and optionally relay wakeups through Redis.

    Redis carries no Event payload and is never authoritative: followers always
    re-read the next persisted sequence from PostgreSQL after each wakeup.
    """

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        instance_id: str | None = None,
    ) -> None:
        self._slots: dict[tuple[str, UUID], _WakeupSlot] = {}
        self._cancellation_slots: dict[tuple[str, UUID], _WakeupSlot] = {}
        self._slots_lock = asyncio.Lock()
        self._redis_url = redis_url
        self._instance_id = instance_id
        self._redis: Any | None = None
        self._pubsub: Any | None = None
        self._listener: asyncio.Task[None] | None = None

    @property
    def active_subscription_count(self) -> int:
        """Return the number of active run-event subscription slots."""
        return len(self._slots)

    async def start(self) -> None:
        """Begin receiving remote best-effort wakeups without blocking startup."""
        if self._redis_url is not None and self._listener is None:
            self._listener = asyncio.create_task(self._listen_to_redis())

    async def close(self) -> None:
        """Release optional Redis resources and local listener state."""
        if self._listener is not None:
            self._listener.cancel()
            with suppress(asyncio.CancelledError):
                await self._listener
            self._listener = None
        if self._pubsub is not None:
            with suppress(Exception):
                await self._pubsub.aclose()
            self._pubsub = None
        if self._redis is not None:
            with suppress(Exception):
                await self._redis.aclose()
            self._redis = None

    @asynccontextmanager
    async def subscribe(
        self, tenant_id: str, run_id: UUID
    ) -> AsyncGenerator[LiveEventSubscription]:
        """Retain local state only while at least one follower is active."""
        async with self._subscribe(self._slots, tenant_id, run_id) as subscription:
            yield subscription

    @asynccontextmanager
    async def subscribe_cancellation(
        self, tenant_id: str, run_id: UUID
    ) -> AsyncGenerator[LiveEventSubscription]:
        """Retain one owner-local cancellation signal subscription."""
        async with self._subscribe(
            self._cancellation_slots, tenant_id, run_id
        ) as subscription:
            yield subscription

    @asynccontextmanager
    async def _subscribe(
        self,
        slots: dict[tuple[str, UUID], _WakeupSlot],
        tenant_id: str,
        run_id: UUID,
    ) -> AsyncGenerator[LiveEventSubscription]:
        key = (tenant_id, run_id)
        async with self._slots_lock:
            slot = slots.setdefault(key, _WakeupSlot())
            slot.subscribers += 1
        try:
            yield LiveEventSubscription(slot)
        finally:
            async with self._slots_lock:
                slot.subscribers -= 1
                if slot.subscribers == 0 and slots.get(key) is slot:
                    slots.pop(key)

    async def publish(self, tenant_id: str, run_id: UUID) -> None:
        """Wake local followers and best-effort relay the wakeup remotely."""
        await self._wake_local(tenant_id, run_id)
        if self._redis_url is None:
            return
        try:
            redis = self._redis_client()
            await redis.publish(
                _CHANNEL,
                json.dumps({"tenant_id": tenant_id, "run_id": str(run_id)}),
            )
        except Exception:
            # PostgreSQL polling is the loss-tolerant fallback.
            return

    async def publish_cancellation(
        self,
        tenant_id: str,
        run_id: UUID,
        *,
        owner_instance_id: str,
    ) -> None:
        """Best-effort wake only the instance owning the current Run claim."""
        if self._instance_id == owner_instance_id:
            await self._wake_local_cancellation(tenant_id, run_id)
        if self._redis_url is None or not owner_instance_id:
            return
        try:
            redis = self._redis_client()
            await redis.publish(
                _CANCELLATION_CHANNEL,
                json.dumps(
                    {
                        "kind": "cancellation",
                        "owner_instance_id": owner_instance_id,
                        "tenant_id": tenant_id,
                        "run_id": str(run_id),
                    }
                ),
            )
        except Exception:
            # The persisted intent remains authoritative.
            return

    async def _wake_local(self, tenant_id: str, run_id: UUID) -> None:
        await self._wake_slot(self._slots, tenant_id, run_id)

    async def _wake_local_cancellation(self, tenant_id: str, run_id: UUID) -> None:
        await self._wake_slot(self._cancellation_slots, tenant_id, run_id)

    async def _wake_slot(
        self,
        slots: dict[tuple[str, UUID], _WakeupSlot],
        tenant_id: str,
        run_id: UUID,
    ) -> None:
        key = (tenant_id, run_id)
        async with self._slots_lock:
            slot = slots.get(key)
        if slot is None:
            return
        async with slot.condition:
            slot.generation += 1
            slot.condition.notify_all()

    def _redis_client(self) -> Any:
        if self._redis is None:
            if self._redis_url is None:
                raise RuntimeError("Redis wakeups are not configured")
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=0.1,
                socket_timeout=0.1,
            )
        return self._redis

    async def _listen_to_redis(self) -> None:
        """Mirror remote signal payloads locally; retry after Redis loss."""
        while True:
            try:
                redis = self._redis_client()
                pubsub = redis.pubsub()
                self._pubsub = pubsub
                await pubsub.subscribe(_CHANNEL, _CANCELLATION_CHANNEL)
                async for message in pubsub.listen():
                    if message.get("type") != "message":
                        continue
                    payload = json.loads(message["data"])
                    tenant_id = str(payload["tenant_id"])
                    run_id = UUID(str(payload["run_id"]))
                    if payload.get("kind") == "cancellation":
                        if payload.get("owner_instance_id") == self._instance_id:
                            await self._wake_local_cancellation(tenant_id, run_id)
                    else:
                        await self._wake_local(tenant_id, run_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(0.25)
            finally:
                if self._pubsub is not None:
                    with suppress(Exception):
                        await self._pubsub.aclose()
                    self._pubsub = None
