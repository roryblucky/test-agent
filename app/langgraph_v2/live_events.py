"""Best-effort wakeups for durable, PostgreSQL-backed Run Event followers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis

_CHANNEL = "langgraph_v2:run-events"


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
    ) -> None:
        self._slots: dict[tuple[str, UUID], _WakeupSlot] = {}
        self._slots_lock = asyncio.Lock()
        self._redis_url = redis_url
        self._redis: Any | None = None
        self._pubsub: Any | None = None
        self._listener: asyncio.Task[None] | None = None

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
    ) -> AsyncIterator[LiveEventSubscription]:
        """Retain local state only while at least one follower is active."""
        key = (tenant_id, run_id)
        async with self._slots_lock:
            slot = self._slots.setdefault(key, _WakeupSlot())
            slot.subscribers += 1
        try:
            yield LiveEventSubscription(slot)
        finally:
            async with self._slots_lock:
                slot.subscribers -= 1
                if slot.subscribers == 0 and self._slots.get(key) is slot:
                    self._slots.pop(key)

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

    async def _wake_local(self, tenant_id: str, run_id: UUID) -> None:
        key = (tenant_id, run_id)
        async with self._slots_lock:
            slot = self._slots.get(key)
        if slot is None:
            return
        async with slot.condition:
            slot.generation += 1
            slot.condition.notify_all()

    def _redis_client(self) -> Any:
        if self._redis is None:
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
                self._pubsub = redis.pubsub()
                await self._pubsub.subscribe(_CHANNEL)
                async for message in self._pubsub.listen():
                    if message.get("type") != "message":
                        continue
                    payload = json.loads(message["data"])
                    await self._wake_local(
                        str(payload["tenant_id"]), UUID(str(payload["run_id"]))
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(0.25)
            finally:
                if self._pubsub is not None:
                    with suppress(Exception):
                        await self._pubsub.aclose()
                    self._pubsub = None
