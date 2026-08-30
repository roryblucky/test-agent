"""Tenant-scoped durable cancellation requests for v2 Runs."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.run_events import (
    ClaimFenced,
    EventInput,
    EventRecord,
    RunNotFound,
    _canonical_envelope,  # pyright: ignore[reportPrivateUsage] -- shared journal primitive
)

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class CancellationIntentRecord(BaseModel):
    """One durable request to cooperatively stop a Run."""

    tenant_id: str
    run_id: UUID
    requested_at: datetime


class CancellationRequestResult(BaseModel):
    """Outcome of atomically inspecting a Run and recording its intent."""

    run_id: UUID
    run_status: str
    owner_instance_id: str
    intent: CancellationIntentRecord | None = None

    @property
    def accepted(self) -> bool:
        """Return whether a durable cancellation intent exists."""
        return self.intent is not None


class CancellationRepository:
    """Persist cancellation intent while holding the tenant-scoped Run lock."""

    def __init__(
        self,
        pool: AsyncConnectionPool[Any],
        *,
        wakeups: LiveEventWakeups | None = None,
    ) -> None:
        self._pool = pool
        self._wakeups = wakeups

    async def request(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
    ) -> CancellationRequestResult:
        """Record one idempotent intent without changing Run or Event state."""
        intent_row = None
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        SELECT status, owner_instance_id
                        FROM langgraph_v2.runs
                        WHERE tenant_id = %s AND run_id = %s
                        FOR UPDATE
                        """,
                        (tenant_id, run_id),
                    )
                    run = await cursor.fetchone()
                    if run is None:
                        raise RunNotFound(str(run_id))
                    if run["status"] not in _TERMINAL_STATUSES:
                        await cursor.execute(
                            """
                            INSERT INTO langgraph_v2.cancellation_intents (
                                tenant_id, run_id
                            ) VALUES (%s, %s)
                            ON CONFLICT (tenant_id, run_id) DO NOTHING
                            """,
                            (tenant_id, run_id),
                        )
                        await cursor.execute(
                            """
                            SELECT tenant_id, run_id, requested_at
                            FROM langgraph_v2.cancellation_intents
                            WHERE tenant_id = %s AND run_id = %s
                            """,
                            (tenant_id, run_id),
                        )
                        intent_row = await cursor.fetchone()

        result = CancellationRequestResult(
            run_id=run_id,
            run_status=run["status"],
            owner_instance_id=run["owner_instance_id"],
            intent=(
                CancellationIntentRecord.model_validate(intent_row)
                if intent_row is not None
                else None
            ),
        )
        if result.accepted and self._wakeups is not None:
            await self._wakeups.publish_cancellation(
                tenant_id,
                run_id,
                owner_instance_id=result.owner_instance_id,
            )
        return result

    async def is_requested(self, *, tenant_id: str, run_id: UUID) -> bool:
        """Read the authoritative tenant-scoped cancellation intent."""
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT 1
                    FROM langgraph_v2.cancellation_intents
                    WHERE tenant_id = %s AND run_id = %s
                    """,
                    (tenant_id, run_id),
                )
                return await cursor.fetchone() is not None

    async def apply_if_requested(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
    ) -> EventRecord | None:
        """Atomically stop one matching owner after its intent is durable."""
        event_key = f"lifecycle:cancelled:{execution_epoch}"
        persisted: EventRecord | None = None
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        SELECT status, owner_instance_id, execution_epoch,
                               next_event_sequence,
                               expires_at > clock_timestamp() AS claim_active
                        FROM langgraph_v2.runs
                        WHERE tenant_id = %s AND run_id = %s
                        FOR UPDATE
                        """,
                        (tenant_id, run_id),
                    )
                    run = await cursor.fetchone()
                    if run is None:
                        raise RunNotFound(str(run_id))
                    if run["status"] == "cancelled":
                        await cursor.execute(
                            """
                            SELECT tenant_id, run_id, sequence, event_key, type,
                                   step, data, created_at
                            FROM langgraph_v2.events
                            WHERE tenant_id = %s AND run_id = %s
                              AND event_key = %s
                            """,
                            (tenant_id, run_id, event_key),
                        )
                        prior = await cursor.fetchone()
                        if prior is None:
                            raise ClaimFenced(str(run_id))
                        return EventRecord.model_validate(prior)
                    if (
                        run["status"] != "running"
                        or run["owner_instance_id"] != owner_instance_id
                        or run["execution_epoch"] != execution_epoch
                        or not run["claim_active"]
                    ):
                        raise ClaimFenced(str(run_id))
                    await cursor.execute(
                        """
                        SELECT 1
                        FROM langgraph_v2.cancellation_intents
                        WHERE tenant_id = %s AND run_id = %s
                        """,
                        (tenant_id, run_id),
                    )
                    if await cursor.fetchone() is None:
                        return None
                    event = EventInput(
                        event_key=event_key,
                        type="stopped",
                        data={"partial": None},
                    )
                    await cursor.execute(
                        """
                        INSERT INTO langgraph_v2.events (
                            tenant_id, run_id, sequence, event_key, type, step,
                            data, canonical_envelope
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING tenant_id, run_id, sequence, event_key, type,
                                  step, data, created_at
                        """,
                        (
                            tenant_id,
                            run_id,
                            run["next_event_sequence"],
                            event.event_key,
                            event.type,
                            event.step,
                            Jsonb(event.data),
                            _canonical_envelope(event),
                        ),
                    )
                    persisted = EventRecord.model_validate(await cursor.fetchone())
                    await cursor.execute(
                        """
                        UPDATE langgraph_v2.runs
                        SET status = 'cancelled',
                            terminal_outcome = %s,
                            completed_at = NULL,
                            owner_instance_id = '',
                            heartbeat_at = clock_timestamp(),
                            expires_at = clock_timestamp(),
                            next_event_sequence = next_event_sequence + 1
                        WHERE tenant_id = %s AND run_id = %s
                          AND status = 'running'
                          AND owner_instance_id = %s
                          AND execution_epoch = %s
                        """,
                        (
                            Jsonb({"status": "cancelled"}),
                            tenant_id,
                            run_id,
                            owner_instance_id,
                            execution_epoch,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ClaimFenced(str(run_id))
        if self._wakeups is not None:
            await self._wakeups.publish(tenant_id, run_id)
        return persisted


class CancellationObserver:
    """Cache only PostgreSQL-confirmed cancellation signals for one owner."""

    def __init__(
        self,
        repository: CancellationRepository,
        wakeups: LiveEventWakeups,
        *,
        tenant_id: str,
        run_id: UUID,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        self._repository = repository
        self._wakeups = wakeups
        self._tenant_id = tenant_id
        self._run_id = run_id
        self._poll_interval_seconds = poll_interval_seconds
        self._observed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start best-effort Redis wake handling with bounded DB polling."""
        if self._task is None:
            self._task = asyncio.create_task(self._watch())

    async def close(self) -> None:
        """Stop the owner-local observer."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def is_requested(self) -> bool:
        """Check PostgreSQL at a boundary unless a prior read confirmed intent."""
        if self._observed.is_set():
            return True
        requested = await self._repository.is_requested(
            tenant_id=self._tenant_id,
            run_id=self._run_id,
        )
        if requested:
            self._observed.set()
        return requested

    async def _watch(self) -> None:
        async with self._wakeups.subscribe_cancellation(
            self._tenant_id, self._run_id
        ) as subscription:
            generation = subscription.generation
            while not await self.is_requested():
                generation = await subscription.wait(
                    generation,
                    timeout_seconds=self._poll_interval_seconds,
                )
