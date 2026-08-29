"""Tenant-scoped persistence for minimal v2 Runs and Events."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from app.langgraph_v2.live_events import LiveEventWakeups


class RepositoryNotFound(LookupError):
    """A tenant-scoped repository lookup is indistinguishable from missing."""


class RunNotFound(RepositoryNotFound):
    """A Run is absent from the requested Tenant boundary."""


class EventNotFound(RepositoryNotFound):
    """An Event is absent from the requested Tenant boundary."""


class EventInvariantConflict(RuntimeError):
    """A stable Event key was reused with a different canonical envelope."""


class ClaimFenced(RuntimeError):
    """The supplied owner and execution epoch cannot write this Run."""


class CancellationObserved(RuntimeError):
    """A durable cancellation intent blocks the next publication boundary."""


class ResumeConflict(RuntimeError):
    """A Run cannot be resumed in its current state."""


CLAIM_LEASE_SECONDS = 30
CLAIM_HEARTBEAT_INTERVAL_SECONDS = CLAIM_LEASE_SECONDS // 3
TerminalStatus = Literal["completed", "failed"]


class EventInput(BaseModel):
    """One producer-keyed Event to append to a Run."""

    event_key: str = Field(min_length=1)
    type: str = Field(min_length=1)
    step: str | None = None
    data: Any = None


class RunRecord(BaseModel):
    """Persisted minimal Run state."""

    tenant_id: str
    run_id: UUID
    conversation_id: str
    status: str
    terminal_outcome: Any = None
    created_at: datetime
    completed_at: datetime | None = None
    owner_instance_id: str
    execution_epoch: int
    heartbeat_at: datetime
    expires_at: datetime
    checkpoint_id: str | None = None
    checkpoint_ns: str | None = None


class EventRecord(BaseModel):
    """Persisted Event with its Run-local sequence."""

    tenant_id: str
    run_id: UUID
    sequence: int
    event_key: str
    type: str
    step: str | None = None
    data: Any = None
    created_at: datetime


class RunEventRepository:
    """Persist minimal Runs and ordered Events through psycopg3 directly."""

    def __init__(
        self,
        pool: AsyncConnectionPool[Any],
        *,
        live_events: LiveEventWakeups | None = None,
    ) -> None:
        self._pool = pool
        self._live_events = live_events

    @asynccontextmanager
    async def transaction(self):
        """Yield a caller-owned PostgreSQL transaction for atomic finalization."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                yield connection

    async def publish_wakeup(self, tenant_id: str, run_id: UUID) -> None:
        """Publish a committed Run change to live subscribers."""
        await self._publish_wakeup(tenant_id, run_id)

    async def mark_event_conflict_in_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        run_id: UUID,
        event_key: str,
    ) -> None:
        """Persist the canonical failed status after a savepoint rollback."""
        await _mark_event_conflict_in_transaction(
            connection,
            tenant_id=tenant_id,
            run_id=run_id,
            event_key=event_key,
        )

    async def _lock_and_validate_claim(
        self,
        cursor: Any,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
        allow_failed: bool = False,
    ) -> None:
        """Lock a Run and reject missing, expired, or replaced claims."""
        await cursor.execute(
            """
            SELECT status, owner_instance_id, execution_epoch
            FROM langgraph_v2.runs
            WHERE tenant_id = %s AND run_id = %s
            FOR UPDATE
            """,
            (tenant_id, run_id),
        )
        claim = await cursor.fetchone()
        if claim is not None:
            await cursor.execute(
                """
                SELECT expires_at > clock_timestamp() AS claim_active
                FROM langgraph_v2.runs
                WHERE tenant_id = %s AND run_id = %s
                """,
                (tenant_id, run_id),
            )
            claim_active = await cursor.fetchone()
        else:
            claim_active = None
        if (
            claim is None
            or (
                claim["status"] != "running"
                and not (allow_failed and claim["status"] == "failed")
            )
            or claim["owner_instance_id"] != owner_instance_id
            or claim["execution_epoch"] != execution_epoch
            or claim_active is None
            or (claim["status"] == "running" and not claim_active["claim_active"])
        ):
            raise ClaimFenced(str(run_id))

    async def create_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        conversation_id: str,
        owner_instance_id: str,
    ) -> RunRecord:
        """Create a directly executing Run in the running state."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    INSERT INTO langgraph_v2.runs (
                        tenant_id, run_id, conversation_id, status,
                        owner_instance_id, execution_epoch, heartbeat_at,
                        expires_at
                    ) VALUES (%s, %s, %s, 'running', %s, %s, clock_timestamp(),
                    clock_timestamp() + (%s * interval '1 second'))
                    RETURNING tenant_id, run_id, conversation_id, status,
                              terminal_outcome, created_at, completed_at,
                              owner_instance_id, execution_epoch, heartbeat_at,
                              expires_at, checkpoint_id, checkpoint_ns
                    """,
                    (
                        tenant_id,
                        run_id,
                        conversation_id,
                        owner_instance_id,
                        1,
                        CLAIM_LEASE_SECONDS,
                    ),
                )
                row = await cursor.fetchone()
        return RunRecord.model_validate(row)

    async def get_run(self, tenant_id: str, run_id: UUID) -> RunRecord:
        """Return a Run inside one Tenant or conceal it as missing."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, run_id, conversation_id, status,
                           terminal_outcome, created_at, completed_at,
                           owner_instance_id, execution_epoch, heartbeat_at,
                           expires_at, checkpoint_id, checkpoint_ns
                    FROM langgraph_v2.runs
                    WHERE tenant_id = %s AND run_id = %s
                    """,
                    (tenant_id, run_id),
                )
                row = await cursor.fetchone()
        if row is None:
            raise RunNotFound(str(run_id))
        return RunRecord.model_validate(row)

    async def resume_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
    ) -> RunRecord:
        """Atomically claim a stale or interrupted Run for a new epoch."""
        interruption: EventRecord | None = None
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        SELECT tenant_id, run_id, conversation_id, status,
                               terminal_outcome, created_at, completed_at,
                               owner_instance_id, execution_epoch, heartbeat_at,
                               expires_at, checkpoint_id, checkpoint_ns,
                               next_event_sequence,
                               expires_at > clock_timestamp() AS claim_active
                        FROM langgraph_v2.runs
                        WHERE tenant_id = %s AND run_id = %s
                        FOR UPDATE
                        """,
                        (tenant_id, run_id),
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        raise RunNotFound(str(run_id))
                    resumable = (
                        row["status"] == "running" and not row["claim_active"]
                    ) or (
                        row["status"] == "interrupted"
                        and row["owner_instance_id"] == ""
                    )
                    if (
                        not resumable
                        or row["checkpoint_id"] is None
                        or row["checkpoint_ns"] is None
                    ):
                        raise ResumeConflict(str(run_id))
                    stale_running = (
                        row["status"] == "running" and not row["claim_active"]
                    )
                    if stale_running:
                        interruption = await _insert_interruption_event(
                            cursor,
                            tenant_id=tenant_id,
                            run_id=run_id,
                            sequence=row["next_event_sequence"],
                            interrupted_epoch=row["execution_epoch"],
                        )
                    await cursor.execute(
                        """
                        UPDATE langgraph_v2.runs
                        SET status = 'running', owner_instance_id = %s,
                            execution_epoch = execution_epoch + 1,
                            heartbeat_at = clock_timestamp(),
                            expires_at = clock_timestamp() +
                                (%s * interval '1 second'),
                            next_event_sequence = next_event_sequence + %s
                        WHERE tenant_id = %s AND run_id = %s
                        RETURNING tenant_id, run_id, conversation_id, status,
                                  terminal_outcome, created_at, completed_at,
                                  owner_instance_id, execution_epoch,
                                  heartbeat_at, expires_at, checkpoint_id,
                                  checkpoint_ns
                        """,
                        (
                            owner_instance_id,
                            CLAIM_LEASE_SECONDS,
                            int(stale_running),
                            tenant_id,
                            run_id,
                        ),
                    )
                    row = await cursor.fetchone()
        if row is None:
            raise ResumeConflict(str(run_id))
        if interruption is not None:
            await self._publish_wakeup(tenant_id, run_id)
        return RunRecord.model_validate(row)

    async def list_events(self, tenant_id: str, run_id: UUID) -> list[EventRecord]:
        """Return one Tenant's Events in their durable sequence order."""
        await self.get_run(tenant_id, run_id)
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, run_id, sequence, event_key, type, step,
                           data, created_at
                    FROM langgraph_v2.events
                    WHERE tenant_id = %s AND run_id = %s
                    ORDER BY sequence
                    """,
                    (tenant_id, run_id),
                )
                rows = await cursor.fetchall()
        return [EventRecord.model_validate(row) for row in rows]

    async def list_events_after(
        self,
        tenant_id: str,
        run_id: UUID,
        *,
        after_sequence: int,
    ) -> list[EventRecord]:
        """Return this Run's durable Event snapshot strictly after a sequence."""
        await self.get_run(tenant_id, run_id)
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, run_id, sequence, event_key, type, step,
                           data, created_at
                    FROM langgraph_v2.events
                    WHERE tenant_id = %s AND run_id = %s AND sequence > %s
                    ORDER BY sequence
                    """,
                    (tenant_id, run_id, after_sequence),
                )
                rows = await cursor.fetchall()
        return [EventRecord.model_validate(row) for row in rows]

    async def heartbeat(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
    ) -> RunRecord:
        """Refresh a matching, non-expired claim while its Run is running."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await self._lock_and_validate_claim(
                        cursor,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        owner_instance_id=owner_instance_id,
                        execution_epoch=execution_epoch,
                    )
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        UPDATE langgraph_v2.runs
                        SET heartbeat_at = clock_timestamp(),
                            expires_at = clock_timestamp() + (%s * interval '1 second')
                        WHERE tenant_id = %s AND run_id = %s
                        RETURNING tenant_id, run_id, conversation_id, status,
                                  terminal_outcome, created_at, completed_at,
                                  owner_instance_id, execution_epoch, heartbeat_at,
                                  expires_at, checkpoint_id, checkpoint_ns
                        """,
                        (CLAIM_LEASE_SECONDS, tenant_id, run_id),
                    )
                    row = await cursor.fetchone()
        if row is None:
            raise ClaimFenced(str(run_id))
        return RunRecord.model_validate(row)

    async def interrupt_runs_owned_by(
        self,
        owner_instance_id: str,
    ) -> list[RunRecord]:
        """Release every unfinished Run still owned by one shutting-down instance."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        UPDATE langgraph_v2.runs
                        SET status = 'interrupted', owner_instance_id = '',
                            heartbeat_at = clock_timestamp(),
                            expires_at = clock_timestamp()
                        WHERE owner_instance_id = %s AND status = 'running'
                        RETURNING tenant_id, run_id, conversation_id, status,
                                  terminal_outcome, created_at, completed_at,
                                  owner_instance_id, execution_epoch,
                                  heartbeat_at, expires_at, checkpoint_id,
                                  checkpoint_ns
                        """,
                        (owner_instance_id,),
                    )
                    rows = await cursor.fetchall()
        return [RunRecord.model_validate(row) for row in rows]

    async def interrupt_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
    ) -> RunRecord:
        """Release one authoritative claim when local execution cannot start."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await self._lock_and_validate_claim(
                        cursor,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        owner_instance_id=owner_instance_id,
                        execution_epoch=execution_epoch,
                    )
                    await cursor.execute(
                        """
                        UPDATE langgraph_v2.runs
                        SET status = 'interrupted', owner_instance_id = '',
                            heartbeat_at = clock_timestamp(),
                            expires_at = clock_timestamp()
                        WHERE tenant_id = %s AND run_id = %s
                        RETURNING tenant_id, run_id, conversation_id, status,
                                  terminal_outcome, created_at, completed_at,
                                  owner_instance_id, execution_epoch,
                                  heartbeat_at, expires_at, checkpoint_id,
                                  checkpoint_ns
                        """,
                        (tenant_id, run_id),
                    )
                    row = await cursor.fetchone()
        if row is None:
            raise ClaimFenced(str(run_id))
        await self._publish_wakeup(tenant_id, run_id)
        return RunRecord.model_validate(row)

    async def interrupt_expired_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        observed_execution_epoch: int,
    ) -> EventRecord | None:
        """Fence one expired owner and append its interruption Event atomically.

        A follower supplies the epoch it observed.  A conditional update makes
        exactly one stale-claim observer the winner; every loser only replays
        the winner's durable Event.
        """
        event: EventRecord | None = None
        async with self._pool.connection() as connection:
            async with connection.transaction():
                event = await interrupt_expired_run_in_transaction(
                    connection,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    observed_execution_epoch=observed_execution_epoch,
                )
        if event is not None:
            await self._publish_wakeup(tenant_id, run_id)
        return event

    async def get_event(
        self,
        tenant_id: str,
        run_id: UUID,
        event_key: str,
    ) -> EventRecord:
        """Return one producer-keyed Event without crossing Tenant boundaries."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, run_id, sequence, event_key, type, step,
                           data, created_at
                    FROM langgraph_v2.events
                    WHERE tenant_id = %s AND run_id = %s AND event_key = %s
                    """,
                    (tenant_id, run_id, event_key),
                )
                row = await cursor.fetchone()
        if row is None:
            raise EventNotFound(event_key)
        return EventRecord.model_validate(row)

    async def update_checkpoint_pointer(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
        checkpoint_id: str,
        checkpoint_ns: str,
    ) -> RunRecord:
        """Record a committed checkpoint only while its claim is authoritative."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await self._lock_and_validate_claim(
                        cursor,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        owner_instance_id=owner_instance_id,
                        execution_epoch=execution_epoch,
                        allow_failed=True,
                    )
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        UPDATE langgraph_v2.runs
                        SET checkpoint_id = %s, checkpoint_ns = %s
                        WHERE tenant_id = %s AND run_id = %s
                        RETURNING tenant_id, run_id, conversation_id, status,
                                  terminal_outcome, created_at, completed_at,
                                  owner_instance_id, execution_epoch, heartbeat_at,
                                  expires_at, checkpoint_id, checkpoint_ns
                        """,
                        (
                            checkpoint_id,
                            checkpoint_ns,
                            tenant_id,
                            run_id,
                        ),
                    )
                    row = await cursor.fetchone()
        if row is None:
            raise ClaimFenced(str(run_id))
        return RunRecord.model_validate(row)

    async def append_event(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        event: EventInput,
        owner_instance_id: str,
        execution_epoch: int,
    ) -> EventRecord:
        """Append idempotently or fail the Run on a stable-key conflict."""
        return await self._persist_event(
            tenant_id=tenant_id,
            run_id=run_id,
            event=event,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
            completes_run=False,
        )

    async def complete_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        event: EventInput,
        owner_instance_id: str,
        execution_epoch: int,
    ) -> EventRecord:
        """Atomically append the terminal Event and complete its Run."""
        return await self._persist_event(
            tenant_id=tenant_id,
            run_id=run_id,
            event=event,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
            completes_run=True,
        )

    async def complete_run_in_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        run_id: UUID,
        event: EventInput,
        owner_instance_id: str,
        execution_epoch: int,
    ) -> EventRecord:
        """Append a terminal Event inside a caller-owned transaction."""
        return await self._persist_event(
            tenant_id=tenant_id,
            run_id=run_id,
            event=event,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
            completes_run=True,
            connection=connection,
            publish_wakeup=False,
        )

    async def fail_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        event: EventInput,
        owner_instance_id: str,
        execution_epoch: int,
    ) -> EventRecord:
        """Atomically append a terminal error Event and fail its Run."""
        return await self._persist_event(
            tenant_id=tenant_id,
            run_id=run_id,
            event=event,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
            completes_run=True,
            terminal_status="failed",
        )

    async def _persist_event(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        event: EventInput,
        owner_instance_id: str,
        execution_epoch: int,
        completes_run: bool,
        terminal_status: TerminalStatus = "completed",
        connection: Any | None = None,
        publish_wakeup: bool = True,
    ) -> EventRecord:
        canonical_envelope = _canonical_envelope(event)
        conflict = False
        row = None
        async with self._transaction(connection) as connection:
            async with connection.cursor(row_factory=dict_row) as run_cursor:
                await run_cursor.execute(
                    """
                        SELECT next_event_sequence, status, owner_instance_id,
                               execution_epoch
                        FROM langgraph_v2.runs
                        WHERE tenant_id = %s AND run_id = %s
                        FOR UPDATE
                        """,
                    (tenant_id, run_id),
                )
                run_row = await run_cursor.fetchone()
            if run_row is None:
                raise RunNotFound(str(run_id))
            async with connection.cursor(row_factory=dict_row) as claim_cursor:
                await claim_cursor.execute(
                    """
                        SELECT expires_at <= clock_timestamp() AS claim_expired
                        FROM langgraph_v2.runs
                        WHERE tenant_id = %s AND run_id = %s
                        """,
                    (tenant_id, run_id),
                )
                claim_row = await claim_cursor.fetchone()
            if (
                run_row["owner_instance_id"] != owner_instance_id
                or run_row["execution_epoch"] != execution_epoch
                or claim_row is None
                or claim_row["claim_expired"]
            ):
                raise ClaimFenced(str(run_id))
            if completes_run and terminal_status == "completed":
                async with connection.cursor() as cancellation_cursor:
                    await cancellation_cursor.execute(
                        """
                            SELECT 1
                            FROM langgraph_v2.cancellation_intents
                            WHERE tenant_id = %s AND run_id = %s
                            """,
                        (tenant_id, run_id),
                    )
                    if await cancellation_cursor.fetchone() is not None:
                        raise CancellationObserved(str(run_id))
            async with connection.cursor(row_factory=dict_row) as existing_cursor:
                await existing_cursor.execute(
                    """
                        SELECT tenant_id, run_id, sequence, event_key, type, step,
                               data, canonical_envelope, created_at
                        FROM langgraph_v2.events
                        WHERE tenant_id = %s AND run_id = %s AND event_key = %s
                        """,
                    (tenant_id, run_id, event.event_key),
                )
                row = await existing_cursor.fetchone()

            if row is not None:
                if row["canonical_envelope"] != canonical_envelope:
                    await _mark_event_conflict_in_transaction(
                        connection,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        event_key=event.event_key,
                    )
                    conflict = True
                elif run_row["status"] == "completed" and not completes_run:
                    raise ClaimFenced(str(run_id))
            else:
                if run_row["status"] not in {"running", "completed"} or (
                    run_row["status"] == "completed" and not completes_run
                ):
                    raise ClaimFenced(str(run_id))
                sequence = run_row["next_event_sequence"]
                async with connection.cursor(row_factory=dict_row) as event_cursor:
                    await event_cursor.execute(
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
                            sequence,
                            event.event_key,
                            event.type,
                            event.step,
                            Jsonb(event.data),
                            canonical_envelope,
                        ),
                    )
                    row = await event_cursor.fetchone()
                await connection.execute(
                    """
                        UPDATE langgraph_v2.runs
                        SET next_event_sequence = next_event_sequence + 1
                        WHERE tenant_id = %s AND run_id = %s
                        """,
                    (tenant_id, run_id),
                )
            if completes_run and not conflict:
                await _set_terminal_status_in_transaction(
                    connection,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    status=terminal_status,
                    outcome=event.data,
                )
        if conflict:
            raise EventInvariantConflict(event.event_key)
        persisted = EventRecord.model_validate(row)
        if publish_wakeup:
            await self._publish_wakeup(tenant_id, run_id)
        return persisted

    @asynccontextmanager
    async def _transaction(self, connection: Any | None):
        """Use a caller transaction or open one for a standalone operation."""
        if connection is not None:
            yield connection
            return
        async with self._pool.connection() as owned_connection:
            async with owned_connection.transaction():
                yield owned_connection

    async def _publish_wakeup(self, tenant_id: str, run_id: UUID) -> None:
        if self._live_events is not None:
            await self._live_events.publish(tenant_id, run_id)


async def interrupt_expired_run_in_transaction(
    connection: Any,
    *,
    tenant_id: str,
    run_id: UUID,
    observed_execution_epoch: int,
) -> EventRecord | None:
    """CAS one expired `running` claim inside a caller-owned transaction seam."""
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(
            """
            SELECT next_event_sequence
            FROM langgraph_v2.runs
            WHERE tenant_id = %s AND run_id = %s
              AND status = 'running'
              AND execution_epoch = %s
              AND expires_at <= clock_timestamp()
            FOR UPDATE
            """,
            (tenant_id, run_id, observed_execution_epoch),
        )
        run = await cursor.fetchone()
        if run is None:
            return None
        event = await _insert_interruption_event(
            cursor,
            tenant_id=tenant_id,
            run_id=run_id,
            sequence=run["next_event_sequence"],
            interrupted_epoch=observed_execution_epoch,
        )
        await cursor.execute(
            """
            UPDATE langgraph_v2.runs
            SET status = 'interrupted', owner_instance_id = '',
                execution_epoch = execution_epoch + 1,
                heartbeat_at = clock_timestamp(), expires_at = clock_timestamp(),
                next_event_sequence = next_event_sequence + 1
            WHERE tenant_id = %s AND run_id = %s
              AND status = 'running'
              AND execution_epoch = %s
              AND expires_at <= clock_timestamp()
            """,
            (tenant_id, run_id, observed_execution_epoch),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("expired Run CAS lost after row lock")
    return event


async def _insert_interruption_event(
    cursor: Any,
    *,
    tenant_id: str,
    run_id: UUID,
    sequence: int,
    interrupted_epoch: int,
) -> EventRecord:
    """Append the stable boundary that fences followers of an expired epoch."""
    event = EventInput(
        event_key=f"lifecycle:interrupted:{interrupted_epoch}",
        type="error",
        step="lifecycle",
        data={"status": "interrupted", "reason": "claim_expired"},
    )
    await cursor.execute(
        """
        INSERT INTO langgraph_v2.events (
            tenant_id, run_id, sequence, event_key, type, step, data,
            canonical_envelope
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING tenant_id, run_id, sequence, event_key, type, step,
                  data, created_at
        """,
        (
            tenant_id,
            run_id,
            sequence,
            event.event_key,
            event.type,
            event.step,
            Jsonb(event.data),
            _canonical_envelope(event),
        ),
    )
    return EventRecord.model_validate(await cursor.fetchone())


async def _set_terminal_status_in_transaction(
    connection: Any,
    *,
    tenant_id: str,
    run_id: UUID,
    status: TerminalStatus,
    outcome: Any,
) -> None:
    """Apply one terminal Run transition inside the caller's transaction."""
    completed_at = "COALESCE(completed_at, now())" if status == "completed" else "NULL"
    await connection.execute(
        f"""
        UPDATE langgraph_v2.runs
        SET status = %s, terminal_outcome = %s,
            completed_at = {completed_at}
        WHERE tenant_id = %s AND run_id = %s
        """,
        (status, Jsonb(outcome), tenant_id, run_id),
    )


async def _mark_event_conflict_in_transaction(
    connection: Any,
    *,
    tenant_id: str,
    run_id: UUID,
    event_key: str,
) -> None:
    """Mark a Run failed with the stable event-conflict outcome."""
    await connection.execute(
        """
        UPDATE langgraph_v2.runs
        SET status = 'failed', terminal_outcome = %s, completed_at = NULL
        WHERE tenant_id = %s AND run_id = %s
        """,
        (
            Jsonb({"error": "event_invariant_conflict", "event_key": event_key}),
            tenant_id,
            run_id,
        ),
    )


def _canonical_envelope(event: EventInput) -> str:
    return json.dumps(
        {"data": event.data, "step": event.step, "type": event.type},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
