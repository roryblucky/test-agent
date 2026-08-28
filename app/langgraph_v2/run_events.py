"""Tenant-scoped persistence for minimal v2 Runs and Events."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field


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


CLAIM_LEASE_SECONDS = 30
CLAIM_HEARTBEAT_INTERVAL_SECONDS = CLAIM_LEASE_SECONDS // 3


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

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

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
                    ) VALUES (%s, %s, %s, 'running', %s, %s, now(),
                              now() + (%s * interval '1 second'))
                    RETURNING tenant_id, run_id, conversation_id, status,
                              terminal_outcome, created_at, completed_at,
                              owner_instance_id, execution_epoch, heartbeat_at,
                              expires_at
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
                           expires_at
                    FROM langgraph_v2.runs
                    WHERE tenant_id = %s AND run_id = %s
                    """,
                    (tenant_id, run_id),
                )
                row = await cursor.fetchone()
        if row is None:
            raise RunNotFound(str(run_id))
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
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    UPDATE langgraph_v2.runs
                    SET heartbeat_at = now(),
                        expires_at = now() + (%s * interval '1 second')
                    WHERE tenant_id = %s AND run_id = %s
                      AND status = 'running'
                      AND owner_instance_id = %s
                      AND execution_epoch = %s
                      AND expires_at > now()
                    RETURNING tenant_id, run_id, conversation_id, status,
                              terminal_outcome, created_at, completed_at,
                              owner_instance_id, execution_epoch, heartbeat_at,
                              expires_at
                    """,
                    (
                        CLAIM_LEASE_SECONDS,
                        tenant_id,
                        run_id,
                        owner_instance_id,
                        execution_epoch,
                    ),
                )
                row = await cursor.fetchone()
        if row is None:
            raise ClaimFenced(str(run_id))
        return RunRecord.model_validate(row)

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

    async def _persist_event(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        event: EventInput,
        owner_instance_id: str,
        execution_epoch: int,
        completes_run: bool,
    ) -> EventRecord:
        canonical_envelope = _canonical_envelope(event)
        conflict = False
        row = None
        async with self._pool.connection() as connection:
            async with connection.transaction():
                run_cursor = await connection.execute(
                    """
                    SELECT next_event_sequence, status, owner_instance_id,
                           execution_epoch, expires_at <= now()
                    FROM langgraph_v2.runs
                    WHERE tenant_id = %s AND run_id = %s
                    FOR UPDATE
                    """,
                    (tenant_id, run_id),
                )
                run_row = await run_cursor.fetchone()
                if run_row is None:
                    raise RunNotFound(str(run_id))
                if (
                    run_row[1] not in {"running", "completed"}
                    or run_row[2] != owner_instance_id
                    or run_row[3] != execution_epoch
                    or run_row[4]
                    or (run_row[1] == "completed" and not completes_run)
                ):
                    raise ClaimFenced(str(run_id))
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
                        await connection.execute(
                            """
                            UPDATE langgraph_v2.runs
                            SET status = 'failed', terminal_outcome = %s,
                                completed_at = NULL
                            WHERE tenant_id = %s AND run_id = %s
                            """,
                            (
                                Jsonb(
                                    {
                                        "error": "event_invariant_conflict",
                                        "event_key": event.event_key,
                                    }
                                ),
                                tenant_id,
                                run_id,
                            ),
                        )
                        conflict = True
                else:
                    sequence = run_row[0]
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
                    await connection.execute(
                        """
                        UPDATE langgraph_v2.runs
                        SET status = 'completed', terminal_outcome = %s,
                            completed_at = COALESCE(completed_at, now())
                        WHERE tenant_id = %s AND run_id = %s
                        """,
                        (Jsonb(event.data), tenant_id, run_id),
                    )
        if conflict:
            raise EventInvariantConflict(event.event_key)
        return EventRecord.model_validate(row)


def _canonical_envelope(event: EventInput) -> str:
    return json.dumps(
        {"data": event.data, "step": event.step, "type": event.type},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
