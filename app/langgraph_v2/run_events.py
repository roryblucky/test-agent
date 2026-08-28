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


class RunNotFound(LookupError):
    """A Run is absent from the requested Tenant boundary."""


class EventInvariantConflict(RuntimeError):
    """A stable Event key was reused with a different canonical envelope."""


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
    ) -> RunRecord:
        """Create a directly executing Run in the running state."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    INSERT INTO langgraph_v2.runs (
                        tenant_id, run_id, conversation_id, status
                    ) VALUES (%s, %s, %s, 'running')
                    RETURNING tenant_id, run_id, conversation_id, status,
                              terminal_outcome, created_at, completed_at
                    """,
                    (tenant_id, run_id, conversation_id),
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
                           terminal_outcome, created_at, completed_at
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

    async def append_event(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        event: EventInput,
    ) -> EventRecord:
        """Append idempotently or fail the Run on a stable-key conflict."""
        return await self._persist_event(
            tenant_id=tenant_id,
            run_id=run_id,
            event=event,
            completes_run=False,
        )

    async def complete_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        event: EventInput,
    ) -> EventRecord:
        """Atomically append the terminal Event and complete its Run."""
        return await self._persist_event(
            tenant_id=tenant_id,
            run_id=run_id,
            event=event,
            completes_run=True,
        )

    async def _persist_event(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        event: EventInput,
        completes_run: bool,
    ) -> EventRecord:
        canonical_envelope = _canonical_envelope(event)
        conflict = False
        row = None
        async with self._pool.connection() as connection:
            async with connection.transaction():
                run_cursor = await connection.execute(
                    """
                    SELECT next_event_sequence
                    FROM langgraph_v2.runs
                    WHERE tenant_id = %s AND run_id = %s
                    FOR UPDATE
                    """,
                    (tenant_id, run_id),
                )
                run_row = await run_cursor.fetchone()
                if run_row is None:
                    raise RunNotFound(str(run_id))
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
