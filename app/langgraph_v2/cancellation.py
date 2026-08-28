"""Tenant-scoped durable cancellation requests for v2 Runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.run_events import RunNotFound

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
