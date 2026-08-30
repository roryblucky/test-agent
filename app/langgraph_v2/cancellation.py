"""Tenant-scoped durable cancellation requests for v2 Runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.langgraph_v2.runs import ClaimFenced, RunNotFound

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

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    async def request(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
    ) -> CancellationRequestResult:
        """Record one idempotent cancellation intent."""
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
    ) -> bool:
        """Atomically stop one matching owner after its intent is durable."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        SELECT status, owner_instance_id, execution_epoch,
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
                    if (
                        run["status"] == "cancelled"
                        and run["execution_epoch"] == execution_epoch
                    ):
                        return True
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
                        return False
                    await cursor.execute(
                        """
                        UPDATE langgraph_v2.runs
                        SET status = 'cancelled',
                            terminal_outcome = %s,
                            completed_at = NULL,
                            owner_instance_id = '',
                            heartbeat_at = clock_timestamp(),
                            expires_at = clock_timestamp()
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
        return True
