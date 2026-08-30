"""Tenant-scoped persistence for transitional v2 Run lifecycle records."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel


class RepositoryNotFound(LookupError):
    """A tenant-scoped lookup is indistinguishable from missing."""


class RunNotFound(RepositoryNotFound):
    """A Run is absent from the requested Tenant boundary."""


class ClaimFenced(RuntimeError):
    """The supplied owner and execution epoch cannot write this Run."""


class CancellationObserved(RuntimeError):
    """A durable cancellation intent blocks the next publication boundary."""


class ResumeConflict(RuntimeError):
    """A Run cannot be resumed in its current state."""


CLAIM_LEASE_SECONDS = 30
CLAIM_HEARTBEAT_INTERVAL_SECONDS = CLAIM_LEASE_SECONDS // 3
TerminalStatus = Literal["completed", "failed"]


class RunRecord(BaseModel):
    """Persisted transitional Run state."""

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


class RunRepository:
    """Persist only the Run lifecycle still required before task 48."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    @asynccontextmanager
    async def transaction(self):
        """Yield a caller-owned PostgreSQL transaction for finalization."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                yield connection

    async def _lock_claim(
        self,
        cursor: Any,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
        allow_failed: bool = False,
    ) -> None:
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
            lease = await cursor.fetchone()
        else:
            lease = None
        if (
            claim is None
            or (
                claim["status"] != "running"
                and not (allow_failed and claim["status"] == "failed")
            )
            or claim["owner_instance_id"] != owner_instance_id
            or claim["execution_epoch"] != execution_epoch
            or lease is None
            or (claim["status"] == "running" and not lease["claim_active"])
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
                        owner_instance_id, execution_epoch, heartbeat_at, expires_at
                    ) VALUES (%s, %s, %s, 'running', %s, 1, clock_timestamp(),
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
        """Claim a stale or interrupted transitional Run for a new epoch."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        SELECT status, owner_instance_id, checkpoint_id, checkpoint_ns,
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
                    await cursor.execute(
                        """
                        UPDATE langgraph_v2.runs
                        SET status = 'running', owner_instance_id = %s,
                            execution_epoch = execution_epoch + 1,
                            heartbeat_at = clock_timestamp(),
                            expires_at = clock_timestamp() +
                                (%s * interval '1 second')
                        WHERE tenant_id = %s AND run_id = %s
                        RETURNING tenant_id, run_id, conversation_id, status,
                                  terminal_outcome, created_at, completed_at,
                                  owner_instance_id, execution_epoch, heartbeat_at,
                                  expires_at, checkpoint_id, checkpoint_ns
                        """,
                        (
                            owner_instance_id,
                            CLAIM_LEASE_SECONDS,
                            tenant_id,
                            run_id,
                        ),
                    )
                    resumed = await cursor.fetchone()
        return RunRecord.model_validate(resumed)

    async def interrupt_expired_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        observed_execution_epoch: int,
    ) -> RunRecord | None:
        """Release one expired claim if its observed epoch is still current."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    UPDATE langgraph_v2.runs
                    SET status = 'interrupted', owner_instance_id = '',
                        execution_epoch = execution_epoch + 1,
                        heartbeat_at = clock_timestamp(),
                        expires_at = clock_timestamp()
                    WHERE tenant_id = %s AND run_id = %s
                      AND status = 'running'
                      AND execution_epoch = %s
                      AND expires_at <= clock_timestamp()
                    RETURNING tenant_id, run_id, conversation_id, status,
                              terminal_outcome, created_at, completed_at,
                              owner_instance_id, execution_epoch, heartbeat_at,
                              expires_at, checkpoint_id, checkpoint_ns
                    """,
                    (tenant_id, run_id, observed_execution_epoch),
                )
                row = await cursor.fetchone()
        return RunRecord.model_validate(row) if row is not None else None

    async def heartbeat(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
    ) -> RunRecord:
        """Refresh a matching, non-expired claim."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await self._lock_claim(
                        cursor,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        owner_instance_id=owner_instance_id,
                        execution_epoch=execution_epoch,
                    )
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

    async def interrupt_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
    ) -> RunRecord:
        """Release one claim after its request-owned execution stops."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await self._lock_claim(
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
                                  owner_instance_id, execution_epoch, heartbeat_at,
                                  expires_at, checkpoint_id, checkpoint_ns
                        """,
                        (tenant_id, run_id),
                    )
                    row = await cursor.fetchone()
        if row is None:
            raise ClaimFenced(str(run_id))
        return RunRecord.model_validate(row)

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
        """Record a committed checkpoint while its claim is authoritative."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await self._lock_claim(
                        cursor,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        owner_instance_id=owner_instance_id,
                        execution_epoch=execution_epoch,
                        allow_failed=True,
                    )
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
                        (checkpoint_id, checkpoint_ns, tenant_id, run_id),
                    )
                    row = await cursor.fetchone()
        if row is None:
            raise ClaimFenced(str(run_id))
        return RunRecord.model_validate(row)

    async def complete_run_in_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
        outcome: object,
    ) -> None:
        """Complete a claim inside a caller-owned terminal transaction."""
        async with connection.cursor(row_factory=dict_row) as cursor:
            await self._lock_claim(
                cursor,
                tenant_id=tenant_id,
                run_id=run_id,
                owner_instance_id=owner_instance_id,
                execution_epoch=execution_epoch,
            )
            await cursor.execute(
                """
                SELECT 1 FROM langgraph_v2.cancellation_intents
                WHERE tenant_id = %s AND run_id = %s
                """,
                (tenant_id, run_id),
            )
            if await cursor.fetchone() is not None:
                raise CancellationObserved(str(run_id))
        await _set_terminal_status_in_transaction(
            connection,
            tenant_id=tenant_id,
            run_id=run_id,
            status="completed",
            outcome=outcome,
        )

    async def fail_run(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
        error: object,
    ) -> RunRecord:
        """Fail a matching transitional Run."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await self._lock_claim(
                        cursor,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        owner_instance_id=owner_instance_id,
                        execution_epoch=execution_epoch,
                    )
                await _set_terminal_status_in_transaction(
                    connection,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    status="failed",
                    outcome={"error": error},
                )
        return await self.get_run(tenant_id, run_id)


async def _set_terminal_status_in_transaction(
    connection: Any,
    *,
    tenant_id: str,
    run_id: UUID,
    status: TerminalStatus,
    outcome: Any,
) -> None:
    completed_at = "COALESCE(completed_at, now())" if status == "completed" else "NULL"
    await connection.execute(
        f"""
        UPDATE langgraph_v2.runs
        SET status = %s, terminal_outcome = %s, completed_at = {completed_at}
        WHERE tenant_id = %s AND run_id = %s
        """,
        (status, Jsonb(outcome), tenant_id, run_id),
    )
