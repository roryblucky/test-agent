"""Epoch-fenced PhaseResult journal for replay-safe LangGraph nodes."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field, model_validator

from app.langgraph_v2.artifacts import ArtifactRef, ArtifactStore
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.history import DEFAULT_HISTORY_TOKEN_BUDGET
from app.langgraph_v2.run_events import (
    ClaimFenced,
    EventInput,
    EventInvariantConflict,
    EventRecord,
    RunNotFound,
    _canonical_envelope,
    _set_terminal_status_in_transaction,
)

PhaseName = Literal[
    "query",
    "pre_moderation",
    "question_refinement",
    "retrieval",
    "reranking",
    "answer",
    "groundedness",
    "post_moderation",
    "finalization",
]

ALLOWED_PHASE_NAMES: frozenset[PhaseName] = frozenset(
    {
        "query",
        "pre_moderation",
        "question_refinement",
        "retrieval",
        "reranking",
        "answer",
        "groundedness",
        "post_moderation",
        "finalization",
    }
)


class PhaseResultConflict(RuntimeError):
    """A phase key was reused with different normalized content."""


class PhaseResultInput(BaseModel):
    """Normalized phase output plus stable Events committed as one unit."""

    phase_name: PhaseName
    normalized_result: Any = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    events: tuple[EventInput, ...] = ()
    terminal_status: Literal["failed"] | None = None

    @model_validator(mode="after")
    def require_normalized_content(self) -> PhaseResultInput:
        """Require structured output or references to durable Artifacts."""
        if self.normalized_result is None and not self.artifact_refs:
            raise ValueError("a PhaseResult needs normalized_result or artifact_refs")
        if self.normalized_result is not None and not isinstance(
            self.normalized_result, (dict, list)
        ):
            raise ValueError("normalized_result must be a structured object or list")
        _assert_stable_content(self.normalized_result)
        _assert_stable_content(self.artifact_refs)
        for event in self.events:
            _assert_stable_content(event.data)
        return self


class PhaseResultRecord(BaseModel):
    """Durable normalized result keyed by Tenant, Run, and phase."""

    tenant_id: str
    run_id: UUID
    phase_name: PhaseName
    execution_epoch: int
    normalized_result: Any = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    event_keys: tuple[str, ...] = ()
    events: tuple[EventRecord, ...] = ()
    canonical_result: str
    created_at: datetime


@dataclass(frozen=True)
class PhaseExecutionContext:
    """Claim-scoped dependencies made available to a LangGraph node."""

    repository: PhaseResultRepository
    tenant_id: str
    run_id: UUID
    owner_instance_id: str
    execution_epoch: int
    artifact_repository: ArtifactStore | None = None
    message_repository: ConversationMessageRepository | None = None
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET
    cancellation_check: Callable[[], Awaitable[bool]] | None = None


PhaseInvoker = Callable[[], Awaitable[PhaseResultInput]]
PhasePreCommitCheck = Callable[[], Awaitable[None]]


class PhaseResultRepository:
    """Persist and reuse one normalized result per Run phase."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    async def get_completed(
        self,
        tenant_id: str,
        run_id: UUID,
        phase_name: PhaseName,
    ) -> PhaseResultRecord | None:
        """Read a completed PhaseResult without crossing Tenant boundaries."""
        _validate_phase_name(phase_name)
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, run_id, phase_name, execution_epoch,
                           normalized_result, artifact_refs, event_keys,
                           canonical_result, created_at
                    FROM langgraph_v2.phase_results
                    WHERE tenant_id = %s AND run_id = %s AND phase_name = %s
                    """,
                    (tenant_id, run_id, phase_name),
                )
                row = await cursor.fetchone()
        if row is None:
            return None
        record = PhaseResultRecord.model_validate(row)
        return record.model_copy(
            update={"events": tuple(await self._read_events(tenant_id, run_id, record))}
        )

    async def _read_events(
        self,
        tenant_id: str,
        run_id: UUID,
        phase: PhaseResultRecord,
    ) -> list[EventRecord]:
        """Load the stable Events belonging to one journal entry."""
        if not phase.event_keys:
            return []
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, run_id, sequence, event_key, type, step,
                           data, created_at
                    FROM langgraph_v2.events
                    WHERE tenant_id = %s AND run_id = %s
                      AND event_key = ANY(%s)
                    ORDER BY sequence
                    """,
                    (tenant_id, run_id, list(phase.event_keys)),
                )
                rows = await cursor.fetchall()
        return [EventRecord.model_validate(row) for row in rows]

    async def get_or_invoke(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
        phase_name: PhaseName,
        invoke: PhaseInvoker,
        before_commit: PhasePreCommitCheck | None = None,
    ) -> PhaseResultRecord:
        """Reuse a completed result, otherwise invoke and atomically commit it."""
        _validate_phase_name(phase_name)
        existing = await self.get_completed(tenant_id, run_id, phase_name)
        if existing is not None:
            return existing
        candidate = await invoke()
        if candidate.phase_name != phase_name:
            raise ValueError("invoked PhaseResult name does not match requested phase")
        if before_commit is not None:
            await before_commit()
        return await self.commit(
            tenant_id=tenant_id,
            run_id=run_id,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
            phase=candidate,
        )

    async def commit(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
        phase: PhaseResultInput,
    ) -> PhaseResultRecord:
        """Commit a normalized result and all stable Events under one claim."""
        _validate_phase_name(phase.phase_name)
        canonical_result = _canonical_result(phase)
        result_row: dict[str, Any] | None = None
        conflict: RuntimeError | None = None

        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        SELECT status, owner_instance_id, execution_epoch,
                               expires_at
                        FROM langgraph_v2.runs
                        WHERE tenant_id = %s AND run_id = %s
                        FOR UPDATE
                        """,
                        (tenant_id, run_id),
                    )
                    run_row = await cursor.fetchone()
                    if run_row is None:
                        raise RunNotFound(str(run_id))
                    if (
                        run_row["status"] != "running"
                        or run_row["owner_instance_id"] != owner_instance_id
                        or run_row["execution_epoch"] != execution_epoch
                        or run_row["expires_at"] <= await _database_now(cursor)
                    ):
                        raise ClaimFenced(str(run_id))

                    await cursor.execute(
                        """
                        SELECT tenant_id, run_id, phase_name, execution_epoch,
                               normalized_result, artifact_refs, event_keys,
                               canonical_result, created_at
                        FROM langgraph_v2.phase_results
                        WHERE tenant_id = %s AND run_id = %s AND phase_name = %s
                        FOR UPDATE
                        """,
                        (tenant_id, run_id, phase.phase_name),
                    )
                    result_row = await cursor.fetchone()
                    if result_row is not None:
                        if result_row["canonical_result"] != canonical_result:
                            await _fail_for_conflict(
                                connection, tenant_id, run_id, phase.phase_name
                            )
                            conflict = PhaseResultConflict(phase.phase_name)
                        elif set(result_row["event_keys"]) != {
                            event.event_key for event in phase.events
                        }:
                            await _fail_for_conflict(
                                connection, tenant_id, run_id, phase.phase_name
                            )
                            conflict = PhaseResultConflict(phase.phase_name)
                    for event in phase.events:
                        if conflict is not None:
                            break
                        prior_event = next(
                            (
                                candidate
                                for candidate in phase.events
                                if candidate.event_key == event.event_key
                                and candidate is not event
                            ),
                            None,
                        )
                        if prior_event is not None and (
                            _canonical_envelope(prior_event)
                            != _canonical_envelope(event)
                        ):
                            await _fail_for_conflict(
                                connection, tenant_id, run_id, event.event_key
                            )
                            conflict = EventInvariantConflict(event.event_key)
                            break
                        await cursor.execute(
                            """
                            SELECT canonical_envelope
                            FROM langgraph_v2.events
                            WHERE tenant_id = %s AND run_id = %s AND event_key = %s
                            FOR UPDATE
                            """,
                            (tenant_id, run_id, event.event_key),
                        )
                        existing_event = await cursor.fetchone()
                        if existing_event is not None and existing_event[
                            "canonical_envelope"
                        ] != _canonical_envelope(event):
                            await _fail_for_conflict(
                                connection, tenant_id, run_id, event.event_key
                            )
                            conflict = EventInvariantConflict(event.event_key)
                    if conflict is None and result_row is None:
                        await cursor.execute(
                            """
                            INSERT INTO langgraph_v2.phase_results (
                                tenant_id, run_id, phase_name, execution_epoch,
                                normalized_result, artifact_refs, event_keys,
                                canonical_result
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING tenant_id, run_id, phase_name,
                                      execution_epoch, normalized_result,
                                      artifact_refs, event_keys, canonical_result,
                                      created_at
                            """,
                            (
                                tenant_id,
                                run_id,
                                phase.phase_name,
                                execution_epoch,
                                Jsonb(phase.normalized_result),
                                Jsonb(phase.artifact_refs),
                                Jsonb([event.event_key for event in phase.events]),
                                canonical_result,
                            ),
                        )
                        result_row = await cursor.fetchone()

                    if conflict is None:
                        for event in phase.events:
                            await _persist_event_in_transaction(
                                cursor,
                                tenant_id=tenant_id,
                                run_id=run_id,
                                event=event,
                            )
                    if conflict is None and phase.terminal_status is not None:
                        terminal_event = next(
                            (event for event in phase.events if event.type == "error"),
                            None,
                        )
                        await _set_terminal_status_in_transaction(
                            connection,
                            tenant_id=tenant_id,
                            run_id=run_id,
                            status="failed",
                            outcome=(
                                terminal_event.data
                                if terminal_event is not None
                                else phase.normalized_result
                            ),
                        )

        if conflict is not None:
            raise conflict
        if result_row is None:
            raise RuntimeError("PhaseResult commit returned no row")
        record = PhaseResultRecord.model_validate(result_row)
        if not record.event_keys:
            return record
        return record.model_copy(
            update={"events": tuple(await self._read_events(tenant_id, run_id, record))}
        )


async def _database_now(cursor: Any) -> datetime:
    await cursor.execute("SELECT clock_timestamp() AS now")
    row = await cursor.fetchone()
    return row["now"]


async def _persist_event_in_transaction(
    cursor: Any,
    *,
    tenant_id: str,
    run_id: UUID,
    event: EventInput,
) -> EventRecord:
    canonical_envelope = _canonical_envelope(event)
    await cursor.execute(
        """
        SELECT tenant_id, run_id, sequence, event_key, type, step, data,
               canonical_envelope, created_at
        FROM langgraph_v2.events
        WHERE tenant_id = %s AND run_id = %s AND event_key = %s
        """,
        (tenant_id, run_id, event.event_key),
    )
    existing = await cursor.fetchone()
    if existing is not None:
        if existing["canonical_envelope"] != canonical_envelope:
            raise EventInvariantConflict(event.event_key)
        return EventRecord.model_validate(existing)

    await cursor.execute(
        """
        SELECT next_event_sequence
        FROM langgraph_v2.runs
        WHERE tenant_id = %s AND run_id = %s
        FOR UPDATE
        """,
        (tenant_id, run_id),
    )
    run_row = await cursor.fetchone()
    sequence = run_row["next_event_sequence"]
    await cursor.execute(
        """
        INSERT INTO langgraph_v2.events (
            tenant_id, run_id, sequence, event_key, type, step, data,
            canonical_envelope
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING tenant_id, run_id, sequence, event_key, type, step, data,
                  canonical_envelope, created_at
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
    inserted = await cursor.fetchone()
    await cursor.execute(
        """
        UPDATE langgraph_v2.runs
        SET next_event_sequence = next_event_sequence + 1
        WHERE tenant_id = %s AND run_id = %s
        """,
        (tenant_id, run_id),
    )
    return EventRecord.model_validate(inserted)


async def _fail_for_conflict(
    connection: Any,
    tenant_id: str,
    run_id: UUID,
    key: str,
) -> None:
    await connection.execute(
        """
        UPDATE langgraph_v2.runs
        SET status = 'failed', terminal_outcome = %s, completed_at = NULL
        WHERE tenant_id = %s AND run_id = %s
        """,
        (
            Jsonb({"error": "phase_result_invariant_conflict", "key": key}),
            tenant_id,
            run_id,
        ),
    )


def _canonical_result(phase: PhaseResultInput) -> str:
    return json.dumps(
        {
            "artifact_refs": phase.artifact_refs,
            "normalized_result": phase.normalized_result,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_phase_name(phase_name: str) -> None:
    if phase_name not in ALLOWED_PHASE_NAMES:
        raise ValueError(f"unsupported PhaseResult name: {phase_name}")


_VOLATILE_KEYS = frozenset(
    {
        "timestamp",
        "timestamp_ms",
        "duration",
        "duration_ms",
        "owner",
        "owner_instance_id",
        "attempt",
        "attempt_number",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
    }
)


def _assert_stable_content(value: Any) -> None:
    """Reject volatile execution metadata from durable PhaseResult content."""
    if isinstance(value, dict):
        volatile = _VOLATILE_KEYS.intersection(value)
        if volatile:
            raise ValueError(
                "volatile keys are not allowed in PhaseResult content: "
                + ", ".join(sorted(volatile))
            )
        for nested in value.values():
            _assert_stable_content(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_stable_content(nested)
