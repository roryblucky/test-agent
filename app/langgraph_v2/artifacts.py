"""Tenant-scoped production-shaped Artifact seam for v2 retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypedDict
from uuid import UUID, uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.langgraph_v2.authorization import TrustedRequestContext


class ArtifactNotFound(LookupError):
    """Artifact is missing from the requested Tenant boundary."""


class ArtifactInvariantConflict(RuntimeError):
    """An Artifact ID was reused for different immutable data."""


@dataclass(frozen=True)
class ArtifactScope:
    """Authorized Conversation Turn that owns an Artifact."""

    context: TrustedRequestContext
    conversation_id: str
    turn_id: UUID


class ArtifactRecord(BaseModel):
    """Persisted immutable Artifact metadata and payload."""

    tenant_id: str
    conversation_id: str
    turn_id: UUID
    artifact_id: UUID
    artifact_type: str
    payload: Any
    created_at: datetime


class ArtifactRef(TypedDict):
    """Stable reference carried through graph state."""

    artifact_id: str
    artifact_type: str


class ArtifactWriter(Protocol):
    """Minimal write seam required by retrieval orchestration."""

    async def create(
        self,
        *,
        scope: ArtifactScope,
        artifact_type: str,
        payload: Any,
        artifact_id: UUID | None = None,
    ) -> ArtifactRecord:
        """Create or idempotently reuse one Artifact."""
        ...


class ArtifactStore(ArtifactWriter, Protocol):
    """Tenant-scoped Artifact seam including response-boundary reads."""

    async def get(self, *, scope: ArtifactScope, artifact_id: UUID) -> ArtifactRecord:
        """Read one Artifact within a Tenant boundary."""
        ...


class ArtifactRepository:
    """Persist and retrieve immutable tenant-scoped Artifacts."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    async def create(
        self,
        *,
        scope: ArtifactScope,
        artifact_type: str,
        payload: Any,
        artifact_id: UUID | None = None,
    ) -> ArtifactRecord:
        """Create one Artifact within a Tenant boundary."""
        artifact_id = artifact_id or uuid4()
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """SELECT 1
                    FROM langgraph_v2.conversations AS conversations
                    JOIN langgraph_v2.messages AS messages
                      ON messages.tenant_id = conversations.tenant_id
                     AND messages.conversation_id = conversations.conversation_id
                    WHERE conversations.tenant_id = %s
                      AND conversations.conversation_id = %s
                      AND conversations.owner_subject_id = %s
                      AND messages.turn_id = %s
                      AND messages.role = 'user'""",
                    (
                        scope.context.tenant_id,
                        scope.conversation_id,
                        scope.context.subject_id,
                        scope.turn_id,
                    ),
                )
                if await cursor.fetchone() is None:
                    raise ArtifactNotFound(str(artifact_id))
                await cursor.execute(
                    """INSERT INTO langgraph_v2.artifacts
                    (tenant_id, artifact_id, conversation_id, turn_id,
                     artifact_type, payload)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, artifact_id) DO NOTHING
                    RETURNING tenant_id, conversation_id, turn_id, artifact_id,
                              artifact_type, payload, created_at""",
                    (
                        scope.context.tenant_id,
                        artifact_id,
                        scope.conversation_id,
                        scope.turn_id,
                        artifact_type,
                        Jsonb(payload),
                    ),
                )
                row = await cursor.fetchone()
                if row is None:
                    await cursor.execute(
                        """SELECT tenant_id, conversation_id, turn_id, artifact_id,
                                  artifact_type, payload, created_at
                        FROM langgraph_v2.artifacts
                        WHERE tenant_id = %s AND artifact_id = %s""",
                        (scope.context.tenant_id, artifact_id),
                    )
                    existing = await cursor.fetchone()
                    if existing is None:
                        raise ArtifactNotFound(str(artifact_id))
                    record = ArtifactRecord.model_validate(existing)
                    if (
                        record.conversation_id != scope.conversation_id
                        or record.turn_id != scope.turn_id
                        or record.artifact_type != artifact_type
                        or record.payload != payload
                    ):
                        raise ArtifactInvariantConflict(str(artifact_id))
                    return record
        return ArtifactRecord.model_validate(row)

    async def get(self, *, scope: ArtifactScope, artifact_id: UUID) -> ArtifactRecord:
        """Read one Artifact or raise a tenant-scoped not-found error."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """SELECT artifacts.tenant_id, artifacts.conversation_id,
                              artifacts.turn_id, artifacts.artifact_id,
                              artifacts.artifact_type, artifacts.payload,
                              artifacts.created_at
                    FROM langgraph_v2.artifacts AS artifacts
                    JOIN langgraph_v2.conversations AS conversations
                      ON conversations.tenant_id = artifacts.tenant_id
                     AND conversations.conversation_id = artifacts.conversation_id
                    JOIN langgraph_v2.messages AS messages
                      ON messages.tenant_id = artifacts.tenant_id
                     AND messages.conversation_id = artifacts.conversation_id
                     AND messages.turn_id = artifacts.turn_id
                     AND messages.role = 'user'
                    WHERE artifacts.tenant_id = %s
                      AND artifacts.conversation_id = %s
                      AND artifacts.turn_id = %s
                      AND artifacts.artifact_id = %s
                      AND conversations.owner_subject_id = %s""",
                    (
                        scope.context.tenant_id,
                        scope.conversation_id,
                        scope.turn_id,
                        artifact_id,
                        scope.context.subject_id,
                    ),
                )
                row = await cursor.fetchone()
        if row is None:
            raise ArtifactNotFound(str(artifact_id))
        return ArtifactRecord.model_validate(row)
