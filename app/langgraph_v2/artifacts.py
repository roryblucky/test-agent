"""Tenant-scoped production-shaped Artifact seam for v2 retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, TypedDict
from uuid import UUID, uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel


class ArtifactNotFound(LookupError):
    """Artifact is missing from the requested Tenant boundary."""


class ArtifactRecord(BaseModel):
    """Persisted immutable Artifact metadata and payload."""
    tenant_id: str
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
        tenant_id: str,
        artifact_type: str,
        payload: Any,
        artifact_id: UUID | None = None,
    ) -> ArtifactRecord:
        """Create or idempotently reuse one Artifact."""
        ...

class ArtifactStore(ArtifactWriter, Protocol):
    """Tenant-scoped Artifact seam including response-boundary reads."""

    async def get(self, *, tenant_id: str, artifact_id: UUID) -> ArtifactRecord:
        """Read one Artifact within a Tenant boundary."""
        ...


class ArtifactRepository:
    """Persist and retrieve immutable tenant-scoped Artifacts."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    async def create(
        self,
        *,
        tenant_id: str,
        artifact_type: str,
        payload: Any,
        artifact_id: UUID | None = None,
    ) -> ArtifactRecord:
        """Create one Artifact within a Tenant boundary."""
        artifact_id = artifact_id or uuid4()
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """INSERT INTO langgraph_v2.artifacts
                    (tenant_id, artifact_id, artifact_type, payload)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (tenant_id, artifact_id) DO UPDATE
                    SET artifact_type = EXCLUDED.artifact_type,
                        payload = EXCLUDED.payload
                    RETURNING tenant_id, artifact_id, artifact_type, payload, created_at""",
                    (tenant_id, artifact_id, artifact_type, Jsonb(payload)),
                )
                row = await cursor.fetchone()
        return ArtifactRecord.model_validate(row)

    async def get(self, *, tenant_id: str, artifact_id: UUID) -> ArtifactRecord:
        """Read one Artifact or raise a tenant-scoped not-found error."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """SELECT tenant_id, artifact_id, artifact_type, payload, created_at
                    FROM langgraph_v2.artifacts
                    WHERE tenant_id = %s AND artifact_id = %s""",
                    (tenant_id, artifact_id),
                )
                row = await cursor.fetchone()
        if row is None:
            raise ArtifactNotFound(str(artifact_id))
        return ArtifactRecord.model_validate(row)
