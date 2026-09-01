"""Tenant-scoped durable Conversation registry."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.config.models import LangGraphRuntimeMode
from app.langgraph_v2.authorization import TrustedRequestContext


class ConversationNotFound(LookupError):
    """A Conversation is absent from the requested Tenant boundary."""


class ConversationModeConflict(RuntimeError):
    """A Conversation belongs to a different fixed runtime mode."""


class ConversationRecord(BaseModel):
    """One durable Conversation owned by a Tenant Subject."""

    conversation_id: UUID
    tenant_id: str
    owner_subject_id: str
    runtime_mode: LangGraphRuntimeMode
    created_at: datetime
    updated_at: datetime


class ConversationRepository:
    """Create and authorize the minimal durable Conversation registry."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    @staticmethod
    def _conversation_uuid(conversation_id: UUID | str) -> UUID:
        return UUID(str(conversation_id))

    async def create_conversation(
        self,
        *,
        context: TrustedRequestContext,
        runtime_mode: LangGraphRuntimeMode,
    ) -> ConversationRecord:
        """Create a Conversation whose UUID identity is assigned by PostgreSQL."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    INSERT INTO langgraph_v2.conversations (
                        tenant_id, owner_subject_id, runtime_mode
                    ) VALUES (%s, %s, %s)
                    RETURNING conversation_id, tenant_id, owner_subject_id,
                              runtime_mode, created_at, updated_at
                    """,
                    (context.tenant_id, context.subject_id, runtime_mode.value),
                )
                row = await cursor.fetchone()
        assert row is not None
        return ConversationRecord.model_validate(row)

    async def get_conversation(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: UUID | str,
        runtime_mode: LangGraphRuntimeMode | None = None,
    ) -> ConversationRecord:
        """Return only a Conversation owned by the trusted Subject and mode."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT conversation_id, tenant_id, owner_subject_id,
                           runtime_mode, created_at, updated_at
                    FROM langgraph_v2.conversations
                    WHERE conversation_id = %s
                      AND tenant_id = %s
                      AND owner_subject_id = %s
                    """,
                    (
                        self._conversation_uuid(conversation_id),
                        context.tenant_id,
                        context.subject_id,
                    ),
                )
                row = await cursor.fetchone()
        if row is None:
            raise ConversationNotFound(conversation_id)
        conversation = ConversationRecord.model_validate(row)
        if runtime_mode is not None and conversation.runtime_mode is not runtime_mode:
            raise ConversationModeConflict(conversation_id)
        return conversation

    async def touch_conversation(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: UUID | str,
    ) -> None:
        """Record accepted query activity for an authorized Conversation."""
        async with self._pool.connection() as connection:
            result = await connection.execute(
                """
                UPDATE langgraph_v2.conversations
                SET updated_at = clock_timestamp()
                WHERE conversation_id = %s
                  AND tenant_id = %s
                  AND owner_subject_id = %s
                """,
                (
                    self._conversation_uuid(conversation_id),
                    context.tenant_id,
                    context.subject_id,
                ),
            )
        if result.rowcount != 1:
            raise ConversationNotFound(conversation_id)
