"""Tenant-scoped durable Conversation and Message history."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
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


class MessageNotFound(LookupError):
    """A Message is absent from the requested Tenant boundary."""


class RequestNotFound(LookupError):
    """A request has no user Message in the Conversation."""


class MessageInvariantConflict(RuntimeError):
    """A request/role identity was reused for different Message content."""


class ConversationRecord(BaseModel):
    """One durable Conversation owned by a Tenant Subject."""

    conversation_id: UUID
    tenant_id: str
    owner_subject_id: str
    runtime_mode: LangGraphRuntimeMode
    next_message_sequence: int
    created_at: datetime
    updated_at: datetime


class MessageRecord(BaseModel):
    """One durable user or final assistant Message."""

    message_id: UUID
    conversation_id: UUID
    request_id: str
    sequence: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ConversationMessageRepository:
    """Persist authorized Conversations and ordered, request-paired Messages."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    @staticmethod
    def _conversation_uuid(conversation_id: UUID | str) -> UUID:
        """Normalize a boundary value before using the UUID database key."""
        return UUID(str(conversation_id))

    async def create_conversation(
        self,
        *,
        context: TrustedRequestContext,
        runtime_mode: LangGraphRuntimeMode,
    ) -> ConversationRecord:
        """Create a Conversation whose UUID identity is assigned by PostgreSQL."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        INSERT INTO langgraph_v2.conversations (
                            tenant_id, owner_subject_id, runtime_mode
                        ) VALUES (%s, %s, %s)
                        RETURNING conversation_id, tenant_id, owner_subject_id,
                                  runtime_mode, next_message_sequence,
                                  created_at, updated_at
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
            return await self._get_conversation_in_transaction(
                connection,
                context=context,
                conversation_id=self._conversation_uuid(conversation_id),
                runtime_mode=runtime_mode,
            )

    async def _get_conversation_in_transaction(
        self,
        connection: Any,
        *,
        context: TrustedRequestContext,
        conversation_id: UUID,
        runtime_mode: LangGraphRuntimeMode | None,
        for_update: bool = False,
    ) -> ConversationRecord:
        lock = " FOR UPDATE" if for_update else ""
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT conversation_id, tenant_id, owner_subject_id,
                       runtime_mode, next_message_sequence, created_at, updated_at
                FROM langgraph_v2.conversations
                WHERE conversation_id = %s
                  AND tenant_id = %s
                  AND owner_subject_id = %s
                """
                + lock,
                (conversation_id, context.tenant_id, context.subject_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise ConversationNotFound(conversation_id)
        conversation = ConversationRecord.model_validate(row)
        if runtime_mode is not None and conversation.runtime_mode is not runtime_mode:
            raise ConversationModeConflict(conversation_id)
        return conversation

    async def persist_user_message(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: UUID | str,
        request_id: str,
        content: str,
    ) -> MessageRecord:
        """Persist exactly one user Message for a logical request."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                await self._get_conversation_in_transaction(
                    connection,
                    context=context,
                    conversation_id=self._conversation_uuid(conversation_id),
                    runtime_mode=None,
                    for_update=True,
                )
                return await self._persist_message_in_transaction(
                    connection,
                    conversation_id=self._conversation_uuid(conversation_id),
                    request_id=request_id,
                    role="user",
                    content=content,
                )

    async def persist_assistant_message(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: UUID | str,
        request_id: str,
        content: str,
    ) -> MessageRecord:
        """Persist one final assistant Message for an existing user request."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                await self._get_conversation_in_transaction(
                    connection,
                    context=context,
                    conversation_id=self._conversation_uuid(conversation_id),
                    runtime_mode=None,
                    for_update=True,
                )
                await self._require_user_request_in_transaction(
                    connection,
                    conversation_id=self._conversation_uuid(conversation_id),
                    request_id=request_id,
                )
                return await self._persist_message_in_transaction(
                    connection,
                    conversation_id=self._conversation_uuid(conversation_id),
                    request_id=request_id,
                    role="assistant",
                    content=content,
                )

    async def _require_user_request_in_transaction(
        self,
        connection: Any,
        *,
        conversation_id: UUID,
        request_id: str,
    ) -> None:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT 1 FROM langgraph_v2.messages
                WHERE conversation_id = %s AND request_id = %s
                  AND role = 'user'
                """,
                (conversation_id, request_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise RequestNotFound(request_id)

    async def get_message(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: UUID | str,
        message_id: UUID,
    ) -> MessageRecord:
        """Return a Message only after authorizing its Conversation owner."""
        normalized_id = self._conversation_uuid(conversation_id)
        await self.get_conversation(context=context, conversation_id=normalized_id)
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT message_id, conversation_id, request_id, sequence,
                           role, content, created_at
                    FROM langgraph_v2.messages
                    WHERE conversation_id = %s AND message_id = %s
                    """,
                    (normalized_id, message_id),
                )
                row = await cursor.fetchone()
        if row is None:
            raise MessageNotFound(str(message_id))
        return MessageRecord.model_validate(row)

    async def list_messages(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: UUID | str,
    ) -> list[MessageRecord]:
        """Return authorized Conversation Messages in durable sequence order."""
        normalized_id = self._conversation_uuid(conversation_id)
        await self.get_conversation(context=context, conversation_id=normalized_id)
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT message_id, conversation_id, request_id, sequence,
                           role, content, created_at
                    FROM langgraph_v2.messages
                    WHERE conversation_id = %s
                    ORDER BY sequence
                    """,
                    (normalized_id,),
                )
                rows = await cursor.fetchall()
        return [MessageRecord.model_validate(row) for row in rows]

    async def _persist_message_in_transaction(
        self,
        connection: Any,
        *,
        conversation_id: UUID,
        request_id: str,
        role: Literal["user", "assistant"],
        content: str,
    ) -> MessageRecord:
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT message_id, conversation_id, request_id, sequence,
                       role, content, created_at
                FROM langgraph_v2.messages
                WHERE conversation_id = %s AND request_id = %s AND role = %s
                """,
                (conversation_id, request_id, role),
            )
            existing = await cursor.fetchone()
            if existing is not None:
                if existing["content"] != content:
                    raise MessageInvariantConflict(request_id)
                return MessageRecord.model_validate(existing)

            await cursor.execute(
                """
                UPDATE langgraph_v2.conversations
                SET next_message_sequence = next_message_sequence + 1,
                    updated_at = now()
                WHERE conversation_id = %s
                RETURNING next_message_sequence - 1 AS sequence
                """,
                (conversation_id,),
            )
            sequence_row = await cursor.fetchone()
            assert sequence_row is not None
            await cursor.execute(
                """
                INSERT INTO langgraph_v2.messages (
                    conversation_id, request_id, sequence, role, content
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING message_id, conversation_id, request_id, sequence,
                          role, content, created_at
                """,
                (
                    conversation_id,
                    request_id,
                    sequence_row["sequence"],
                    role,
                    content,
                ),
            )
            inserted = await cursor.fetchone()
        assert inserted is not None
        return MessageRecord.model_validate(inserted)
