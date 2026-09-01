"""Tenant-scoped durable Conversation and Message persistence."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import thread_id_for


class ConversationNotFound(LookupError):
    """A Conversation is absent from the requested Tenant boundary."""


class MessageNotFound(LookupError):
    """A Message is absent from the requested Tenant boundary."""


class TurnNotFound(LookupError):
    """A Turn is absent from the requested Tenant/Conversation boundary."""


class MessageInvariantConflict(RuntimeError):
    """An idempotency key was reused for a different Message."""


def turn_id_for_client_request(
    tenant_id: str, conversation_id: str, client_request_id: str
) -> UUID:
    """Derive a collision-safe stable Turn ID for one client retry identity."""
    payload = json.dumps(
        [tenant_id, conversation_id, client_request_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return uuid5(NAMESPACE_URL, payload)


class ConversationRecord(BaseModel):
    """One durable Conversation within a Tenant."""

    tenant_id: str
    conversation_id: str
    owner_subject_id: str
    thread_id: str
    created_at: datetime


class MessageRecord(BaseModel):
    """One durable user or assistant Message."""

    tenant_id: str
    message_id: UUID
    conversation_id: str
    turn_id: UUID
    role: Literal["user", "assistant"]
    content: str
    idempotency_key: str
    created_at: datetime


class TurnRecord(BaseModel):
    """A durable Conversation interaction anchored by its user Message."""

    tenant_id: str
    conversation_id: str
    turn_id: UUID
    created_at: datetime


class ConversationMessageRepository:
    """Persist tenant-isolated Conversations and exactly-once Messages."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    async def create_turn(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
        turn_id: UUID,
        content: str,
        idempotency_key: str,
    ) -> TurnRecord:
        """Create one Turn and its exactly-once user Message atomically."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                await self._resolve_conversation_in_transaction(
                    connection,
                    context=context,
                    conversation_id=conversation_id,
                    for_update=True,
                )
                await self._persist_message_in_transaction(
                    connection,
                    tenant_id=context.tenant_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    role="user",
                    content=content,
                    idempotency_key=idempotency_key,
                )
                return await self._get_turn_in_transaction(
                    connection,
                    tenant_id=context.tenant_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                )

    async def get_turn(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
        turn_id: UUID,
    ) -> TurnRecord:
        """Return an authorized Turn anchored to its authoritative user Message."""
        await self.get_conversation(context=context, conversation_id=conversation_id)
        async with self._pool.connection() as connection:
            return await self._get_turn_in_transaction(
                connection,
                tenant_id=context.tenant_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
            )

    async def resolve_conversation(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: str | None = None,
    ) -> ConversationRecord:
        """Resolve or create a Conversation for the trusted Subject."""
        resolved_conversation_id = conversation_id or str(uuid4())
        async with self._pool.connection() as connection:
            async with connection.transaction():
                return await self._resolve_conversation_in_transaction(
                    connection,
                    context=context,
                    conversation_id=resolved_conversation_id,
                )

    async def _resolve_conversation_in_transaction(
        self,
        connection: Any,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
        for_update: bool = False,
    ) -> ConversationRecord:
        """Resolve within a caller-owned transaction for the admission seam."""
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                INSERT INTO langgraph_v2.conversations (
                    tenant_id, conversation_id, owner_subject_id, thread_id
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id, conversation_id) DO NOTHING
                """,
                (
                    context.tenant_id,
                    conversation_id,
                    context.subject_id,
                    thread_id_for(context.tenant_id, conversation_id),
                ),
            )
            if for_update:
                await cursor.execute(
                    """
                    SELECT 1
                    FROM langgraph_v2.conversations
                    WHERE tenant_id = %s AND conversation_id = %s
                    FOR UPDATE
                    """,
                    (context.tenant_id, conversation_id),
                )
            await cursor.execute(
                """
                SELECT tenant_id, conversation_id, owner_subject_id, thread_id,
                       created_at
                FROM langgraph_v2.conversations
                WHERE tenant_id = %s AND conversation_id = %s
                """,
                (context.tenant_id, conversation_id),
            )
            row = await cursor.fetchone()
        if row is None or row["owner_subject_id"] != context.subject_id:
            raise ConversationNotFound(conversation_id)
        return ConversationRecord.model_validate(row)

    async def get_conversation(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
    ) -> ConversationRecord:
        """Return only a Conversation owned by the trusted Subject."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, conversation_id, owner_subject_id, thread_id,
                           created_at
                    FROM langgraph_v2.conversations
                    WHERE tenant_id = %s AND owner_subject_id = %s
                      AND conversation_id = %s
                    """,
                    (context.tenant_id, context.subject_id, conversation_id),
                )
                row = await cursor.fetchone()
        if row is None:
            raise ConversationNotFound(conversation_id)
        return ConversationRecord.model_validate(row)

    async def persist_assistant_message(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
        content: str,
        idempotency_key: str,
        turn_id: UUID,
    ) -> MessageRecord:
        """Persist exactly one assistant Message for an existing Turn."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                await self._resolve_conversation_in_transaction(
                    connection,
                    context=context,
                    conversation_id=conversation_id,
                )
                await self._require_user_turn_in_transaction(
                    connection,
                    tenant_id=context.tenant_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                )
                return await self._persist_message_in_transaction(
                    connection,
                    tenant_id=context.tenant_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    role="assistant",
                    content=content,
                    idempotency_key=idempotency_key,
                )

    async def _get_turn_in_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        conversation_id: str,
        turn_id: UUID,
    ) -> TurnRecord:
        """Read a Turn using its authoritative user Message timestamp."""
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                    SELECT tenant_id, conversation_id, turn_id, created_at
                FROM langgraph_v2.messages
                WHERE tenant_id = %s AND conversation_id = %s
                  AND turn_id = %s AND role = 'user'
                """,
                (tenant_id, conversation_id, turn_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise TurnNotFound(str(turn_id))
        return TurnRecord.model_validate(row)

    async def _require_user_turn_in_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        conversation_id: str,
        turn_id: UUID,
    ) -> None:
        """Ensure an assistant attaches to an existing user Message Turn."""
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT turn_id
                FROM langgraph_v2.messages
                WHERE tenant_id = %s AND conversation_id = %s
                  AND turn_id = %s AND role = 'user'
                """,
                (tenant_id, conversation_id, turn_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise TurnNotFound(str(turn_id))

    async def get_message(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
        message_id: UUID,
    ) -> MessageRecord:
        """Return a Message only after authorizing its Conversation owner."""
        await self.get_conversation(context=context, conversation_id=conversation_id)
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, message_id, conversation_id, role,
                           turn_id, content, idempotency_key, created_at
                    FROM langgraph_v2.messages
                    WHERE tenant_id = %s AND conversation_id = %s
                      AND message_id = %s
                    """,
                    (context.tenant_id, conversation_id, message_id),
                )
                row = await cursor.fetchone()
        if row is None:
            raise MessageNotFound(str(message_id))
        return MessageRecord.model_validate(row)

    async def list_messages(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
    ) -> list[MessageRecord]:
        """Return authorized Conversation Messages chronologically."""
        await self.get_conversation(context=context, conversation_id=conversation_id)
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, message_id, conversation_id, role,
                           turn_id, content, idempotency_key, created_at
                    FROM langgraph_v2.messages
                    WHERE tenant_id = %s AND conversation_id = %s
                    ORDER BY created_at, message_id
                    """,
                    (context.tenant_id, conversation_id),
                )
                rows = await cursor.fetchall()
        return [MessageRecord.model_validate(row) for row in rows]

    async def _persist_message_in_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        conversation_id: str,
        turn_id: UUID,
        role: Literal["user", "assistant"],
        content: str,
        idempotency_key: str,
    ) -> MessageRecord:
        message_id = uuid4()
        insert_sql = """
            INSERT INTO langgraph_v2.messages (
                tenant_id, message_id, conversation_id, turn_id, role,
                content, idempotency_key
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING tenant_id, message_id, conversation_id, role,
                      turn_id, content, idempotency_key, created_at
            """
        insert_params = (
            tenant_id,
            message_id,
            conversation_id,
            turn_id,
            role,
            content,
            idempotency_key,
        )
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(insert_sql, insert_params)
            row = await cursor.fetchone()
            if row is None:
                await cursor.execute(
                    """
                    SELECT tenant_id, message_id, conversation_id, role,
                           turn_id, content, idempotency_key, created_at
                    FROM langgraph_v2.messages
                    WHERE tenant_id = %s
                      AND (
                          idempotency_key = %s
                          OR (conversation_id = %s AND turn_id = %s AND role = %s)
                      )
                    """,
                    (tenant_id, idempotency_key, conversation_id, turn_id, role),
                )
                row = await cursor.fetchone()
        if row is None:
            raise MessageInvariantConflict(idempotency_key)
        same_message = (
            row["conversation_id"] == conversation_id
            and row["turn_id"] == turn_id
            and row["role"] == role
            and row["content"] == content
            and row["idempotency_key"] == idempotency_key
        )
        if not same_message:
            raise MessageInvariantConflict(idempotency_key)
        return MessageRecord.model_validate(row)
