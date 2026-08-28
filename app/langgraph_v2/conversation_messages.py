"""Tenant-scoped durable Conversation and Message persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.langgraph_v2.run_events import ClaimFenced, RepositoryNotFound


class ConversationNotFound(RepositoryNotFound):
    """A Conversation is absent from the requested Tenant boundary."""


class MessageNotFound(RepositoryNotFound):
    """A Message is absent from the requested Tenant boundary."""


class MessageInvariantConflict(RuntimeError):
    """An idempotency key was reused for a different Message."""


class ConversationRecord(BaseModel):
    """One durable Conversation within a Tenant."""

    tenant_id: str
    conversation_id: str
    created_at: datetime


class MessageRecord(BaseModel):
    """One durable user or assistant Message."""

    tenant_id: str
    message_id: UUID
    conversation_id: str
    run_id: UUID
    role: Literal["user", "assistant"]
    content: str
    idempotency_key: str
    created_at: datetime


class ConversationMessageRepository:
    """Persist tenant-isolated Conversations and exactly-once Messages."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    async def resolve_conversation(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
    ) -> ConversationRecord:
        """Resolve or create the Conversation for a legacy session identity."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                return await self.resolve_conversation_in_transaction(
                    connection,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                )

    async def resolve_conversation_in_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        conversation_id: str,
    ) -> ConversationRecord:
        """Resolve within a caller-owned transaction for the admission seam."""
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                INSERT INTO langgraph_v2.conversations (
                    tenant_id, conversation_id
                ) VALUES (%s, %s)
                ON CONFLICT (tenant_id, conversation_id) DO NOTHING
                """,
                (tenant_id, conversation_id),
            )
            await cursor.execute(
                """
                SELECT tenant_id, conversation_id, created_at
                FROM langgraph_v2.conversations
                WHERE tenant_id = %s AND conversation_id = %s
                """,
                (tenant_id, conversation_id),
            )
            row = await cursor.fetchone()
        return ConversationRecord.model_validate(row)

    async def get_conversation(
        self,
        tenant_id: str,
        conversation_id: str,
    ) -> ConversationRecord:
        """Return a Conversation without crossing Tenant boundaries."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, conversation_id, created_at
                    FROM langgraph_v2.conversations
                    WHERE tenant_id = %s AND conversation_id = %s
                    """,
                    (tenant_id, conversation_id),
                )
                row = await cursor.fetchone()
        if row is None:
            raise ConversationNotFound(conversation_id)
        return ConversationRecord.model_validate(row)

    async def persist_user_message(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        run_id: UUID,
        content: str,
        idempotency_key: str,
    ) -> MessageRecord:
        """Persist exactly one user Message for a Run start."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                return await self.persist_user_message_in_transaction(
                    connection,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    content=content,
                    idempotency_key=idempotency_key,
                )

    async def persist_user_message_in_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        conversation_id: str,
        run_id: UUID,
        content: str,
        idempotency_key: str,
    ) -> MessageRecord:
        """Write the user Message inside a caller-owned start transaction."""
        return await self._persist_message_in_transaction(
            connection,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            run_id=run_id,
            role="user",
            content=content,
            idempotency_key=idempotency_key,
        )

    async def persist_assistant_message_after_completion(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        run_id: UUID,
        content: str,
        idempotency_key: str,
    ) -> MessageRecord:
        """Persist an assistant Message only for an already completed Run."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        SELECT status, conversation_id
                        FROM langgraph_v2.runs
                        WHERE tenant_id = %s AND run_id = %s
                        FOR SHARE
                        """,
                        (tenant_id, run_id),
                    )
                    run = await cursor.fetchone()
                if (
                    run is None
                    or run["status"] != "completed"
                    or run["conversation_id"] != conversation_id
                ):
                    raise ClaimFenced(str(run_id))
                return await self._persist_message_in_transaction(
                    connection,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    role="assistant",
                    content=content,
                    idempotency_key=idempotency_key,
                )

    async def persist_assistant_message_in_terminal_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        conversation_id: str,
        run_id: UUID,
        owner_instance_id: str,
        execution_epoch: int,
        content: str,
        idempotency_key: str,
    ) -> MessageRecord:
        """Write a safe answer through a caller-owned, epoch-fenced transaction."""
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT status, conversation_id, owner_instance_id,
                       execution_epoch,
                       expires_at > clock_timestamp() AS claim_active
                FROM langgraph_v2.runs
                WHERE tenant_id = %s AND run_id = %s
                FOR UPDATE
                """,
                (tenant_id, run_id),
            )
            run = await cursor.fetchone()
        if (
            run is None
            or run["status"] != "running"
            or run["conversation_id"] != conversation_id
            or run["owner_instance_id"] != owner_instance_id
            or run["execution_epoch"] != execution_epoch
            or not run["claim_active"]
        ):
            raise ClaimFenced(str(run_id))
        return await self._persist_message_in_transaction(
            connection,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            run_id=run_id,
            role="assistant",
            content=content,
            idempotency_key=idempotency_key,
        )

    async def get_message(self, tenant_id: str, message_id: UUID) -> MessageRecord:
        """Return a Message without revealing another Tenant's record."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, message_id, conversation_id, run_id, role,
                           content, idempotency_key, created_at
                    FROM langgraph_v2.messages
                    WHERE tenant_id = %s AND message_id = %s
                    """,
                    (tenant_id, message_id),
                )
                row = await cursor.fetchone()
        if row is None:
            raise MessageNotFound(str(message_id))
        return MessageRecord.model_validate(row)

    async def list_messages(
        self,
        tenant_id: str,
        conversation_id: str,
    ) -> list[MessageRecord]:
        """Return one Conversation's Messages in durable chronological order."""
        await self.get_conversation(tenant_id, conversation_id)
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, message_id, conversation_id, run_id, role,
                           content, idempotency_key, created_at
                    FROM langgraph_v2.messages
                    WHERE tenant_id = %s AND conversation_id = %s
                    ORDER BY created_at, message_id
                    """,
                    (tenant_id, conversation_id),
                )
                rows = await cursor.fetchall()
        return [MessageRecord.model_validate(row) for row in rows]

    async def _persist_message_in_transaction(
        self,
        connection: Any,
        *,
        tenant_id: str,
        conversation_id: str,
        run_id: UUID,
        role: Literal["user", "assistant"],
        content: str,
        idempotency_key: str,
    ) -> MessageRecord:
        message_id = uuid4()
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                INSERT INTO langgraph_v2.messages (
                    tenant_id, message_id, conversation_id, run_id, role,
                    content, idempotency_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING tenant_id, message_id, conversation_id, run_id, role,
                          content, idempotency_key, created_at
                """,
                (
                    tenant_id,
                    message_id,
                    conversation_id,
                    run_id,
                    role,
                    content,
                    idempotency_key,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                await cursor.execute(
                    """
                    SELECT tenant_id, message_id, conversation_id, run_id, role,
                           content, idempotency_key, created_at
                    FROM langgraph_v2.messages
                    WHERE tenant_id = %s AND idempotency_key = %s
                    """,
                    (tenant_id, idempotency_key),
                )
                row = await cursor.fetchone()
        if row is None or (
            row["conversation_id"] != conversation_id
            or row["run_id"] != run_id
            or row["role"] != role
            or row["content"] != content
        ):
            raise MessageInvariantConflict(idempotency_key)
        return MessageRecord.model_validate(row)
