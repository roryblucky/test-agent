"""Tenant-scoped durable Conversation and Message persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import thread_id_for
from app.langgraph_v2.run_events import ClaimFenced, RepositoryNotFound


class ConversationNotFound(RepositoryNotFound):
    """A Conversation is absent from the requested Tenant boundary."""


class MessageNotFound(RepositoryNotFound):
    """A Message is absent from the requested Tenant boundary."""


class TurnNotFound(RepositoryNotFound):
    """A Turn is absent from the requested Tenant/Conversation boundary."""


class MessageInvariantConflict(RuntimeError):
    """An idempotency key was reused for a different Message."""


class ResumeExpired(RuntimeError):
    """A Turn is no longer inside its fixed Resume window."""


DEFAULT_RESUME_TTL = timedelta(hours=1)


def resume_deadline_for(
    created_at: datetime,
    configured_resume_ttl: timedelta | int = DEFAULT_RESUME_TTL,
) -> datetime:
    """Calculate the immutable Resume deadline from the user Message timestamp."""
    ttl = (
        configured_resume_ttl
        if isinstance(configured_resume_ttl, timedelta)
        else timedelta(seconds=configured_resume_ttl)
    )
    return created_at + ttl


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
    run_id: UUID
    turn_id: UUID
    role: Literal["user", "assistant"]
    content: str
    idempotency_key: str
    created_at: datetime


class TurnRecord(BaseModel):
    """A durable interaction and its fixed user-Message Resume anchor."""

    tenant_id: str
    conversation_id: str
    turn_id: UUID
    run_id: UUID
    created_at: datetime
    resume_deadline: datetime


class ConversationMessageRepository:
    """Persist tenant-isolated Conversations and exactly-once Messages."""

    def __init__(
        self,
        pool: AsyncConnectionPool[Any],
        *,
        resume_ttl: timedelta | int = DEFAULT_RESUME_TTL,
    ) -> None:
        self._pool = pool
        self._resume_ttl = (
            resume_ttl
            if isinstance(resume_ttl, timedelta)
            else timedelta(seconds=resume_ttl)
        )

    async def create_turn(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
        run_id: UUID,
        content: str,
        idempotency_key: str,
        turn_id: UUID | None = None,
    ) -> TurnRecord:
        """Create one Turn and its exactly-once user Message atomically."""
        resolved_turn_id = turn_id or uuid5(
            NAMESPACE_URL,
            f"{context.tenant_id}:{conversation_id}:{idempotency_key}",
        )
        async with self._pool.connection() as connection:
            async with connection.transaction():
                await self._resolve_conversation_in_transaction(
                    connection,
                    context=context,
                    conversation_id=conversation_id,
                )
                await self._persist_message_in_transaction(
                    connection,
                    tenant_id=context.tenant_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    turn_id=resolved_turn_id,
                    role="user",
                    content=content,
                    idempotency_key=idempotency_key,
                )
                return await self._get_turn_in_transaction(
                    connection,
                    tenant_id=context.tenant_id,
                    conversation_id=conversation_id,
                    turn_id=resolved_turn_id,
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

    async def get_turn_for_resume(
        self,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
        turn_id: UUID,
        now: datetime | None = None,
    ) -> TurnRecord:
        """Return a Turn only while its original user-Message deadline is active."""
        turn = await self.get_turn(
            context=context,
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        current_time = now or datetime.now(UTC)
        if current_time >= turn.resume_deadline:
            raise ResumeExpired(str(turn_id))
        return turn

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

    async def get_conversation_by_thread(
        self,
        *,
        context: TrustedRequestContext,
        thread_id: str,
    ) -> ConversationRecord:
        """Resolve a thread only through its authorized Conversation owner."""
        async with self._pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT tenant_id, conversation_id, owner_subject_id, thread_id,
                           created_at
                    FROM langgraph_v2.conversations
                    WHERE tenant_id = %s AND owner_subject_id = %s
                      AND thread_id = %s
                    """,
                    (context.tenant_id, context.subject_id, thread_id),
                )
                row = await cursor.fetchone()
        if row is None:
            raise ConversationNotFound(thread_id)
        return ConversationRecord.model_validate(row)

    async def persist_user_message(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        run_id: UUID,
        content: str,
        idempotency_key: str,
        turn_id: UUID | None = None,
    ) -> MessageRecord:
        """Persist exactly one user Message for a Run start."""
        async with self._pool.connection() as connection:
            async with connection.transaction():
                return await self.persist_user_message_in_transaction(
                    connection,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    turn_id=turn_id,
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
        turn_id: UUID | None = None,
    ) -> MessageRecord:
        """Write the user Message inside a caller-owned start transaction."""
        return await self._persist_message_in_transaction(
            connection,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            run_id=run_id,
            turn_id=turn_id or run_id,
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
        turn_id: UUID | None = None,
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
                resolved_turn_id = (
                    turn_id
                    or await self._turn_id_for_run(
                        connection,
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        run_id=run_id,
                    )
                    or run_id
                )
                return await self._persist_message_in_transaction(
                    connection,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    turn_id=resolved_turn_id,
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
        turn_id: UUID | None = None,
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
        resolved_turn_id = (
            turn_id
            or await self._turn_id_for_run(
                connection,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                run_id=run_id,
            )
            or run_id
        )
        return await self._persist_message_in_transaction(
            connection,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            run_id=run_id,
            turn_id=resolved_turn_id,
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
                SELECT tenant_id, conversation_id, turn_id, run_id, created_at
                FROM langgraph_v2.messages
                WHERE tenant_id = %s AND conversation_id = %s
                  AND turn_id = %s AND role = 'user'
                """,
                (tenant_id, conversation_id, turn_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise TurnNotFound(str(turn_id))
        return TurnRecord.model_validate(
            {
                **row,
                "resume_deadline": resume_deadline_for(
                    row["created_at"], self._resume_ttl
                ),
            }
        )

    async def _turn_id_for_run(
        self,
        connection: Any,
        *,
        tenant_id: str,
        conversation_id: str,
        run_id: UUID,
    ) -> UUID | None:
        """Resolve the Turn from its authoritative user Message during migration."""
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT turn_id
                FROM langgraph_v2.messages
                WHERE tenant_id = %s AND conversation_id = %s
                  AND run_id = %s AND role = 'user'
                """,
                (tenant_id, conversation_id, run_id),
            )
            row = await cursor.fetchone()
        return None if row is None else row["turn_id"]

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
                    SELECT tenant_id, message_id, conversation_id, run_id, role,
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
                    SELECT tenant_id, message_id, conversation_id, run_id, role,
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
        run_id: UUID,
        turn_id: UUID,
        role: Literal["user", "assistant"],
        content: str,
        idempotency_key: str,
    ) -> MessageRecord:
        message_id = uuid4()
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                INSERT INTO langgraph_v2.messages (
                    tenant_id, message_id, conversation_id, run_id, turn_id, role,
                    content, idempotency_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING tenant_id, message_id, conversation_id, run_id, role,
                          turn_id, content, idempotency_key, created_at
                """,
                (
                    tenant_id,
                    message_id,
                    conversation_id,
                    run_id,
                    turn_id,
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
                           turn_id, content, idempotency_key, created_at
                    FROM langgraph_v2.messages
                    WHERE tenant_id = %s AND idempotency_key = %s
                    """,
                    (tenant_id, idempotency_key),
                )
                row = await cursor.fetchone()
        if row is None or (
            row["conversation_id"] != conversation_id
            or row["run_id"] != run_id
            or row["turn_id"] != turn_id
            or row["role"] != role
            or row["content"] != content
        ):
            raise MessageInvariantConflict(idempotency_key)
        return MessageRecord.model_validate(row)
