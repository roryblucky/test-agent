from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
    MessageInvariantConflict,
    ResumeExpired,
    TurnNotFound,
)


@pytest.mark.asyncio
async def test_turn_creation_is_idempotent_and_resume_deadline_uses_user_message_time(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        repository = ConversationMessageRepository(
            pool, resume_ttl=timedelta(minutes=5)
        )
        await repository.resolve_conversation(
            context=context,
            conversation_id="session-1",
        )
        turn_id = uuid4()
        first = await repository.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key="turn-1:user",
        )
        repeated = await repository.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key="turn-1:user",
        )

        assert repeated == first
        assert first.turn_id == turn_id
        original_deadline = first.resume_deadline
        assert original_deadline - first.created_at == timedelta(minutes=5)
        message = (
            await repository.list_messages(context=context, conversation_id="session-1")
        )[0]
        assert message.turn_id == turn_id

        with pytest.raises(MessageInvariantConflict):
            await repository.create_turn(
                context=context,
                conversation_id="session-1",
                turn_id=turn_id,
                content="changed",
                idempotency_key="turn-1:user",
            )
        with pytest.raises(MessageInvariantConflict):
            await repository.create_turn(
                context=context,
                conversation_id="session-1",
                turn_id=turn_id,
                content="hello",
                idempotency_key="turn-1:different-key",
            )

        repository_with_longer_ttl = ConversationMessageRepository(
            pool, resume_ttl=timedelta(minutes=30)
        )
        retried_after_config_change = await repository_with_longer_ttl.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key="turn-1:user",
        )
        turn = await repository_with_longer_ttl.get_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
        )
        assert retried_after_config_change == first
        assert turn.resume_deadline == original_deadline
        with pytest.raises(ResumeExpired):
            await repository.get_turn_for_resume(
                context=context,
                conversation_id="session-1",
                turn_id=turn_id,
                now=original_deadline + timedelta(seconds=1),
            )


@pytest.mark.asyncio
async def test_session_identity_resolves_once_per_tenant(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)

        first = await repository.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="session-1",
        )
        repeated = await repository.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="session-1",
        )
        isolated = await repository.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-b", subject_id="subject-a"),
            conversation_id="session-1",
        )

        assert first == repeated
        assert first.conversation_id == isolated.conversation_id
        assert first.tenant_id != isolated.tenant_id
        with pytest.raises(ConversationNotFound):
            await repository.get_conversation(
                context=TrustedRequestContext(
                    tenant_id="tenant-c", subject_id="subject-a"
                ),
                conversation_id=first.conversation_id,
            )


@pytest.mark.asyncio
async def test_turn_messages_are_idempotent_without_a_run(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        await repository.resolve_conversation(
            context=context,
            conversation_id="session-1",
        )
        turn_id = uuid4()

        first_turn = await repository.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key=f"turn:{turn_id}:user",
        )
        repeated_turn = await repository.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key=f"turn:{turn_id}:user",
        )
        first = await repository.persist_assistant_message(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="answer",
            idempotency_key=f"turn:{turn_id}:assistant",
        )
        repeated = await repository.persist_assistant_message(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="answer",
            idempotency_key=f"turn:{turn_id}:assistant",
        )

        assert repeated == first
        assert repeated_turn == first_turn
        records = await repository.list_messages(
            context=context, conversation_id="session-1"
        )
        assert [(record.role, record.content) for record in records] == [
            ("user", "hello"),
            ("assistant", "answer"),
        ]
        with pytest.raises(MessageInvariantConflict):
            await repository.persist_assistant_message(
                context=context,
                conversation_id="session-1",
                turn_id=turn_id,
                content="changed",
                idempotency_key=f"turn:{turn_id}:assistant",
            )
        with pytest.raises(ConversationNotFound):
            await repository.get_message(
                context=TrustedRequestContext(
                    tenant_id="tenant-b", subject_id="subject-a"
                ),
                conversation_id="session-1",
                message_id=first.message_id,
            )


@pytest.mark.asyncio
async def test_assistant_message_requires_an_existing_user_turn(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        messages = ConversationMessageRepository(pool)
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        await messages.resolve_conversation(
            context=context,
            conversation_id="session-1",
        )
        turn_id = uuid4()
        await messages.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="question",
            idempotency_key=f"turn:{turn_id}:user",
        )
        assistant = await messages.persist_assistant_message(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="safe answer",
            idempotency_key=f"turn:{turn_id}:assistant",
        )
        assert assistant.turn_id == turn_id
        with pytest.raises(ConversationNotFound):
            await messages.persist_assistant_message(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-b"
                ),
                conversation_id="session-1",
                turn_id=turn_id,
                content="unauthorized",
                idempotency_key="unauthorized:assistant",
            )
        with pytest.raises(TurnNotFound):
            await messages.persist_assistant_message(
                context=context,
                conversation_id="session-1",
                turn_id=uuid4(),
                content="orphan",
                idempotency_key="orphan:assistant",
            )
