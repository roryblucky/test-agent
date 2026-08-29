from __future__ import annotations

from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
)


@pytest.mark.asyncio
async def test_conversation_authorization_persists_subject_and_stable_thread(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")

        first = await repository.resolve_conversation(
            context=context, conversation_id="conversation-1"
        )
        repeated = await repository.resolve_conversation(
            context=context, conversation_id="conversation-1"
        )

        assert first == repeated
        assert first.owner_subject_id == "subject-a"
        distinct = await repository.resolve_conversation(
            context=context, conversation_id="conversation-2"
        )
        assert first.thread_id
        assert first.thread_id == repeated.thread_id
        assert first.thread_id != distinct.thread_id

        with pytest.raises(ConversationNotFound):
            await repository.get_conversation(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-b"
                ),
                conversation_id="conversation-1",
            )
        with pytest.raises(ConversationNotFound):
            await repository.get_conversation(
                context=TrustedRequestContext(
                    tenant_id="tenant-b", subject_id="subject-a"
                ),
                conversation_id="conversation-1",
            )
        with pytest.raises(ConversationNotFound):
            await repository.get_conversation(
                context=context,
                conversation_id="missing-conversation",
            )


@pytest.mark.asyncio
async def test_thread_lookup_and_messages_require_conversation_authorization(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        conversation = await repository.resolve_conversation(
            context=context, conversation_id="conversation-1"
        )
        message = await repository.persist_user_message(
            tenant_id="tenant-a",
            conversation_id=conversation.conversation_id,
            run_id=uuid4(),
            content="hello",
            idempotency_key="authorization:user",
        )

        assert (
            await repository.get_conversation_by_thread(
                context=context, thread_id=conversation.thread_id
            )
        ) == conversation
        assert (
            await repository.get_message(
                context=context,
                conversation_id=conversation.conversation_id,
                message_id=message.message_id,
            )
        ) == message
        assert await repository.list_messages(
            context=context, conversation_id=conversation.conversation_id
        ) == [message]

        other_subject = TrustedRequestContext(
            tenant_id="tenant-a", subject_id="subject-b"
        )
        with pytest.raises(ConversationNotFound):
            await repository.get_conversation_by_thread(
                context=other_subject, thread_id=conversation.thread_id
            )
        with pytest.raises(ConversationNotFound):
            await repository.get_conversation_by_thread(
                context=TrustedRequestContext(
                    tenant_id="tenant-b", subject_id="subject-a"
                ),
                thread_id=conversation.thread_id,
            )
        with pytest.raises(ConversationNotFound):
            await repository.get_conversation_by_thread(
                context=context, thread_id="missing-thread"
            )
        with pytest.raises(ConversationNotFound):
            await repository.get_message(
                context=other_subject,
                conversation_id=conversation.conversation_id,
                message_id=message.message_id,
            )
        with pytest.raises(ConversationNotFound):
            await repository.list_messages(
                context=other_subject,
                conversation_id=conversation.conversation_id,
            )
        with pytest.raises(ConversationNotFound):
            await repository.list_messages(
                context=TrustedRequestContext(
                    tenant_id="tenant-b", subject_id="subject-a"
                ),
                conversation_id=conversation.conversation_id,
            )
        with pytest.raises(ConversationNotFound):
            await repository.list_messages(
                context=context, conversation_id="missing-conversation"
            )
