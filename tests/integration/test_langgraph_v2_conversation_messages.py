from __future__ import annotations

import asyncio
from uuid import UUID

import pytest
from psycopg_pool import AsyncConnectionPool

from app.config.models import LangGraphRuntimeMode
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationModeConflict,
    ConversationNotFound,
    MessageInvariantConflict,
    RequestNotFound,
)


@pytest.mark.asyncio
async def test_database_generates_uuid_conversation_with_fixed_mode(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")

        conversation = await repository.create_conversation(
            context=context,
            runtime_mode=LangGraphRuntimeMode.LINEAR,
        )

        assert UUID(str(conversation.conversation_id)) == conversation.conversation_id
        assert conversation.runtime_mode is LangGraphRuntimeMode.LINEAR
        with pytest.raises(ConversationModeConflict):
            await repository.get_conversation(
                context=context,
                conversation_id=conversation.conversation_id,
                runtime_mode=LangGraphRuntimeMode.AGENT,
            )


@pytest.mark.asyncio
async def test_request_id_pairs_idempotent_ordered_messages(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        conversation = await repository.create_conversation(
            context=context,
            runtime_mode=LangGraphRuntimeMode.LINEAR,
        )

        user = await repository.persist_user_message(
            context=context,
            conversation_id=conversation.conversation_id,
            request_id="request-1",
            content="hello",
        )
        repeated_user = await repository.persist_user_message(
            context=context,
            conversation_id=conversation.conversation_id,
            request_id="request-1",
            content="hello",
        )
        assistant = await repository.persist_assistant_message(
            context=context,
            conversation_id=conversation.conversation_id,
            request_id="request-1",
            content="answer",
        )

        assert repeated_user == user
        assert user.request_id == assistant.request_id == "request-1"
        assert [user.sequence, assistant.sequence] == [1, 2]
        assert [message.role for message in await repository.list_messages(
            context=context,
            conversation_id=conversation.conversation_id,
        )] == ["user", "assistant"]

        with pytest.raises(MessageInvariantConflict):
            await repository.persist_user_message(
                context=context,
                conversation_id=conversation.conversation_id,
                request_id="request-1",
                content="changed",
            )
        with pytest.raises(RequestNotFound):
            await repository.persist_assistant_message(
                context=context,
                conversation_id=conversation.conversation_id,
                request_id="missing-request",
                content="orphan",
            )


@pytest.mark.asyncio
async def test_message_sequence_is_atomic_across_concurrent_requests(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=8
    ) as pool:
        repository = ConversationMessageRepository(pool)
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        conversation = await repository.create_conversation(
            context=context,
            runtime_mode=LangGraphRuntimeMode.LINEAR,
        )

        await asyncio.gather(
            *(
                repository.persist_user_message(
                    context=context,
                    conversation_id=conversation.conversation_id,
                    request_id=f"request-{number}",
                    content=f"question-{number}",
                )
                for number in range(8)
            )
        )

        messages = await repository.list_messages(
            context=context,
            conversation_id=conversation.conversation_id,
        )
        assert [message.sequence for message in messages] == list(range(1, 9))


@pytest.mark.asyncio
async def test_conversation_authorization_does_not_depend_on_message_tenant_columns(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)
        owner = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        conversation = await repository.create_conversation(
            context=owner,
            runtime_mode=LangGraphRuntimeMode.LINEAR,
        )
        message = await repository.persist_user_message(
            context=owner,
            conversation_id=conversation.conversation_id,
            request_id="request-1",
            content="private",
        )

        with pytest.raises(ConversationNotFound):
            await repository.get_message(
                context=TrustedRequestContext(
                    tenant_id="tenant-b", subject_id="subject-a"
                ),
                conversation_id=conversation.conversation_id,
                message_id=message.message_id,
            )
