from __future__ import annotations

import pytest
from psycopg_pool import AsyncConnectionPool

from app.config.models import LangGraphRuntimeMode
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
)


@pytest.mark.asyncio
async def test_conversation_authorization_is_tenant_and_subject_scoped(
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
            request_id="authorization-request",
            content="hello",
        )

        assert await repository.get_message(
            context=owner,
            conversation_id=conversation.conversation_id,
            message_id=message.message_id,
        ) == message
        for unauthorized in (
            TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-b"),
            TrustedRequestContext(tenant_id="tenant-b", subject_id="subject-a"),
        ):
            with pytest.raises(ConversationNotFound):
                await repository.get_conversation(
                    context=unauthorized,
                    conversation_id=conversation.conversation_id,
                )
            with pytest.raises(ConversationNotFound):
                await repository.get_message(
                    context=unauthorized,
                    conversation_id=conversation.conversation_id,
                    message_id=message.message_id,
                )
            with pytest.raises(ConversationNotFound):
                await repository.list_messages(
                    context=unauthorized,
                    conversation_id=conversation.conversation_id,
                )
