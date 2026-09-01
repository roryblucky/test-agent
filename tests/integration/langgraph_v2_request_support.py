"""Shared authorized request setup for graph integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from psycopg_pool import AsyncConnectionPool

from app.config.models import LangGraphRuntimeMode
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import ConversationMessageRepository


@dataclass(frozen=True)
class RequestScope:
    """Authorized Conversation request used by graph tests."""

    context: TrustedRequestContext
    conversation_id: UUID
    request_id: str


async def seed_request_scope(
    pool: AsyncConnectionPool[Any],
    *,
    request_id: UUID | str | None = None,
    context: TrustedRequestContext | None = None,
) -> RequestScope:
    """Create one authorized Conversation and user request Message."""
    resolved_context = context or TrustedRequestContext(
        tenant_id="tenant-a", subject_id="subject-a"
    )
    resolved_request_id = str(request_id or uuid4())
    messages = ConversationMessageRepository(pool)
    conversation = await messages.create_conversation(
        context=resolved_context,
        runtime_mode=LangGraphRuntimeMode.LINEAR,
    )
    await messages.persist_user_message(
        context=resolved_context,
        conversation_id=conversation.conversation_id,
        request_id=resolved_request_id,
        content="question",
    )
    return RequestScope(
        context=resolved_context,
        conversation_id=conversation.conversation_id,
        request_id=resolved_request_id,
    )
