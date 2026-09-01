"""Shared authorized request setup for graph integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from psycopg_pool import AsyncConnectionPool

from app.config.models import LangGraphRuntimeMode
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversations import ConversationRepository


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
    """Create one authorized Conversation request scope."""
    resolved_context = context or TrustedRequestContext(
        tenant_id="tenant-a", subject_id="subject-a"
    )
    resolved_request_id = str(request_id or uuid4())
    conversations = ConversationRepository(pool)
    conversation = await conversations.create_conversation(
        context=resolved_context,
        runtime_mode=LangGraphRuntimeMode.LINEAR,
    )
    return RequestScope(
        context=resolved_context,
        conversation_id=conversation.conversation_id,
        request_id=resolved_request_id,
    )
