"""Shared authorized Turn setup for graph integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    TurnNotFound,
)


@dataclass(frozen=True)
class TurnScope:
    """Authorized Conversation Turn used by graph tests."""

    context: TrustedRequestContext
    conversation_id: str
    turn_id: UUID


async def seed_turn_scope(
    pool: AsyncConnectionPool[Any],
    *,
    turn_id: UUID | None = None,
    context: TrustedRequestContext | None = None,
) -> TurnScope:
    """Create one authorized Conversation and user Turn."""
    resolved_context = context or TrustedRequestContext(
        tenant_id="tenant-a", subject_id="subject-a"
    )
    resolved_turn_id = turn_id or uuid4()
    messages = ConversationMessageRepository(pool)
    await messages.resolve_conversation(
        context=resolved_context,
        conversation_id="c1",
    )
    try:
        await messages.get_turn(
            context=resolved_context,
            conversation_id="c1",
            turn_id=resolved_turn_id,
        )
    except TurnNotFound:
        await messages.create_turn(
            context=resolved_context,
            conversation_id="c1",
            turn_id=resolved_turn_id,
            content="question",
            idempotency_key=f"turn:{resolved_turn_id}:user",
        )
    return TurnScope(
        context=resolved_context,
        conversation_id="c1",
        turn_id=resolved_turn_id,
    )
