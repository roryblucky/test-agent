"""Shared authorized Artifact setup for graph integration tests."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.artifacts import ArtifactScope
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    TurnNotFound,
)


async def seed_artifact_scope(
    pool: AsyncConnectionPool[Any],
    *,
    turn_id: UUID | None = None,
    context: TrustedRequestContext | None = None,
) -> ArtifactScope:
    """Create the authorized Conversation and user Turn for an Artifact test."""
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
    return ArtifactScope(
        context=resolved_context,
        conversation_id="c1",
        turn_id=resolved_turn_id,
    )
