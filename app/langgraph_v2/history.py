"""Deterministic Conversation history selection for the v2 graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from app.langgraph_v2.conversation_messages import MessageRecord

DEFAULT_HISTORY_TOKEN_BUDGET = 8_000


class ConversationTurn(BaseModel):
    """One complete user/assistant exchange supplied to an LLM actor."""

    model_config = ConfigDict(frozen=True)

    user: str
    assistant: str


def to_model_message_history(
    turns: Sequence[ConversationTurn],
) -> list[ModelMessage]:
    """Convert durable complete turns to PydanticAI's native history format."""
    messages: list[ModelMessage] = []
    for turn in turns:
        messages.append(ModelRequest(parts=[UserPromptPart(turn.user)]))
        messages.append(ModelResponse(parts=[TextPart(turn.assistant)]))
    return messages


def select_sliding_window_history(
    messages: Sequence[MessageRecord],
    *,
    token_budget: int,
    current_run_id: UUID | None = None,
) -> list[ConversationTurn]:
    """Select the newest complete turns under a conservative UTF-8 budget."""
    if token_budget < 0:
        raise ValueError("token_budget must not be negative")

    messages_by_run: dict[UUID, dict[str, MessageRecord]] = defaultdict(dict)
    run_order: list[UUID] = []
    for message in sorted(
        messages, key=lambda item: (item.created_at, item.message_id)
    ):
        if message.run_id == current_run_id:
            continue
        if message.run_id not in messages_by_run:
            run_order.append(message.run_id)
        messages_by_run[message.run_id][message.role] = message

    complete_turns = [
        ConversationTurn(
            user=messages_by_run[run_id]["user"].content,
            assistant=messages_by_run[run_id]["assistant"].content,
        )
        for run_id in run_order
        if messages_by_run[run_id].keys() >= {"user", "assistant"}
    ]

    selected: list[ConversationTurn] = []
    consumed = 0
    for turn in reversed(complete_turns):
        turn_tokens = len(turn.user.encode()) + len(turn.assistant.encode())
        if consumed + turn_tokens > token_budget:
            break
        selected.append(turn)
        consumed += turn_tokens
    selected.reverse()
    return selected
