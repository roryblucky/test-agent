"""Deterministic Conversation history selection for the v2 graph."""

from __future__ import annotations

import re
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
MESSAGE_FRAMING_TOKENS = 4
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


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


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens deterministically without depending on one model tokenizer.

    ASCII words use one token per four characters, while every non-ASCII or
    punctuation code point counts as one token. Whitespace is represented by
    the surrounding message framing rather than counted independently.
    """
    total = 0
    for token in _TOKEN_PATTERN.findall(text):
        if token.isascii() and token.isalnum():
            total += max(1, (len(token) + 3) // 4)
        else:
            total += len(token)
    return total


def estimate_turn_tokens(turn: ConversationTurn) -> int:
    """Estimate one complete turn, including both message envelopes."""
    return (
        estimate_text_tokens(turn.user)
        + estimate_text_tokens(turn.assistant)
        + (2 * MESSAGE_FRAMING_TOKENS)
    )


def estimate_history_tokens(turns: Sequence[ConversationTurn]) -> int:
    """Return the deterministic budget consumed by complete turns."""
    return sum(estimate_turn_tokens(turn) for turn in turns)


def select_sliding_window_history(
    messages: Sequence[MessageRecord],
    *,
    token_budget: int,
    current_turn_id: UUID | None = None,
) -> list[ConversationTurn]:
    """Select the newest complete turns under the deterministic token budget."""
    if token_budget < 0:
        raise ValueError("token_budget must not be negative")

    messages_by_turn: dict[UUID, dict[str, MessageRecord]] = defaultdict(dict)
    turn_order: list[UUID] = []
    for message in sorted(
        messages, key=lambda item: (item.created_at, item.message_id)
    ):
        if message.turn_id == current_turn_id:
            continue
        if message.turn_id not in messages_by_turn:
            turn_order.append(message.turn_id)
        messages_by_turn[message.turn_id][message.role] = message

    complete_turns = [
        ConversationTurn(
            user=messages_by_turn[turn_id]["user"].content,
            assistant=messages_by_turn[turn_id]["assistant"].content,
        )
        for turn_id in turn_order
        if messages_by_turn[turn_id].keys() >= {"user", "assistant"}
    ]

    selected: list[ConversationTurn] = []
    consumed = 0
    for turn in reversed(complete_turns):
        turn_tokens = estimate_turn_tokens(turn)
        if consumed + turn_tokens > token_budget:
            break
        selected.append(turn)
        consumed += turn_tokens
    selected.reverse()
    return selected
