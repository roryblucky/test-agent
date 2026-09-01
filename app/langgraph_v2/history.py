"""Deterministic Conversation history selection for the v2 graph."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

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


class ConversationExchange(BaseModel):
    """One complete user/assistant exchange supplied to an LLM actor."""

    model_config = ConfigDict(frozen=True)

    user: str
    assistant: str


def to_model_message_history(
    exchanges: Sequence[ConversationExchange],
) -> list[ModelMessage]:
    """Convert durable complete exchanges to PydanticAI's native history format."""
    messages: list[ModelMessage] = []
    for exchange in exchanges:
        messages.append(ModelRequest(parts=[UserPromptPart(exchange.user)]))
        messages.append(ModelResponse(parts=[TextPart(exchange.assistant)]))
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


def estimate_exchange_tokens(exchange: ConversationExchange) -> int:
    """Estimate one complete exchange, including both message envelopes."""
    return (
        estimate_text_tokens(exchange.user)
        + estimate_text_tokens(exchange.assistant)
        + (2 * MESSAGE_FRAMING_TOKENS)
    )


def estimate_history_tokens(exchanges: Sequence[ConversationExchange]) -> int:
    """Return the deterministic budget consumed by complete exchanges."""
    return sum(estimate_exchange_tokens(exchange) for exchange in exchanges)


def select_sliding_window_history(
    messages: Sequence[MessageRecord],
    *,
    token_budget: int,
    current_request_id: str | None = None,
) -> list[ConversationExchange]:
    """Select the newest complete request pairs under the token budget."""
    if token_budget < 0:
        raise ValueError("token_budget must not be negative")

    messages_by_request: dict[str, dict[str, MessageRecord]] = defaultdict(dict)
    request_order: list[str] = []
    for message in sorted(messages, key=lambda item: item.sequence):
        if message.request_id == current_request_id:
            continue
        if message.request_id not in messages_by_request:
            request_order.append(message.request_id)
        messages_by_request[message.request_id][message.role] = message

    complete_exchanges = [
        ConversationExchange(
            user=messages_by_request[request_id]["user"].content,
            assistant=messages_by_request[request_id]["assistant"].content,
        )
        for request_id in request_order
        if messages_by_request[request_id].keys() >= {"user", "assistant"}
    ]

    selected: list[ConversationExchange] = []
    consumed = 0
    for exchange in reversed(complete_exchanges):
        exchange_tokens = estimate_exchange_tokens(exchange)
        if consumed + exchange_tokens > token_budget:
            break
        selected.append(exchange)
        consumed += exchange_tokens
    selected.reverse()
    return selected
