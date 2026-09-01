"""Project checkpointed Conversation Messages into bounded model context."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

DEFAULT_HISTORY_TOKEN_BUDGET = 8_000
MESSAGE_FRAMING_TOKENS = 4
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")
_CONVERSATION_MESSAGE_KEY = "conversation_message"
_REQUEST_ID_KEY = "request_id"


class RequestIdentityConflict(RuntimeError):
    """A request identity was reused for a different user query."""


class ConversationExchange(BaseModel):
    """One complete user/final-assistant exchange supplied to an LLM actor."""

    model_config = ConfigDict(frozen=True)

    user: str
    assistant: str


def conversation_message_id(request_id: str, role: Literal["user", "assistant"]) -> str:
    """Return the stable reducer identity for one logical Conversation Message."""
    return f"{request_id}:{role}"


def user_conversation_message(request_id: str, query: str) -> HumanMessage:
    """Create the only user Message shape admitted to Conversation state."""
    return HumanMessage(
        content=query,
        id=conversation_message_id(request_id, "user"),
        additional_kwargs={
            _CONVERSATION_MESSAGE_KEY: True,
            _REQUEST_ID_KEY: request_id,
        },
    )


def assistant_conversation_message(request_id: str, answer: str) -> AIMessage:
    """Create the final assistant Message admitted to Conversation state."""
    return AIMessage(
        content=answer,
        id=conversation_message_id(request_id, "assistant"),
        additional_kwargs={
            _CONVERSATION_MESSAGE_KEY: True,
            _REQUEST_ID_KEY: request_id,
        },
    )


def validate_request_identity(
    messages: Sequence[BaseMessage], *, request_id: str, query: str
) -> None:
    """Reject reuse of a request ID for a different logical query."""
    expected_id = conversation_message_id(request_id, "user")
    for message in messages:
        if message.id == expected_id:
            if message.text != query:
                raise RequestIdentityConflict(request_id)


def request_user_message_update(
    messages: Sequence[BaseMessage], *, request_id: str, query: str
) -> list[BaseMessage]:
    """Return a user Message only when the logical request is not present."""
    validate_request_identity(messages, request_id=request_id, query=query)
    expected_id = conversation_message_id(request_id, "user")
    if any(message.id == expected_id for message in messages):
        return []
    return [user_conversation_message(request_id, query)]


def to_model_message_history(
    exchanges: Sequence[ConversationExchange],
) -> list[ModelMessage]:
    """Convert complete exchanges to PydanticAI's native history format."""
    messages: list[ModelMessage] = []
    for exchange in exchanges:
        messages.append(ModelRequest(parts=[UserPromptPart(exchange.user)]))
        messages.append(ModelResponse(parts=[TextPart(exchange.assistant)]))
    return messages


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens deterministically without one model's tokenizer."""
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


def select_conversation_context(
    messages: Sequence[BaseMessage],
    *,
    token_budget: int,
    current_request_id: str | None = None,
) -> list[ConversationExchange]:
    """Select newest complete, application-owned request pairs under budget."""
    if token_budget < 0:
        raise ValueError("token_budget must not be negative")

    requests: dict[str, dict[str, str]] = {}
    request_order: list[str] = []
    for message in messages:
        metadata = cast(
            Mapping[str, object],
            message.additional_kwargs,  # pyright: ignore[reportUnknownMemberType]
        )
        if metadata.get(_CONVERSATION_MESSAGE_KEY) is not True:
            continue
        request_id = metadata.get(_REQUEST_ID_KEY)
        if not isinstance(request_id, str) or request_id == current_request_id:
            continue
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        if message.id != conversation_message_id(request_id, role):
            continue
        content = message.text
        if request_id not in requests:
            requests[request_id] = {}
            request_order.append(request_id)
        requests[request_id][role] = content

    complete = [
        ConversationExchange(
            user=requests[request_id]["user"],
            assistant=requests[request_id]["assistant"],
        )
        for request_id in request_order
        if requests[request_id].keys() >= {"user", "assistant"}
    ]
    selected: list[ConversationExchange] = []
    consumed = 0
    for exchange in reversed(complete):
        exchange_tokens = estimate_exchange_tokens(exchange)
        if consumed + exchange_tokens > token_budget:
            break
        selected.append(exchange)
        consumed += exchange_tokens
    selected.reverse()
    return selected
