from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.langgraph_v2.conversation_context import (
    RequestIdentityConflict,
    assistant_conversation_message,
    estimate_history_tokens,
    request_user_message_update,
    select_conversation_context,
    user_conversation_message,
)


def _exchange(request_number: int, user: str, assistant: str):
    request_id = f"request-{request_number}"
    return [
        user_conversation_message(request_id, user),
        assistant_conversation_message(request_id, assistant),
    ]


def test_empty_checkpoint_selects_no_exchanges() -> None:
    assert select_conversation_context([], token_budget=10) == []


def test_one_oversized_exchange_is_excluded_whole() -> None:
    assert select_conversation_context(
        _exchange(1, "abc", "def"), token_budget=9
    ) == []


def test_newest_complete_exchanges_are_selected_without_splitting() -> None:
    messages = [
        *_exchange(1, "u1", "a1"),
        *_exchange(2, "u2", "a2"),
        *_exchange(3, "u3", "a3"),
    ]

    selected = select_conversation_context(messages, token_budget=20)

    assert [exchange.model_dump() for exchange in selected] == [
        {"user": "u2", "assistant": "a2"},
        {"user": "u3", "assistant": "a3"},
    ]
    assert estimate_history_tokens(selected) == 20


def test_current_incomplete_and_internal_messages_never_enter_context() -> None:
    messages = [
        *_exchange(1, "complete", "answer"),
        user_conversation_message("request-2", "failed input"),
        *_exchange(3, "current input", "old retry output"),
        HumanMessage(content="planner prompt"),
        AIMessage(content="planner output"),
        ToolMessage(content="tool output", tool_call_id="tool-1"),
    ]

    selected = select_conversation_context(
        messages,
        token_budget=100,
        current_request_id="request-3",
    )

    assert [exchange.model_dump() for exchange in selected] == [
        {"user": "complete", "assistant": "answer"}
    ]


def test_same_request_and_query_is_not_duplicated() -> None:
    existing = [user_conversation_message("request-1", "question")]

    assert request_user_message_update(
        existing, request_id="request-1", query="question"
    ) == []


def test_same_request_with_different_query_conflicts() -> None:
    existing = [user_conversation_message("request-1", "question")]

    with pytest.raises(RequestIdentityConflict):
        request_user_message_update(
            existing, request_id="request-1", query="different"
        )
