from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from app.langgraph_v2.conversation_messages import MessageRecord
from app.langgraph_v2.history import (
    estimate_history_tokens,
    select_sliding_window_history,
)


def _message(
    request_number: int,
    role: Literal["user", "assistant"],
    content: str,
    sequence: int,
) -> MessageRecord:
    return MessageRecord(
        message_id=UUID(int=sequence),
        conversation_id=UUID(int=1),
        request_id=f"request-{request_number}",
        sequence=sequence,
        role=role,
        content=content,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _exchange(
    request_number: int, user: str, assistant: str, sequence: int
) -> list[MessageRecord]:
    return [
        _message(request_number, "user", user, sequence),
        _message(request_number, "assistant", assistant, sequence + 1),
    ]


def test_empty_history_selects_no_exchanges() -> None:
    assert select_sliding_window_history([], token_budget=10) == []


def test_one_oversized_exchange_is_excluded_whole() -> None:
    assert select_sliding_window_history(
        _exchange(1, "abc", "def", 1), token_budget=9
    ) == []


def test_newest_complete_exchanges_are_selected_without_splitting() -> None:
    messages = [
        *_exchange(1, "u1", "a1", 1),
        *_exchange(2, "u2", "a2", 3),
        *_exchange(3, "u3", "a3", 5),
    ]

    selected = select_sliding_window_history(messages, token_budget=20)

    assert [exchange.model_dump() for exchange in selected] == [
        {"user": "u2", "assistant": "a2"},
        {"user": "u3", "assistant": "a3"},
    ]
    assert estimate_history_tokens(selected) == 20


def test_current_and_incomplete_requests_never_enter_history() -> None:
    messages = [
        *_exchange(1, "complete", "answer", 1),
        _message(2, "user", "failed input", 3),
        *_exchange(3, "current input", "impossible retry output", 4),
    ]

    selected = select_sliding_window_history(
        messages,
        token_budget=100,
        current_request_id="request-3",
    )

    assert [exchange.model_dump() for exchange in selected] == [
        {"user": "complete", "assistant": "answer"}
    ]
