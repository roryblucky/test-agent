from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

from app.langgraph_v2.conversation_messages import MessageRecord
from app.langgraph_v2.history import (
    estimate_history_tokens,
    select_sliding_window_history,
)


def _message(
    turn_number: int,
    role: Literal["user", "assistant"],
    content: str,
    order: int,
) -> MessageRecord:
    return MessageRecord(
        tenant_id="tenant-a",
        message_id=UUID(int=order),
        conversation_id="conversation-1",
        turn_id=UUID(int=turn_number),
        role=role,
        content=content,
        idempotency_key=f"{turn_number}:{role}",
        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=order),
    )


def _turn(
    turn_number: int, user: str, assistant: str, order: int
) -> list[MessageRecord]:
    return [
        _message(turn_number, "user", user, order),
        _message(turn_number, "assistant", assistant, order + 1),
    ]


def test_empty_history_selects_no_turns() -> None:
    assert select_sliding_window_history([], token_budget=10) == []


def test_one_oversized_turn_is_excluded_whole() -> None:
    messages = _turn(1, "abc", "def", 1)

    assert select_sliding_window_history(messages, token_budget=9) == []


def test_turn_on_exact_budget_boundary_is_included() -> None:
    messages = _turn(1, "abc", "def", 1)

    assert [
        turn.model_dump()
        for turn in select_sliding_window_history(messages, token_budget=10)
    ] == [{"user": "abc", "assistant": "def"}]


def test_old_turns_are_evicted_without_splitting_recent_turns() -> None:
    messages = [
        *_turn(1, "u1", "a1", 1),
        *_turn(2, "u2", "a2", 3),
        *_turn(3, "u3", "a3", 5),
    ]

    selected = select_sliding_window_history(messages, token_budget=20)

    assert [turn.model_dump() for turn in selected] == [
        {"user": "u2", "assistant": "a2"},
        {"user": "u3", "assistant": "a3"},
    ]
    assert estimate_history_tokens(selected) == 20


def test_current_and_incomplete_turns_never_enter_history() -> None:
    current_turn_id = UUID(int=3)
    messages = [
        *_turn(1, "complete", "answer", 1),
        _message(2, "user", "failed input", 3),
        _message(3, "user", "current input", 4),
        _message(3, "assistant", "impossible retry output", 5),
    ]

    selected = select_sliding_window_history(
        messages,
        token_budget=100,
        current_turn_id=current_turn_id,
    )

    assert [turn.model_dump() for turn in selected] == [
        {"user": "complete", "assistant": "answer"}
    ]


def test_history_groups_user_and_assistant_messages_by_turn() -> None:
    messages = [
        _message(9, "user", "question", 1),
        _message(9, "assistant", "answer", 2),
    ]

    selected = select_sliding_window_history(messages, token_budget=100)

    assert [turn.model_dump() for turn in selected] == [
        {"user": "question", "assistant": "answer"}
    ]
