"""Built-in history processors for the coordinator agent.

History processors are callables ``(list[ModelMessage]) -> list[ModelMessage]``
that transform the message history before it is sent to the model.

They are passed to pydantic-ai's ``Agent(history_processors=[...])`` and
run in order on every model call.

Common use-cases:
- Message compression  (limit context window usage)
- Conversation summarisation
- System prompt injection
- Message filtering / redaction
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

HistoryProcessor = Callable[[list[ModelMessage]], list[ModelMessage]]


def trim_history(max_messages: int = 20) -> HistoryProcessor:
    """Return a processor that targets the last *max_messages* messages.

    Always preserves the complete first turn and never splits an ordinary turn
    or Tool call/return chain. An atomic turn may make the result exceed the
    target rather than leave an orphaned protocol message.
    """

    def _processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        if len(messages) <= max_messages:
            return messages

        groups: list[list[ModelMessage]] = []
        for message in messages:
            starts_group = isinstance(message, ModelRequest) and not (
                groups
                and isinstance(groups[-1][-1], ModelResponse)
                and _is_tool_exchange(groups[-1][-1], message)
            )
            if not groups or starts_group:
                groups.append([])
            groups[-1].append(message)

        kept_groups = [groups[0]]
        kept_count = len(groups[0])
        tail_groups: list[list[ModelMessage]] = []
        for group in reversed(groups[1:]):
            if (
                tail_groups
                and kept_count
                + sum(len(item) for item in tail_groups)
                + len(group)
                > max_messages
            ):
                break
            tail_groups.append(group)
        kept_groups.extend(reversed(tail_groups))
        return [message for group in kept_groups for message in group]

    return _processor


def _is_tool_exchange(response: ModelResponse, request: ModelRequest) -> bool:
    """Return whether adjacent messages are a Tool call and its return."""
    calls = {
        part.tool_call_id
        for part in response.parts
        if isinstance(part, ToolCallPart)
    }
    returns = {
        part.tool_call_id
        for part in request.parts
        if isinstance(part, ToolReturnPart)
    }
    return bool(calls & returns)


def filter_thinking() -> HistoryProcessor:
    """Return a processor that removes ThinkingPart from history.

    Useful to reduce token usage — thinking traces are often large
    and not needed for subsequent turns.
    """
    from pydantic_ai.messages import ThinkingPart

    def _processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        result: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelResponse):
                filtered_parts = [
                    p for p in msg.parts if not isinstance(p, ThinkingPart)
                ]
                if filtered_parts:
                    result.append(replace(msg, parts=filtered_parts))
            else:
                result.append(msg)
        return result

    return _processor
