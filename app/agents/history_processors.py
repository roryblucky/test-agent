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

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
)


def trim_history(max_messages: int = 20) -> callable:
    """Return a processor that keeps only the last *max_messages* messages.

    Always preserves the first message (system prompt / initial query).
    Crucially, ensures that ToolCallPart and ToolReturnPart pairs are not
    orphaned or split during truncation, avoiding Pydantic-AI errors.
    """

    def _processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        if len(messages) <= max_messages:
            return messages

        # Always keep the first message
        kept = [messages[0]]

        # Start looking backwards to gather up to max_messages - 1
        # We process backwards and accumulate, then reverse at the end.
        tail = []
        i = len(messages) - 1

        while i > 0 and len(tail) < (max_messages - 1):
            msg = messages[i]
            tail.append(msg)

            # If this message contains a ToolReturnPart, we MUST also include
            # the corresponding ModelRequest that contained the ToolCallPart.
            # In pydantic-ai, typical flow: ModelRequest(ToolCall) -> ModelResponse(ToolReturn)
            import pydantic_ai.messages as p_msg

            if isinstance(msg, p_msg.ModelResponse) and any(
                isinstance(p, p_msg.ToolReturnPart) for p in msg.parts
            ):
                # We need the preceding ModelRequest that has the tool calls
                if i - 1 > 0 and isinstance(messages[i - 1], p_msg.ModelRequest):
                    i -= 1
                    tail.append(messages[i])

            i -= 1

        tail.reverse()
        return kept + tail

    return _processor


def filter_thinking() -> callable:
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
                    # Create a new ModelResponse with filtered parts
                    new_msg = ModelResponse(
                        parts=filtered_parts,
                        model_name=msg.model_name,
                        timestamp=msg.timestamp,
                    )
                    result.append(new_msg)
            else:
                result.append(msg)
        return result

    return _processor
