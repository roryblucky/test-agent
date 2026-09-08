"""History ownership and atomic legacy trimming coverage."""

from typing import Any, cast

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from app.agents.history_processors import trim_history
from app.agents.query_understanding import create_query_understanding_agent
from app.core.model_registry import ModelRegistry


class _Registry:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def create_agent(self, model_name: str, **kwargs: Any) -> object:
        assert model_name == "intent"
        self.kwargs = kwargs
        return object()


def test_query_understanding_factory_does_not_own_a_second_history_budget() -> None:
    registry = _Registry()

    create_query_understanding_agent(cast(ModelRegistry, registry))

    assert registry.kwargs is not None
    assert "history_processors" not in registry.kwargs


def test_trim_history_preserves_complete_ordinary_turns() -> None:
    messages = [
        message
        for index in range(12)
        for message in (
            ModelRequest(parts=[UserPromptPart(f"u{index}")]),
            ModelResponse(parts=[TextPart(f"a{index}")]),
        )
    ]

    selected = trim_history(20)(messages)

    contents: list[str] = []
    for message in selected:
        part = message.parts[0]
        assert isinstance(part, (UserPromptPart, TextPart))
        assert isinstance(part.content, str)
        contents.append(part.content)
    assert contents == [
        "u0",
        "a0",
        *[value for index in range(3, 12) for value in (f"u{index}", f"a{index}")],
    ]


def test_trim_history_keeps_tool_call_return_chain_atomic() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(parts=[TextPart("first answer")]),
        ModelRequest(parts=[UserPromptPart("research")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="lookup",
                    args={"query": "AAPL"},
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="lookup",
                    content={"price": 100},
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart("research answer")]),
    ]

    selected = trim_history(4)(messages)

    assert selected == messages
