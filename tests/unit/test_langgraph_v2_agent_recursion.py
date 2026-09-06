"""Agent recursion-limit safety coverage."""

from typing import TypedDict

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph  # pyright: ignore[reportMissingTypeStubs]

from app.langgraph_v2.agent_coordination import AGENT_RECURSION_LIMIT


class _LoopState(TypedDict):
    steps: int


@pytest.mark.asyncio
async def test_unexpected_agent_cycle_is_fatal_at_the_configured_limit() -> None:
    def loop(state: _LoopState) -> _LoopState:
        return {"steps": state["steps"] + 1}

    builder = StateGraph(_LoopState)
    builder.add_node("loop", loop)  # pyright: ignore[reportUnknownMemberType]
    builder.add_edge(START, "loop")  # pyright: ignore[reportUnknownMemberType]
    builder.add_edge("loop", "loop")  # pyright: ignore[reportUnknownMemberType]
    graph = builder.compile()  # pyright: ignore[reportUnknownMemberType]

    with pytest.raises(GraphRecursionError):
        await graph.ainvoke(  # pyright: ignore[reportUnknownMemberType]
            {"steps": 0}, {"recursion_limit": AGENT_RECURSION_LIMIT}
        )
