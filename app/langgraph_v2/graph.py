"""Minimal typed LangGraph used by the v2 tracer."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph_v2.contracts import TracerQueryResponse, TracerStreamEvent


class TracerState(TypedDict):
    """Typed state for the ingress-to-finalization tracer graph."""

    query: str
    conversation_id: str
    client_request_id: str | None
    events: list[dict[str, Any]]


class TracerStateUpdate(TypedDict, total=False):
    """Partial state update returned by one tracer node."""

    events: list[dict[str, Any]]


async def _query(state: TracerState) -> TracerStateUpdate:
    query = state["query"]
    return {
        "events": [
            TracerStreamEvent(
                type="step_start",
                step="query",
                sequence=1,
            ).model_dump(exclude_none=True),
            TracerStreamEvent(
                type="step_completed",
                step="query",
                data={"query": query},
                sequence=2,
            ).model_dump(exclude_none=True),
        ]
    }


async def _finalize(state: TracerState) -> TracerStateUpdate:
    events = list(state["events"])
    response = TracerQueryResponse(
        query=state["query"],
        conversation_id=state["conversation_id"],
        metadata={"steps_executed": ["query", "finalization"]},
    )
    events.extend(
        [
            TracerStreamEvent(
                type="step_start",
                step="finalization",
                sequence=3,
            ).model_dump(exclude_none=True),
            TracerStreamEvent(
                type="step_completed",
                step="finalization",
                data={"status": "completed"},
                sequence=4,
            ).model_dump(exclude_none=True),
            TracerStreamEvent(
                type="done",
                data=response.model_dump(by_alias=True),
                sequence=5,
            ).model_dump(exclude_none=True),
        ]
    )
    return {"events": events}


def build_tracer_graph() -> CompiledStateGraph:
    """Compile the deterministic ingress-to-finalization LangGraph."""
    builder = StateGraph(TracerState)
    builder.add_node("query", _query)
    builder.add_node("finalization", _finalize)
    builder.add_edge(START, "query")
    builder.add_edge("query", "finalization")
    builder.add_edge("finalization", END)
    return builder.compile()


tracer_graph = build_tracer_graph()
