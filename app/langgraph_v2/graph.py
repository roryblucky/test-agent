"""Minimal typed LangGraph used by the v2 tracer."""

from __future__ import annotations

import unicodedata
from typing import Any, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph_v2.contracts import TracerQueryResponse, TracerStreamEvent
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultInput
from app.langgraph_v2.run_events import EventInput, EventRecord


class TracerState(TypedDict):
    """Typed state for the ingress-to-finalization tracer graph."""

    query: str
    conversation_id: str
    client_request_id: str | None
    events: list[dict[str, Any]]


class TracerStateUpdate(TypedDict, total=False):
    """Partial state update returned by one tracer node."""

    events: list[dict[str, Any]]


async def _query(
    state: TracerState,
    *,
    phase_context: PhaseExecutionContext | None = None,
) -> TracerStateUpdate:
    query = state["query"]
    canonical = canonical_query(query)

    async def invoke() -> PhaseResultInput:
        return PhaseResultInput(
            phase_name="query",
            normalized_result={"query": canonical, "history_snapshot": []},
            events=(
                EventInput(
                    event_key="phase:query:step_start:1",
                    type="step_start",
                    step="query",
                ),
                EventInput(
                    event_key="phase:query:step_completed:1",
                    type="step_completed",
                    step="query",
                    data={"query": canonical},
                ),
            ),
        )

    if phase_context is None:
        phase = await invoke()
        return {
            "events": [
                _event_state(event, index)
                for index, event in enumerate(phase.events, 1)
            ]
        }
    result = await phase_context.repository.get_or_invoke(
        tenant_id=phase_context.tenant_id,
        run_id=phase_context.run_id,
        owner_instance_id=phase_context.owner_instance_id,
        execution_epoch=phase_context.execution_epoch,
        phase_name="query",
        invoke=invoke,
    )
    return {"events": [_event_state(event, event.sequence) for event in result.events]}


def _event_state(event: EventInput | EventRecord, sequence: int) -> dict[str, Any]:
    """Convert journal or in-memory event data into graph state."""
    return TracerStreamEvent(
        event_key=event.event_key,
        type=cast(Any, event.type),
        step=event.step,
        data=event.data,
        sequence=sequence,
    ).model_dump(exclude_none=True)


def canonical_query(query: str) -> str:
    """Normalize query text without changing internal whitespace."""
    return unicodedata.normalize("NFC", query.replace("\r\n", "\n")).strip()


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
                event_key="phase:finalization:step_start:1",
                type="step_start",
                step="finalization",
                sequence=3,
            ).model_dump(exclude_none=True),
            TracerStreamEvent(
                event_key="phase:finalization:step_completed:1",
                type="step_completed",
                step="finalization",
                data={"status": "completed"},
                sequence=4,
            ).model_dump(exclude_none=True),
            TracerStreamEvent(
                event_key="lifecycle:completed:0",
                type="done",
                data=response.model_dump(by_alias=True),
                sequence=5,
            ).model_dump(exclude_none=True),
        ]
    )
    return {"events": events}


def build_tracer_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    phase_context: PhaseExecutionContext | None = None,
) -> CompiledStateGraph:
    """Compile the deterministic ingress-to-finalization LangGraph."""
    builder = StateGraph(TracerState)

    async def query_node(state: TracerState) -> TracerStateUpdate:
        return await _query(state, phase_context=phase_context)

    builder.add_node("query", query_node)
    builder.add_node("finalization", _finalize)
    builder.add_edge(START, "query")
    builder.add_edge("query", "finalization")
    builder.add_edge("finalization", END)
    return builder.compile(checkpointer=checkpointer)


tracer_graph = build_tracer_graph()
