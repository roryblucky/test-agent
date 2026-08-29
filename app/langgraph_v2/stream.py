"""Request-owned LangGraph streaming projected onto the v2 SSE contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.langgraph_v2.contracts import TracerStreamEvent

_STREAM_MODES = ["updates", "custom"]
_EVENT_TYPES = {
    "step_start",
    "step_completed",
    "token",
    "citations",
    "error",
    "done",
    "stopped",
}


async def stream_graph(
    graph: Any,
    graph_input: Any | None,
    *,
    config: RunnableConfig | None = None,
) -> AsyncIterator[str]:
    """Yield one legacy-compatible SSE frame for each public graph update.

    ``graph_input`` may be a new graph state or ``None`` when LangGraph should
    load the state associated with ``config``'s checkpoint thread.  The graph
    iterator belongs to this request: closing or cancelling this iterator
    closes the underlying LangGraph iterator before control returns to the
    caller.
    """
    graph_iterator = graph.astream(
        graph_input,
        config=config,
        stream_mode=_STREAM_MODES,
        durability="sync",
    )
    seen_event_keys: set[str] = set()
    next_sequence = 0

    try:
        async for stream_part in graph_iterator:
            mode, data = _stream_part(stream_part)
            if mode not in _STREAM_MODES:
                continue

            candidates = _event_mappings(data)

            for candidate in candidates:
                event_key = candidate.get("event_key")
                if isinstance(event_key, str) and event_key in seen_event_keys:
                    continue
                event, next_sequence = _event_from_mapping(
                    candidate,
                    next_sequence=next_sequence,
                    mode=mode,
                )
                seen_event_keys.add(event.event_key)
                yield event.to_sse()
    finally:
        close = getattr(graph_iterator, "aclose", None)
        if close is not None:
            close_task = asyncio.ensure_future(close())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                # Cancellation of the graph must not leave its cleanup half-done.
                await asyncio.shield(close_task)
                raise


def _stream_part(part: Any) -> tuple[str | None, Any]:
    """Read LangGraph's pinned multiple-mode tuple shape."""
    if isinstance(part, tuple) and len(part) == 2 and isinstance(part[0], str):
        return part[0], part[1]
    return None, part


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if dump is None:
        return None
    dumped = dump()
    return dumped if isinstance(dumped, Mapping) else None


def _event_mappings(value: Any) -> list[Mapping[str, Any]]:
    """Extract event-shaped values without exposing arbitrary graph state."""
    mapping = _as_mapping(value)
    if mapping is not None:
        if isinstance(mapping.get("events"), (list, tuple)):
            return _event_mappings(mapping["events"])
        nested_event = mapping.get("event")
        if nested_event is not None:
            return _event_mappings(nested_event)
        if isinstance(mapping.get("type"), str) and mapping["type"] in _EVENT_TYPES:
            return [mapping]
        if len(mapping) == 1:
            return _event_mappings(next(iter(mapping.values())))
        return []
    if isinstance(value, (list, tuple)):
        events: list[Mapping[str, Any]] = []
        for item in value:
            events.extend(_event_mappings(item))
        return events
    return []


def _event_from_mapping(
    mapping: Mapping[str, Any],
    *,
    next_sequence: int,
    mode: str,
) -> tuple[TracerStreamEvent, int]:
    # Graph state updates may carry node-local journal sequences (and repeated
    # snapshots), so the SSE projection owns one contiguous public sequence.
    sequence = next_sequence + 1

    event_key = mapping.get("event_key")
    if not isinstance(event_key, str) or not event_key:
        event_key = f"stream:{mode}:{sequence}"
    event = TracerStreamEvent(
        event_key=event_key,
        type=mapping["type"],
        sequence=sequence,
        step=mapping.get("step"),
        data=mapping.get("data"),
    )
    return event, sequence
