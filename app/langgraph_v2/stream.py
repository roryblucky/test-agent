"""Request-owned LangGraph streaming projected onto the v2 SSE contract."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol, cast

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from app.langgraph_v2.contracts import LiveStreamEvent

_LOGGER = logging.getLogger(__name__)

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


class RequestOwnedGraph(Protocol):
    """Minimal LangGraph stream interface owned by one receiving request."""

    def astream(
        self,
        graph_input: Any | None,
        /,
        *,
        config: RunnableConfig | None = None,
        stream_mode: list[str] | str | None = None,
        durability: str | None = None,
    ) -> AsyncIterator[Any]:
        """Return the asynchronous graph iterator for this request."""
        ...


class GraphStreamCleanupError(RuntimeError):
    """The underlying LangGraph iterator failed while being closed."""


async def _close_graph_iterator(
    graph_iterator: Any,
) -> tuple[bool, BaseException | None]:
    """Await graph cleanup despite repeated cancellation signals."""
    close = getattr(graph_iterator, "aclose", None)
    if close is None:
        return False, None
    close_task = asyncio.ensure_future(close())
    cancelled = False
    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            # Retrieve the task's exception below while retaining ownership of
            # the cleanup task until it has reached a terminal state.
            break
    cleanup_error: BaseException | None = None
    try:
        await close_task
    except asyncio.CancelledError as error:
        cleanup_error = error
    except BaseException as error:
        cleanup_error = GraphStreamCleanupError(str(error))
    return cancelled, cleanup_error


async def stream_graph(
    graph: RequestOwnedGraph,
    graph_input: Any | None,
    *,
    config: RunnableConfig | None = None,
    terminal_sink: Callable[[LiveStreamEvent], Awaitable[None]] | None = None,
) -> AsyncGenerator[str]:
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
    pending_terminal: LiveStreamEvent | None = None
    pending_terminal_update_seen = False
    deferred_frames: list[str] = []

    primary_error: BaseException | None = None
    try:
        async for stream_part in graph_iterator:
            mode, data = _stream_part(stream_part)
            if mode not in _STREAM_MODES:
                continue

            if mode == "updates":
                if pending_terminal is not None:
                    pending_terminal_update_seen = True
                continue

            for candidate in _event_mappings(data):
                event = LiveStreamEvent.model_validate(candidate)
                if event.checkpoint_terminal:
                    pending_terminal = event
                    deferred_frames.append(event.to_sse())
                    continue
                frame = event.to_sse()
                if pending_terminal is None:
                    yield frame
                else:
                    deferred_frames.append(frame)

        if pending_terminal is not None:
            if not pending_terminal_update_seen:
                raise RuntimeError(
                    "checkpoint terminal event was not confirmed by graph updates"
                )
            if terminal_sink is not None:
                await terminal_sink(pending_terminal)
            for frame in deferred_frames:
                yield frame
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_cancelled, cleanup_error = await _close_graph_iterator(graph_iterator)
        if cleanup_error is not None and (
            primary_error is not None or cleanup_cancelled
        ):
            _LOGGER.warning(
                "Request-owned graph cleanup failed after a primary exception",
                exc_info=(
                    type(cleanup_error),
                    cleanup_error,
                    cleanup_error.__traceback__,
                ),
            )
        if primary_error is None:
            if cleanup_cancelled:
                raise asyncio.CancelledError
            if cleanup_error is not None:
                raise cleanup_error


def _stream_part(part: object) -> tuple[str | None, object]:
    """Read LangGraph's pinned multiple-mode tuple shape."""
    if isinstance(part, tuple):
        tuple_part = cast(tuple[object, ...], part)
        if len(tuple_part) == 2 and isinstance(tuple_part[0], str):
            return tuple_part[0], tuple_part[1]
        return None, tuple_part
    return None, part


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    if isinstance(value, BaseModel):
        return value.model_dump()
    return None


def _event_mappings(value: object) -> list[Mapping[str, Any]]:
    """Extract event-shaped values without exposing arbitrary graph state."""
    mapping = _as_mapping(value)
    if mapping is not None:
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
        for item in cast(list[object] | tuple[object, ...], value):
            events.extend(_event_mappings(item))
        return events
    return []
