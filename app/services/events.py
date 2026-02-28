"""SSE (Server-Sent Events) protocol models and event emitter.

Defines a structured event protocol for streaming pipeline progress
and results to the client.  Every flow step emits ``step_start`` and
``step_completed`` events, and LLM steps additionally emit ``token``
events.

Event types
-----------

.. list-table::
   :header-rows: 1

   * - type
     - description
   * - ``step_start``
     - A pipeline step is beginning.
   * - ``step_completed``
     - A pipeline step has finished, includes step result payload.
   * - ``token``
     - A single LLM streaming token.
   * - ``done``
     - The entire pipeline has finished, includes final result.
   * - ``error``
     - An error occurred, pipeline terminated.

This protocol is consistent with industry patterns
(LangGraph streaming events, OpenAI Assistants streaming,
Vercel AI SDK data stream protocol).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    """SSE event types."""

    STEP_START = "step_start"
    STEP_COMPLETED = "step_completed"
    TOKEN = "token"
    THINKING = "thinking"
    DONE = "done"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A single SSE event."""

    type: EventType
    step: str | None = None
    data: Any = None

    def to_sse(self) -> str:
        r"""Serialise to SSE wire format (``data: ...\\n\\n``)."""
        payload = {"type": self.type.value}
        if self.step is not None:
            payload["step"] = self.step
        if self.data is not None:
            payload["data"] = self.data
        return f"data: {json.dumps(payload, default=str)}\n\n"


class EventEmitter:
    """Async event emitter that bridges pipeline execution with SSE output.

    The pipeline pushes events via :meth:`emit` / :meth:`emit_token`.
    The API layer consumes events via ``async for event in emitter``.

    Usage in flow engine::

        emitter = EventEmitter()
        await emitter.emit_step_start("retriever")
        ...
        await emitter.emit_step_completed("retriever", {"documents": [...]})

    Usage in API layer::

        async for sse_line in emitter:
            yield sse_line  # already formatted as SSE
    """

    def __init__(self, maxsize: int = 2000) -> None:
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    # ------------------------------------------------------------------
    # Producer API (called by flow engine / orchestrator)
    # ------------------------------------------------------------------

    async def emit(self, event: StreamEvent) -> None:
        """Push a single event.

        If the queue is full (slow client), this will block until space is available.
        If blocking takes too long, we might want to drop events or error out,
        but for now we rely on standard asyncio backpressure.
        """
        if self._closed:
            return

        try:
            # wait for space if queue is full
            await self._queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Fallback for unexpected queue errors
            pass

    async def emit_step_start(self, step_name: str) -> None:
        """Convenience: emit a ``step_start`` event."""
        await self.emit(StreamEvent(type=EventType.STEP_START, step=step_name))

    async def emit_step_completed(self, step_name: str, result: Any = None) -> None:
        """Convenience: emit a ``step_completed`` event with result payload."""
        await self.emit(
            StreamEvent(type=EventType.STEP_COMPLETED, step=step_name, data=result)
        )

    async def emit_token(self, token: str) -> None:
        """Convenience: emit a single LLM streaming token."""
        await self.emit(StreamEvent(type=EventType.TOKEN, data=token))

    async def emit_thinking(self, content: str) -> None:
        """Convenience: emit a thinking / reasoning step."""
        await self.emit(StreamEvent(type=EventType.THINKING, data=content))

    async def emit_done(self, data: Any = None) -> None:
        """Convenience: emit the final ``done`` event and close."""
        await self.emit(StreamEvent(type=EventType.DONE, data=data))
        await self.close()

    async def emit_error(self, error: str) -> None:
        """Convenience: emit an ``error`` event and close."""
        await self.emit(StreamEvent(type=EventType.ERROR, data=error))
        await self.close()

    async def close(self) -> None:
        """Signal that no more events will be emitted."""
        if self._closed:
            return

        try:
            # We must put the sentinel BEFORE setting _closed=True
            # Otherwise our own emit/put logic might be bypassed if
            # we rely on _closed in other places (though we use raw _queue.put here)
            await self._queue.put(None)  # sentinel
        except asyncio.QueueFull:
            # If queue is full during close, force space or ignore
            pass
        finally:
            self._closed = True

    # ------------------------------------------------------------------
    # Consumer API (used by API layer)
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter_sse()

    async def _iter_sse(self) -> AsyncIterator[str]:
        """Yield SSE-formatted strings until closed."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event.to_sse()
