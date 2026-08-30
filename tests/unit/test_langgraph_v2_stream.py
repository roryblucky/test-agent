import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.langgraph_v2.contracts import LiveStreamEvent
from app.langgraph_v2.stream import stream_graph


def _payload(frame: str) -> dict[str, Any]:
    return json.loads(frame.removeprefix("data: ").strip())


class _FakeGraphStream:
    def __init__(self, parts: list[Any]) -> None:
        self._parts = parts
        self.closed = False
        self.close_completed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if not self._parts:
            raise StopAsyncIteration
        part = self._parts.pop(0)
        if isinstance(part, asyncio.Event):
            await part.wait()
        return part

    async def aclose(self) -> None:
        self.closed = True
        await asyncio.sleep(0)
        self.close_completed = True


class _FakeGraph:
    def __init__(self, stream: AsyncIterator[Any]) -> None:
        self.stream = stream
        self.inputs: list[Any] = []
        self.options: list[dict[str, Any]] = []

    def astream(self, graph_input: Any, **options: Any) -> AsyncIterator[Any]:
        self.inputs.append(graph_input)
        self.options.append(options)
        return self.stream


def test_checkpoint_terminal_marker_is_internal_to_the_custom_stream() -> None:
    event = LiveStreamEvent(
        type="done",
        data={"answer": "complete"},
        checkpoint_terminal=True,
    )

    assert event.to_stream_payload()["checkpoint_terminal"] is True
    assert "checkpoint_terminal" not in _payload(event.to_sse())


@pytest.mark.asyncio
async def test_stream_graph_preserves_terminal_marker_from_event_model() -> None:
    terminal_events: list[LiveStreamEvent] = []
    graph = _FakeGraph(
        _FakeGraphStream(
            [
                (
                    "custom",
                    LiveStreamEvent(
                        type="done",
                        data={"answer": "complete"},
                        checkpoint_terminal=True,
                    ),
                ),
                ("updates", {"finalization": {"final_response": {}}}),
            ]
        )
    )

    async def terminal_sink(event: LiveStreamEvent) -> None:
        terminal_events.append(event)

    frames = [
        frame
        async for frame in stream_graph(
            graph,
            {"query": "hello"},
            terminal_sink=terminal_sink,
        )
    ]

    assert [event.type for event in terminal_events] == ["done"]
    assert [_payload(frame) for frame in frames] == [
        {"type": "done", "data": {"answer": "complete"}}
    ]


@pytest.mark.asyncio
async def test_stream_graph_translates_approved_modes_and_ignores_diagnostics() -> None:
    stream = _FakeGraphStream(
        [
            (
                "updates",
                {
                    "answer": {
                        "events": [
                            {
                                "event_key": "phase:answer:step_start:1",
                                "type": "step_start",
                                "step": "answer",
                                "sequence": 17,
                            }
                        ]
                    }
                },
            ),
            (
                "custom",
                {
                    "event_key": "phase:answer:token:1",
                    "type": "token",
                    "data": "hello",
                },
            ),
            (
                "custom",
                {
                    "type": "progress",
                    "step": "retriever",
                    "data": {"completed": 1, "total": 3},
                },
            ),
            ("custom", "unknown custom text"),
            (
                "messages",
                (" world", {"langgraph_node": "answer"}),
            ),
            (
                "messages",
                ("private reasoning", {"langgraph_node": "tool"}),
            ),
            {
                "type": "updates",
                "data": {
                    "private": {
                        "event_key": "phase:private:step_start:1",
                        "type": "step_start",
                    }
                },
            },
            ("checkpoints", {"state": "private"}),
            ("debug", {"state": "private"}),
        ]
    )
    graph = _FakeGraph(stream)

    frames = [frame async for frame in stream_graph(graph, {"query": "hello"})]

    assert [_payload(frame) for frame in frames] == [
        {"type": "token", "data": "hello"},
        {
            "type": "progress",
            "step": "retriever",
            "data": {"completed": 1, "total": 3},
        },
    ]
    assert graph.inputs == [{"query": "hello"}]
    assert graph.options == [
        {
            "config": None,
            "stream_mode": ["updates", "custom"],
            "durability": "sync",
        }
    ]
    assert stream.closed is True
    assert stream.close_completed is True


@pytest.mark.asyncio
async def test_stream_graph_accepts_none_as_a_checkpoint_resume_input() -> None:
    graph = _FakeGraph(
        _FakeGraphStream(
            [
                (
                    "custom",
                    {
                        "type": "step_completed",
                        "data": {"query": "continued"},
                    },
                )
            ]
        )
    )

    frames = [
        frame
        async for frame in stream_graph(
            graph,
            None,
            config={"configurable": {"thread_id": "thread-1"}},
        )
    ]

    assert _payload(frames[0]) == {
        "type": "step_completed",
        "data": {"query": "continued"},
    }
    assert graph.inputs == [None]
    assert graph.options == [
        {
            "config": {"configurable": {"thread_id": "thread-1"}},
            "stream_mode": ["updates", "custom"],
            "durability": "sync",
        }
    ]


@pytest.mark.asyncio
async def test_answer_error_update_does_not_confirm_a_failing_checkpoint() -> None:
    terminal_events: list[Any] = []
    answer_error = {
        "type": "error",
        "data": "answer failed",
        "checkpoint_terminal": True,
    }

    async def failing_checkpoint_stream() -> AsyncIterator[Any]:
        yield ("custom", answer_error)
        yield ("updates", {"answer": {"answer_error": "answer failed"}})
        raise RuntimeError("sync checkpoint failed")

    async def terminal_sink(event: Any) -> None:
        terminal_events.append(event)

    consumer = stream_graph(
        _FakeGraph(failing_checkpoint_stream()),
        {"query": "hello"},
        terminal_sink=terminal_sink,
    )

    with pytest.raises(RuntimeError, match="sync checkpoint failed"):
        await anext(consumer)

    assert terminal_events == []


@pytest.mark.asyncio
async def test_stream_graph_closes_and_awaits_iterator_on_consumer_cancellation() -> (
    None
):
    blocked = asyncio.Event()
    stream = _FakeGraphStream([("updates", blocked)])
    graph = _FakeGraph(stream)
    consumer = stream_graph(graph, {"query": "hello"})

    task = asyncio.create_task(consumer.__anext__())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True
    assert stream.close_completed is True


@pytest.mark.asyncio
async def test_stream_graph_closes_and_awaits_iterator_when_consumer_closes() -> None:
    stream = _FakeGraphStream(
        [
            (
                "custom",
                {"type": "token", "data": "first"},
            ),
            asyncio.Event(),
        ]
    )
    consumer = stream_graph(_FakeGraph(stream), {"query": "hello"})

    await consumer.__anext__()
    await consumer.aclose()

    assert stream.closed is True
    assert stream.close_completed is True


@pytest.mark.asyncio
async def test_stream_graph_preserves_cancellation_during_failing_close() -> None:
    class CloseBlockingStream:
        def __init__(self) -> None:
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()
            self.close_completed = False

        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.close_completed = True
            raise RuntimeError("close failed")

    stream = CloseBlockingStream()
    consumer = stream_graph(_FakeGraph(stream), {"query": "hello"})
    task = asyncio.create_task(anext(consumer))
    await stream.close_started.wait()

    task.cancel()
    stream.release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.close_completed is True
