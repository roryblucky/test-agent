import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

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
    def __init__(self, stream: _FakeGraphStream) -> None:
        self.stream = stream
        self.inputs: list[Any] = []
        self.options: list[dict[str, Any]] = []

    def astream(self, graph_input: Any, **options: Any) -> _FakeGraphStream:
        self.inputs.append(graph_input)
        self.options.append(options)
        return self.stream


@pytest.mark.asyncio
async def test_stream_graph_translates_approved_modes_and_ignores_diagnostics() -> None:
    graph = _FakeGraph(
        _FakeGraphStream(
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
    )

    frames = [frame async for frame in stream_graph(graph, {"query": "hello"})]

    assert [_payload(frame) for frame in frames] == [
        {"type": "step_start", "sequence": 1, "step": "answer"},
        {"type": "token", "sequence": 2, "data": "hello"},
        {"type": "token", "sequence": 3, "data": " world"},
    ]
    assert graph.inputs == [{"query": "hello"}]
    assert graph.options == [
        {
            "config": None,
            "stream_mode": ["updates", "custom", "messages"],
            "durability": "sync",
        }
    ]
    assert graph.stream.closed is True
    assert graph.stream.close_completed is True


@pytest.mark.asyncio
async def test_stream_graph_accepts_none_as_a_checkpoint_resume_input() -> None:
    graph = _FakeGraph(
        _FakeGraphStream(
            [
                (
                    "updates",
                    {
                        "query": {
                            "event_key": "phase:query:step_completed:1",
                            "type": "step_completed",
                            "data": {"query": "continued"},
                        }
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
        "sequence": 1,
        "data": {"query": "continued"},
    }
    assert graph.inputs == [None]
    assert graph.options == [
        {
            "config": {"configurable": {"thread_id": "thread-1"}},
            "stream_mode": ["updates", "custom", "messages"],
            "durability": "sync",
        }
    ]


@pytest.mark.asyncio
async def test_stream_graph_closes_and_awaits_iterator_on_consumer_cancellation() -> None:
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
