from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any
from uuid import UUID

import pytest
from fastapi import HTTPException
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.run_events import RunEventRepository
from app.langgraph_v2.runtime import LocalRunRuntime
from app.langgraph_v2.stream import GraphStreamCleanupError
from tests.integration.test_langgraph_v2_tracer import (
    persistent_tracer_app,
    seed_subject_conversation,
    stream_request,
    v2_stream_endpoint,
)


def _event_frame(frame: str) -> dict[str, Any]:
    return json.loads(frame.removeprefix("data: ").strip())


class _CompletedStream:
    async def __anext__(self) -> Any:
        if hasattr(self, "done"):
            raise StopAsyncIteration
        self.done = True
        return (
            "custom",
            {
                "event_key": "lifecycle:completed:0",
                "type": "done",
                "data": {"answer": "canonical answer"},
            },
        )

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def aclose(self) -> None:
        self.closed = True


class _CompletedGraph:
    def __init__(self) -> None:
        self.stream = _CompletedStream()
        self.astream_called = False

    def astream(self, state: Any, **options: Any) -> _CompletedStream:
        del state, options
        self.astream_called = True
        return self.stream


class _RealtimeStream:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.completed = False
        self.close_completed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if self.completed:
            raise StopAsyncIteration
        if not self.started.is_set():
            self.started.set()
            return (
                "custom",
                {
                    "event_key": "phase:answer:token:0",
                    "type": "token",
                    "data": "partial",
                },
            )
        await self.release.wait()
        if self.release.is_set():
            self.completed = True
            return (
                "updates",
                {
                    "finalization": {
                        "events": [
                            {
                                "event_key": "lifecycle:completed:0",
                                "type": "done",
                                "data": {"answer": "partial complete"},
                            }
                        ]
                    }
                },
            )
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True
        await asyncio.sleep(0)
        self.close_completed = True
        self.release.set()


class _RealtimeGraph:
    def __init__(self) -> None:
        self.stream = _RealtimeStream()

    def astream(self, state: Any, **options: Any) -> _RealtimeStream:
        del state, options
        return self.stream


@pytest.mark.asyncio
async def test_query_executes_astream_in_request_and_persists_one_assistant_message(
    langgraph_v2_migrated_database_url: str,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )
    graph = _CompletedGraph()
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, graph)

    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        frames = [frame async for frame in response.body_iterator]

        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            messages = await ConversationMessageRepository(pool).list_messages(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="conversation-1",
            )

            run = await RunEventRepository(pool).get_run(
                "tenant-a", UUID(response.headers["x-run-id"])
            )

    assert graph.astream_called is True
    assert _event_frame(frames[0]) == {
        "type": "done",
        "sequence": 1,
        "data": {"answer": "canonical answer"},
    }
    assert response.headers["x-thread-id"]
    assert run.status == "completed"
    assert [(message.role, message.content) for message in messages] == [
        ("user", "hello"),
        ("assistant", "canonical answer"),
    ]


@pytest.mark.asyncio
async def test_closing_query_sse_closes_graph_and_interrupts_run(
    langgraph_v2_migrated_database_url: str,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )

    class BlockingStream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.closed = False
            self.close_completed = False

        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self) -> Any:
            self.started.set()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            self.closed = True
            await asyncio.sleep(0)
            self.close_completed = True

    class BlockingGraph:
        def __init__(self) -> None:
            self.stream = BlockingStream()

        def astream(self, state: Any, **options: Any) -> BlockingStream:
            del state, options
            return self.stream

    graph = BlockingGraph()
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, graph)
    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        subscriber = response.body_iterator
        pending_read = asyncio.create_task(anext(subscriber))
        await graph.stream.started.wait()
        pending_read.cancel()
        with suppress(asyncio.CancelledError):
            await pending_read
        await subscriber.aclose()

        run = await RunEventRepository(app.state.langgraph_v2_postgres_pool).get_run(
            "tenant-a", UUID(response.headers["x-run-id"])
        )

    assert graph.stream.closed is True
    assert graph.stream.close_completed is True
    assert run.status == "interrupted"
    assert run.owner_instance_id == ""


@pytest.mark.asyncio
async def test_query_yields_answer_delta_before_graph_completion(
    langgraph_v2_migrated_database_url: str,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )
    graph = _RealtimeGraph()
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, graph)

    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        subscriber = response.body_iterator
        first = await anext(subscriber)
        assert _event_frame(first)["data"] == "partial"

        graph.stream.release.set()
        remaining = [frame async for frame in subscriber]

        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            messages = await ConversationMessageRepository(pool).list_messages(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="conversation-1",
            )

    assert _event_frame(remaining[-1])["data"]["answer"] == "partial complete"
    assert [(message.role, message.content) for message in messages] == [
        ("user", "hello"),
        ("assistant", "partial complete"),
    ]


@pytest.mark.asyncio
async def test_closing_query_after_answer_token_closes_graph_and_persists_no_partial_assistant_message(
    langgraph_v2_migrated_database_url: str,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )
    graph = _RealtimeGraph()
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, graph)

    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        subscriber = response.body_iterator
        token_frame = _event_frame(await anext(subscriber))
        assert token_frame["type"] == "token"
        assert token_frame["data"] == "partial"

        pending_read = asyncio.create_task(anext(subscriber))
        await graph.stream.started.wait()
        pending_read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_read

        await subscriber.aclose()

        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            messages = await ConversationMessageRepository(pool).list_messages(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="conversation-1",
            )
            run = await RunEventRepository(pool).get_run(
                "tenant-a", UUID(response.headers["x-run-id"])
            )

    assert graph.stream.closed is True
    assert graph.stream.close_completed is True
    assert run.status == "interrupted"
    assert [(message.role, message.content) for message in messages] == [
        ("user", "hello"),
    ]


@pytest.mark.asyncio
async def test_graph_close_failure_is_reported_after_request_cleanup(
    langgraph_v2_migrated_database_url: str,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )

    class FailingCloseStream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.close_calls = 0

        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self) -> Any:
            self.started.set()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("graph close failed")

    class FailingCloseGraph:
        def __init__(self) -> None:
            self.stream = FailingCloseStream()

        def astream(self, state: Any, **options: Any) -> FailingCloseStream:
            del state, options
            return self.stream

    graph = FailingCloseGraph()
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, graph)
    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        pending_read = asyncio.create_task(anext(response.body_iterator))
        await graph.stream.started.wait()
        pending_read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_read
        await response.body_iterator.aclose()
        run = await RunEventRepository(app.state.langgraph_v2_postgres_pool).get_run(
            "tenant-a", UUID(response.headers["x-run-id"])
        )

    assert graph.stream.close_calls == 1
    assert run.status == "interrupted"
    assert run.owner_instance_id == ""


@pytest.mark.asyncio
async def test_graph_close_failure_without_primary_error_is_reported(
    langgraph_v2_migrated_database_url: str,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )

    class FailingCloseStream:
        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            raise RuntimeError("normal close failed")

    class FailingCloseGraph:
        def astream(self, state: Any, **options: Any) -> FailingCloseStream:
            del state, options
            return FailingCloseStream()

    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url, FailingCloseGraph()
    )
    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        with pytest.raises(GraphStreamCleanupError, match="normal close failed"):
            await anext(response.body_iterator)
        run = await RunEventRepository(app.state.langgraph_v2_postgres_pool).get_run(
            "tenant-a", UUID(response.headers["x-run-id"])
        )

    assert run.status == "interrupted"
    assert run.owner_instance_id == ""


@pytest.mark.asyncio
async def test_query_preserves_shutdown_admission_contract(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)
    runtime = LocalRunRuntime()
    await runtime.stop_and_wait_for_checkpoint_boundary()
    app.state.langgraph_v2_runtime = runtime

    async with app.router.lifespan_context(app):
        with pytest.raises(HTTPException) as raised:
            await v2_stream_endpoint(app)(
                V2QueryRequest(query="hello"),
                stream_request(app),
                request_context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
            )

    assert raised.value.status_code == 503
    assert raised.value.detail == "LangGraph v2 is shutting down"
