from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from typing import Any, Self

import pytest
import uvicorn
from fastapi import FastAPI
from pydantic_ai.usage import RunUsage

from app.langgraph_v2.answer import AnswerOutput, PydanticAIAnswerActor
from app.langgraph_v2.api import GraphStream, register_v2_routes
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import thread_checkpoint_config, thread_id_for
from app.langgraph_v2.graph import build_linear_graph
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.test_langgraph_v2_linear_core import (
    configure_linear_tenant,
    seed_subject_conversation,
)


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id="d1", content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        return RerankingResult(documents=documents)


class _BlockingPydanticStream:
    def __init__(self) -> None:
        self.entered = False
        self.exited = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.cleanup_completed = asyncio.Event()

    async def __aenter__(self) -> Self:
        self.entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exited.set()
        await self.allow_cleanup.wait()
        self.cleanup_completed.set()

    async def stream_output(
        self, *, debounce_by: float | None
    ) -> AsyncIterator[AnswerOutput]:
        assert debounce_by is None
        yield AnswerOutput(answer="partial")
        await asyncio.Event().wait()

    async def get_output(self) -> AnswerOutput:
        raise AssertionError("disconnect must prevent complete model output")

    def usage(self) -> RunUsage:
        return RunUsage()


class _PydanticAgent:
    def __init__(self, stream: _BlockingPydanticStream) -> None:
        self.stream = stream

    def run_stream(self, prompt: str, **kwargs: Any) -> _BlockingPydanticStream:
        del prompt, kwargs
        return self.stream


class _TrackedGraph:
    def __init__(self, *, thread_id: str) -> None:
        self.target: GraphStream | None = None
        self.config = thread_checkpoint_config(thread_id=thread_id, checkpoint_ns="")
        self.cancelled = asyncio.Event()
        self.closed = asyncio.Event()

    def astream(
        self, state: Mapping[str, Any], **options: Any
    ) -> AsyncIterator[object]:
        async def iterate() -> AsyncIterator[object]:
            assert self.target is not None
            if options.get("config") is None:
                options["config"] = self.config
            target_stream = self.target.astream(state, **options)
            try:
                async for item in target_stream:
                    yield item
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            finally:
                close = getattr(target_stream, "aclose", None)
                if close is not None:
                    await close()
                self.closed.set()

        return iterate()


@asynccontextmanager
async def _serve_uvicorn(app: FastAPI) -> AsyncGenerator[int]:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(128)
    port = int(server_socket.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            timeout_graceful_shutdown=1,
        )
    )
    server_task = asyncio.create_task(server.serve(sockets=[server_socket]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        yield port
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=5)
        finally:
            server_socket.close()


async def _abort_and_wait(writer: asyncio.StreamWriter) -> None:
    writer.transport.abort()
    with suppress(TimeoutError, ConnectionError):
        await asyncio.wait_for(writer.wait_closed(), timeout=2)


@asynccontextmanager
async def _serve_tcp_forwarding_proxy(upstream_port: int) -> AsyncGenerator[int]:
    connections: set[asyncio.Task[None]] = set()

    async def copy(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        while data := await reader.read(4096):
            writer.write(data)
            await writer.drain()

    async def forward(
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        connections.add(task)
        upstream_writer: asyncio.StreamWriter | None = None
        copy_tasks: tuple[asyncio.Task[None], ...] = ()
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", upstream_port),
                timeout=5,
            )
            try:
                client_to_upstream = asyncio.create_task(
                    copy(client_reader, upstream_writer)
                )
                copy_tasks = (client_to_upstream,)
                upstream_to_client = asyncio.create_task(
                    copy(upstream_reader, client_writer)
                )
                copy_tasks = (client_to_upstream, upstream_to_client)
                _, pending = await asyncio.wait(
                    set(copy_tasks),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for pending_task in pending:
                    pending_task.cancel()
            finally:
                try:
                    for copy_task in copy_tasks:
                        if not copy_task.done():
                            copy_task.cancel()
                    await asyncio.wait_for(
                        asyncio.gather(*copy_tasks, return_exceptions=True),
                        timeout=5,
                    )
                finally:
                    await _abort_and_wait(upstream_writer)
        finally:
            await _abort_and_wait(client_writer)
            connections.discard(task)

    server = await asyncio.start_server(forward, "127.0.0.1", 0)
    sockets = server.sockets
    assert sockets
    port = int(sockets[0].getsockname()[1])
    try:
        yield port
    finally:
        server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), timeout=5)
        finally:
            connection_snapshot = tuple(connections)
            for connection in connection_snapshot:
                connection.cancel()
            if connection_snapshot:
                await asyncio.wait_for(
                    asyncio.gather(*connection_snapshot, return_exceptions=True),
                    timeout=5,
                )


@pytest.mark.asyncio
async def test_real_tcp_disconnect_cancels_and_awaits_graph_and_pydantic_stream(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "uvicorn-disconnect"
    context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url,
        conversation_id,
    )
    model_stream = _BlockingPydanticStream()
    answer_actor = PydanticAIAnswerActor(
        _PydanticAgent(model_stream)  # type: ignore[arg-type]
    )
    tracked_graph = _TrackedGraph(
        thread_id=thread_id_for(context.tenant_id, conversation_id)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=langgraph_v2_migrated_database_url),
        ):
            tracked_graph.target = build_linear_graph(
                app.state.langgraph_v2_checkpointer,
                tenant_id="tenant-a",
                request_context=context,
                retriever=_Retriever(),
                ranker=_Ranker(),
                answer_actor=answer_actor,
            )
            yield

    app = FastAPI(lifespan=lifespan)
    configure_linear_tenant(app)
    register_v2_routes(app, enabled=True, linear_graph_override=tracked_graph)

    async with _serve_uvicorn(app) as upstream_port:
        async with _serve_tcp_forwarding_proxy(upstream_port) as proxy_port:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", proxy_port),
                timeout=5,
            )
            try:
                body = json.dumps(
                    {"query": "hello", "sessionId": conversation_id},
                    separators=(",", ":"),
                ).encode()
                request = (
                    b"POST /v2/query/stream HTTP/1.1\r\n"
                    + f"Host: 127.0.0.1:{proxy_port}\r\n".encode()
                    + b"Content-Type: application/json\r\n"
                    + b"X-Application-Id: tenant-a\r\n"
                    + b"X-Subject-Id: subject-a\r\n"
                    + b"Connection: close\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                writer.write(request)
                await asyncio.wait_for(writer.drain(), timeout=5)
                received = b""
                try:
                    async with asyncio.timeout(5):
                        while b'"data": "partial"' not in received:
                            chunk = await reader.read(4096)
                            assert chunk
                            received += chunk
                except TimeoutError as error:
                    raise AssertionError(received.decode(errors="replace")) from error
                writer.transport.abort()
                try:
                    await asyncio.wait_for(model_stream.exited.wait(), timeout=5)
                    assert not tracked_graph.closed.is_set()
                finally:
                    model_stream.allow_cleanup.set()
                await asyncio.wait_for(tracked_graph.closed.wait(), timeout=5)
                await asyncio.wait_for(model_stream.cleanup_completed.wait(), timeout=5)
            finally:
                model_stream.allow_cleanup.set()
                await _abort_and_wait(writer)

    assert tracked_graph.cancelled.is_set()
    assert tracked_graph.closed.is_set()
    assert model_stream.entered is True
    assert model_stream.exited.is_set()
    assert model_stream.cleanup_completed.is_set()
