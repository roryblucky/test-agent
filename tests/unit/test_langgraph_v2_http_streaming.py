from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast

import pytest
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from app.langgraph_v2.api import (
    _RequestOwnedStreamingResponse,  # pyright: ignore[reportPrivateUsage]
)


def _http_scope(spec_version: str = "2.4") -> Scope:
    return cast(Scope, {"type": "http", "asgi": {"spec_version": spec_version}})


@pytest.mark.asyncio
async def test_send_failure_preserves_starlette_client_disconnect_contract() -> None:
    async def content() -> AsyncIterator[str]:
        yield "hello"

    async def receive() -> Message:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def failing_send(message: Message) -> None:
        del message
        raise OSError("client socket closed")

    response = _RequestOwnedStreamingResponse(content())
    with pytest.raises(ClientDisconnect):
        await response(_http_scope(), receive, failing_send)


@pytest.mark.asyncio
async def test_pre_24_send_failure_preserves_starlette_oserror_contract() -> None:
    async def content() -> AsyncIterator[str]:
        yield "hello"

    async def receive() -> Message:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def failing_send(message: Message) -> None:
        del message
        raise OSError("client socket closed")

    response = _RequestOwnedStreamingResponse(content())
    with pytest.raises(OSError, match="client socket closed"):
        await response(_http_scope("2.3"), receive, failing_send)


@pytest.mark.asyncio
async def test_receive_failure_is_propagated_and_stream_is_awaited_closed() -> None:
    stream_started = asyncio.Event()
    stream_closed = asyncio.Event()

    async def content() -> AsyncIterator[str]:
        try:
            stream_started.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            stream_closed.set()

    async def failing_receive() -> Message:
        await stream_started.wait()
        raise RuntimeError("receive failed")

    async def send(message: Message) -> None:
        del message

    response = _RequestOwnedStreamingResponse(content())
    with pytest.raises(RuntimeError, match="receive failed"):
        await asyncio.wait_for(
            response(_http_scope(), failing_receive, send),
            timeout=1,
        )
    assert stream_closed.is_set()
