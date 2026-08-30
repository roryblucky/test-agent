"""Shared authorized Artifact setup for graph integration tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from uuid import uuid4

from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    TurnNotFound,
)
from app.langgraph_v2.graph import TracerGraph, build_tracer_graph


class _AuthorizedArtifactGraph:
    def __init__(
        self,
        pool: AsyncConnectionPool[Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._pool = pool
        self._args = args
        self._kwargs = kwargs
        self._graph: TracerGraph | None = None

    async def _get_graph(self) -> TracerGraph:
        if self._graph is not None:
            return self._graph
        context = self._kwargs.pop(
            "request_context",
            TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
        )
        conversation_id = "c1"
        messages = ConversationMessageRepository(self._pool)
        await messages.resolve_conversation(
            context=context,
            conversation_id=conversation_id,
        )
        turn_id = self._kwargs.pop("current_turn_id", None) or uuid4()
        try:
            await messages.get_turn(
                context=context,
                conversation_id=conversation_id,
                turn_id=turn_id,
            )
        except TurnNotFound:
            await messages.create_turn(
                context=context,
                conversation_id=conversation_id,
                turn_id=turn_id,
                content="question",
                idempotency_key=f"turn:{turn_id}:user",
            )
        self._graph = build_tracer_graph(
            *self._args,
            current_turn_id=turn_id,
            request_context=context,
            **self._kwargs,
        )
        return self._graph

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await (await self._get_graph()).ainvoke(*args, **kwargs)

    async def aget_state(self, *args: Any, **kwargs: Any) -> Any:
        return await (await self._get_graph()).aget_state(*args, **kwargs)

    def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async def iterate() -> AsyncIterator[Any]:
            stream = (await self._get_graph()).astream(*args, **kwargs)
            async for item in stream:
                yield item

        return iterate()


def build_artifact_test_graph(
    pool: AsyncConnectionPool[Any],
    *args: Any,
    **kwargs: Any,
) -> TracerGraph:
    """Lazily create an authorized Turn for an Artifact-backed test graph."""
    return cast(TracerGraph, _AuthorizedArtifactGraph(pool, args, kwargs))
