"""Test-only FastAPI registration for the minimal v2 tracer."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, FastAPI, Header
from fastapi.responses import StreamingResponse

from app.langgraph_v2.contracts import TracerStreamEvent, V2QueryRequest
from app.langgraph_v2.graph import TracerState, tracer_graph


class TracerGraph(Protocol):
    """Minimal graph invocation seam used by the test-only HTTP adapter."""

    async def ainvoke(self, state: TracerState) -> dict[str, Any]:
        """Run one graph invocation and return its final state."""
        ...


def create_tracer_router(graph: TracerGraph) -> APIRouter:
    """Create the test-only router around an injected graph invocation seam."""
    router = APIRouter(tags=["LangGraph v2 tracer"])

    @router.post("/v2/query/stream")
    async def query_stream(
        request: V2QueryRequest,
        x_application_id: Annotated[str, Header(alias="X-Application-Id")],
        x_user_groups: Annotated[str, Header(alias="X-User-Groups")] = "",
    ) -> StreamingResponse:
        """Run the deterministic tracer and return its events as SSE."""
        del x_application_id, x_user_groups
        run_id = str(uuid.uuid4())
        conversation_id = request.conversation_id or str(uuid.uuid4())

        async def event_generator() -> AsyncIterator[str]:
            result = await graph.ainvoke(
                {
                    "query": request.query,
                    "conversation_id": conversation_id,
                    "client_request_id": request.client_request_id,
                    "events": [],
                }
            )
            for raw_event in result["events"]:
                yield TracerStreamEvent.model_validate(raw_event).to_sse()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Run-Id": run_id,
                "X-Conversation-Id": conversation_id,
            },
        )

    return router


def register_tracer_routes(
    app: FastAPI,
    *,
    enabled: bool,
    graph: TracerGraph | None = None,
) -> None:
    """Register the test-only tracer routes when explicitly enabled."""
    if enabled:
        selected_graph = graph or cast(TracerGraph, tracer_graph)
        app.include_router(create_tracer_router(selected_graph))
