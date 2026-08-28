"""Test-only FastAPI registration for the minimal v2 tracer."""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import StreamingResponse
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.contracts import TracerStreamEvent, V2QueryRequest
from app.langgraph_v2.graph import TracerState, tracer_graph
from app.langgraph_v2.run_events import EventInput, RunEventRepository

_INSTANCE_ID = os.environ.get("LANGGRAPH_V2_INSTANCE_ID", socket.gethostname())


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
        payload: V2QueryRequest,
        http_request: Request,
        x_application_id: Annotated[str, Header(alias="X-Application-Id")],
        x_user_groups: Annotated[str, Header(alias="X-User-Groups")] = "",
    ) -> StreamingResponse:
        """Run the deterministic tracer and return its events as SSE."""
        del x_user_groups
        run_id = uuid.uuid4()
        conversation_id = payload.conversation_id or str(uuid.uuid4())

        async def event_generator() -> AsyncIterator[str]:
            configured_pool = getattr(
                http_request.app.state,
                "langgraph_v2_postgres_pool",
                None,
            )
            if configured_pool is None:
                raise RuntimeError("LangGraph v2 PostgreSQL is not configured")
            pool = cast(AsyncConnectionPool[Any], configured_pool)
            repository = RunEventRepository(pool)
            await repository.create_run(
                tenant_id=x_application_id,
                run_id=run_id,
                conversation_id=conversation_id,
                owner_instance_id=_INSTANCE_ID,
            )
            result = await graph.ainvoke(
                {
                    "query": payload.query,
                    "conversation_id": conversation_id,
                    "client_request_id": payload.client_request_id,
                    "events": [],
                }
            )
            for raw_event in result["events"]:
                event = TracerStreamEvent.model_validate(raw_event)
                event_input = EventInput(
                    event_key=event.event_key,
                    type=event.type,
                    step=event.step,
                    data=event.data,
                )
                if event.type == "done":
                    persisted = await repository.complete_run(
                        tenant_id=x_application_id,
                        run_id=run_id,
                        event=event_input,
                        owner_instance_id=_INSTANCE_ID,
                    )
                else:
                    persisted = await repository.append_event(
                        tenant_id=x_application_id,
                        run_id=run_id,
                        event=event_input,
                        owner_instance_id=_INSTANCE_ID,
                    )
                yield event.model_copy(
                    update={"sequence": persisted.sequence}
                ).to_sse()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Run-Id": str(run_id),
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
