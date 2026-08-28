"""Test-only FastAPI registration for the minimal v2 tracer."""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.checkpointing import (
    FencedAsyncPostgresSaver,
    checkpoint_namespace_for,
    initial_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.contracts import TracerStreamEvent, V2QueryRequest
from app.langgraph_v2.graph import TracerState, build_tracer_graph, tracer_graph
from app.langgraph_v2.run_events import (
    CLAIM_HEARTBEAT_INTERVAL_SECONDS,
    ClaimFenced,
    EventInput,
    RunEventRepository,
)

_INSTANCE_ID = os.environ.get("LANGGRAPH_V2_INSTANCE_ID", socket.gethostname())


async def _refresh_claim(
    repository: RunEventRepository,
    tenant_id: str,
    run_id: uuid.UUID,
    owner_instance_id: str,
    execution_epoch: int,
) -> None:
    """Refresh a request's claim until it is cancelled or fenced."""
    while True:
        await asyncio.sleep(CLAIM_HEARTBEAT_INTERVAL_SECONDS)
        try:
            await repository.heartbeat(
                tenant_id=tenant_id,
                run_id=run_id,
                owner_instance_id=owner_instance_id,
                execution_epoch=execution_epoch,
            )
        except ClaimFenced:
            return


class TracerGraph(Protocol):
    """Minimal graph invocation seam used by the test-only HTTP adapter."""

    async def ainvoke(
        self,
        state: TracerState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Run one graph invocation and return its final state."""
        ...


def create_tracer_router(graph: TracerGraph | None = None) -> APIRouter:
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
            claim = await repository.create_run(
                tenant_id=x_application_id,
                run_id=run_id,
                conversation_id=conversation_id,
                owner_instance_id=_INSTANCE_ID,
            )
            configured_checkpointer = getattr(
                http_request.app.state,
                "langgraph_v2_checkpointer",
                None,
            )
            selected_graph = graph or tracer_graph
            graph_config: RunnableConfig | None = None
            if graph is None and configured_checkpointer is not None:

                async def write_checkpoint_pointer(
                    checkpoint_id: str,
                    checkpoint_ns: str,
                ) -> None:
                    await repository.update_checkpoint_pointer(
                        tenant_id=x_application_id,
                        run_id=run_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                        checkpoint_id=checkpoint_id,
                        checkpoint_ns=checkpoint_ns,
                    )

                checkpoint_ns = checkpoint_namespace_for(
                    x_application_id,
                    str(run_id),
                    claim.execution_epoch,
                )
                selected_graph = build_tracer_graph(
                    FencedAsyncPostgresSaver(
                        pool,
                        checkpoint_namespace=checkpoint_ns,
                        pointer_writer=write_checkpoint_pointer,
                    )
                )
                graph_config = initial_checkpoint_config(
                    thread_id=thread_id_for(
                        x_application_id,
                        conversation_id,
                    ),
                    checkpoint_ns=checkpoint_ns,
                )
            heartbeat_task = asyncio.create_task(
                _refresh_claim(
                    repository,
                    x_application_id,
                    run_id,
                    claim.owner_instance_id,
                    claim.execution_epoch,
                )
            )
            try:
                state: TracerState = {
                    "query": payload.query,
                    "conversation_id": conversation_id,
                    "client_request_id": payload.client_request_id,
                    "events": [],
                }
                if graph_config is None:
                    result = await selected_graph.ainvoke(state)
                else:
                    result = await selected_graph.ainvoke(state, config=graph_config)
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
                            owner_instance_id=claim.owner_instance_id,
                            execution_epoch=claim.execution_epoch,
                        )
                    else:
                        persisted = await repository.append_event(
                            tenant_id=x_application_id,
                            run_id=run_id,
                            event=event_input,
                            owner_instance_id=claim.owner_instance_id,
                            execution_epoch=claim.execution_epoch,
                        )
                    yield event.model_copy(
                        update={"sequence": persisted.sequence}
                    ).to_sse()
            finally:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task

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
        app.include_router(create_tracer_router(graph))
