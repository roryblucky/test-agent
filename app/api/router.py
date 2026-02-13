"""Unified FastAPI router.

All endpoints share the same router; the ``X-Application-Id`` header
determines which tenant's components are loaded.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import TenantContext, get_tenant, get_tenant_manager
from app.api.schemas import HealthResponse, QueryRequest, QueryResponse
from app.services.events import EventEmitter
from app.services.exceptions import ContentFlaggedError
from app.services.tenant_manager import TenantManager

router = APIRouter(prefix="/api/v1", tags=["KMS"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    tenant: TenantContext = Depends(get_tenant),
) -> QueryResponse:
    """Execute the full RAG pipeline (non-streaming) for the tenant."""
    flow = tenant.manager.get_flow_engine(tenant.app_id)
    try:
        ctx = await flow.execute(request.query)
    except ContentFlaggedError as exc:
        return QueryResponse(
            query=request.query,
            moderation=exc.result,
            answer="Your query was flagged by content moderation.",
        )
    return QueryResponse.from_flow_context(ctx)


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    tenant: TenantContext = Depends(get_tenant),
) -> StreamingResponse:
    """Execute the RAG pipeline with real-time SSE streaming.

    Event protocol (one JSON per ``data:`` line):

    - ``{"type": "step_start",     "step": "retriever"}``
    - ``{"type": "step_completed", "step": "retriever", "data": {…}}``
    - ``{"type": "token",          "data": "partial text"}``
    - ``{"type": "done",           "data": {…final QueryResponse…}}``
    - ``{"type": "error",          "data": "error message"}``

    This protocol is consistent with LangGraph streaming events,
    OpenAI Assistants streaming, and Vercel AI SDK data stream protocol.
    """

    async def event_generator() -> AsyncIterator[str]:
        emitter = EventEmitter()
        flow = tenant.manager.get_flow_engine(tenant.app_id)

        async def run_pipeline() -> None:
            """Execute the pipeline in a background task."""
            try:
                ctx = await flow.execute(request.query, emitter=emitter)
                result = QueryResponse.from_flow_context(ctx)
                await emitter.emit_done(result.model_dump())
            except ContentFlaggedError as exc:
                await emitter.emit_error(str(exc))
            except Exception as exc:
                await emitter.emit_error(str(exc))

        # Run pipeline concurrently — emitter yields events in real time
        pipeline_task = asyncio.create_task(run_pipeline())

        try:
            async for sse_line in emitter:
                yield sse_line
        finally:
            # Ensure the pipeline task is cleaned up
            if not pipeline_task.done():
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------
# Health & utility endpoints
# ------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(
    tenant_manager: TenantManager = Depends(get_tenant_manager),
) -> HealthResponse:
    """Health check — returns loaded tenant IDs."""
    return HealthResponse(status="ok", tenants=tenant_manager.tenant_ids)
