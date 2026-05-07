"""Unified FastAPI router.

All endpoints share the same router; the ``X-Application-Id`` header
determines which tenant's components are loaded.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.dependencies import TenantContext, get_tenant, get_tenant_manager
from app.api.schemas import HealthResponse, QueryRequest, QueryResponse
from app.memory.session_store import BaseSessionStore, InMemorySessionStore
from app.services.events import EventEmitter
from app.services.exceptions import ContentFlaggedError
from app.services.tenant_manager import TenantManager

router = APIRouter(prefix="/api/v1", tags=["KMS"])

# Active streaming sessions: {"app_id:session_id": EventEmitter}
# Used by the stop endpoint to signal cancellation.
_active_sessions: dict[str, EventEmitter] = {}


def _create_session_store() -> BaseSessionStore:
    """Create session store based on ``SESSION_STORE_URL`` env var.

    - Set ``SESSION_STORE_URL=redis://redis-svc:6379/0`` for production.
    - Leave unset for local dev (in-memory fallback).
    """
    url = os.environ.get("SESSION_STORE_URL")
    if url and url.startswith("redis"):
        from app.memory.redis_session_store import RedisSessionStore

        return RedisSessionStore(url)
    return InMemorySessionStore()


_session_store = _create_session_store()


def get_session_store() -> BaseSessionStore:
    """Get the active session store instance."""
    return _session_store


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    raw_request: Request,
    tenant: Annotated[TenantContext, Depends(get_tenant)],
    session_store: Annotated[BaseSessionStore, Depends(get_session_store)],
) -> QueryResponse:
    """Execute the full RAG pipeline (non-streaming) for the tenant."""
    session_id = request.session_id or str(uuid.uuid4())
    store_key = f"{tenant.app_id}:{session_id}"
    message_history = await session_store.get(store_key)

    flow = tenant.manager.get_flow_engine(tenant.app_id)
    try:
        ctx = await flow.execute(
            request.query,
            session_id=session_id,
            message_history=message_history,
        )
    except ContentFlaggedError as exc:
        return QueryResponse(
            query=request.query,
            moderation=exc.result,
            answer="Your query was flagged by content moderation.",
            session_id=session_id,
        )

    # Persist updated conversation history
    if ctx.new_messages:
        full_history = message_history + ctx.new_messages
        await session_store.save(store_key, full_history)

    # Record token usage for rate limiting (fire-and-forget)
    rate_limiter = getattr(raw_request.app.state, "rate_limiter", None)
    if rate_limiter and ctx.total_usage.total_tokens > 0:
        asyncio.create_task(
            rate_limiter.record_token_usage(
                tenant.app_id, ctx.total_usage.total_tokens
            )
        )

    return QueryResponse.from_flow_context(ctx)


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    tenant: Annotated[TenantContext, Depends(get_tenant)],
    session_store: Annotated[BaseSessionStore, Depends(get_session_store)],
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
    session_id = request.session_id or str(uuid.uuid4())
    store_key = f"{tenant.app_id}:{session_id}"

    async def event_generator() -> AsyncIterator[str]:
        emitter = EventEmitter()
        flow = tenant.manager.get_flow_engine(tenant.app_id)
        message_history = await session_store.get(store_key)

        # Register for stop signal lookup
        _active_sessions[store_key] = emitter

        async def run_pipeline() -> None:
            """Execute the pipeline in a background task."""
            try:
                ctx = await flow.execute(
                    request.query,
                    emitter=emitter,
                    session_id=session_id,
                    message_history=message_history,
                )
                # Persist updated conversation history
                if ctx.new_messages:
                    full_history = message_history + ctx.new_messages
                    await session_store.save(store_key, full_history)

                # Record token usage for rate limiting
                rate_limiter = getattr(
                    tenant.manager, "_rate_limiter", None
                )
                if rate_limiter is None:
                    # Fallback — not available in closure, skip
                    pass
                if ctx.total_usage.total_tokens > 0 and rate_limiter:
                    await rate_limiter.record_token_usage(
                        tenant.app_id, ctx.total_usage.total_tokens
                    )

                # Only emit done if not already stopped/closed
                if not emitter.is_cancelled:
                    result = QueryResponse.from_flow_context(ctx)
                    await emitter.emit_done(result.model_dump())
            except ContentFlaggedError as exc:
                if not emitter.is_cancelled:
                    await emitter.emit_error(str(exc))
            except Exception as exc:
                if not emitter.is_cancelled:
                    await emitter.emit_error(str(exc))

        # Run pipeline concurrently — emitter yields events in real time
        pipeline_task = asyncio.create_task(run_pipeline())

        try:
            async for sse_line in emitter:
                yield sse_line
        finally:
            # Cleanup: unregister session and cancel pipeline
            _active_sessions.pop(store_key, None)
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
# Stop generation endpoint
# ------------------------------------------------------------------


class StopRequest(BaseModel):
    """Request body for stopping an active generation."""

    session_id: str = Field(description="Session ID of the active stream to stop")


class StopResponse(BaseModel):
    """Response from a stop request."""

    status: str  # "stopped" | "not_found"
    session_id: str


@router.post("/query/stop", response_model=StopResponse)
async def stop_generation(
    request: StopRequest,
    tenant: Annotated[TenantContext, Depends(get_tenant)],
) -> StopResponse:
    """Stop an active streaming generation.

    Signals the pipeline to stop producing tokens.  The stream will
    emit a ``stopped`` event with the partial response collected so far,
    then close.
    """
    store_key = f"{tenant.app_id}:{request.session_id}"
    emitter = _active_sessions.get(store_key)

    if emitter is None:
        return StopResponse(status="not_found", session_id=request.session_id)

    emitter.cancel()
    return StopResponse(status="stopped", session_id=request.session_id)


# ------------------------------------------------------------------
# Health & utility endpoints
# ------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(
    tenant_manager: Annotated[TenantManager, Depends(get_tenant_manager)],
) -> HealthResponse:
    """Health check — returns loaded tenant IDs."""
    return HealthResponse(status="ok", tenants=tenant_manager.tenant_ids)
