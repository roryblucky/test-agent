"""RAG KMS Application — FastAPI entry point."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.router import router
from app.config.loader import load_config
from app.core.http_client_pool import HttpClientPool
from app.services.tenant_manager import TenantManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request timeout middleware
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 120  # 2 min max per request


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Cancel requests that exceed the configured timeout.

    SSE streaming responses are excluded — they have their own lifecycle
    managed by the client disconnect.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        # Skip timeout for SSE streaming endpoints
        if request.url.path.endswith("/stream"):
            return await call_next(request)
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return Response(
                content='{"detail":"Request timed out"}',
                status_code=504,
                media_type="application/json",
            )


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — initialise and tear down shared resources."""
    # Startup
    from app.core.telemetry import TelemetryService

    # Initialize OpenTelemetry
    TelemetryService("agent-kms-api")

    http_pool = HttpClientPool()
    configs = load_config("config.json")
    app.state.tenant_manager = TenantManager(configs, http_pool)
    app.state.http_pool = http_pool

    logger.info("KMS started — tenants: %s", app.state.tenant_manager.tenant_ids)

    yield

    # Shutdown — close Redis session store if active
    from app.api.router import _session_store

    if hasattr(_session_store, "close"):
        await _session_store.close()

    await http_pool.close_all()
    logger.info("KMS shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG KMS",
    description="Multi-tenant RAG Knowledge Management System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(TimeoutMiddleware)
app.include_router(router)
