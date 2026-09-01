"""RAG KMS Application — FastAPI entry point."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.api.admin_router import admin_router
from app.api.router import router
from app.config.loader import load_config
from app.core.audit_middleware import AuditMiddleware
from app.core.http_client_pool import HttpClientPool
from app.core.rate_limit_middleware import TenantRateLimitMiddleware
from app.langgraph_v2.api import register_v2_routes
from app.langgraph_v2.postgres import postgres_lifespan
from app.services.tenant_manager import TenantManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency Limiter Middleware
# ---------------------------------------------------------------------------


class ConcurrencyLimiterMiddleware(BaseHTTPMiddleware):
    """Limit legacy requests without queueing request-owned v2 streams."""

    def __init__(self, app: ASGIApp, max_concurrent_requests: int = 100):
        super().__init__(app)
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply concurrency limit to the request dispatch."""
        path = request.url.path
        if path == "/v2/query/stream":
            return await call_next(request)
        async with self._semaphore:
            return await call_next(request)


# ---------------------------------------------------------------------------
# Request timeout middleware
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 120  # 2 min max per request


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Cancel requests that exceed the configured timeout.

    SSE streaming responses are excluded — they have their own lifecycle
    managed by the client disconnect.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply timeout bounds to the request dispatch unless it's an SSE stream."""
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
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan — initialise and tear down shared resources."""
    # Startup
    from app.api.router import get_session_store
    from app.config.config_reloader import ConfigReloader
    from app.core.audit import AuditLogger, AuditSink, BigQueryAuditSink, FileAuditSink
    from app.core.rate_limiter import create_rate_limiter
    from app.core.telemetry import TelemetryService
    from app.langgraph_v2.output_assessments import (
        BigQueryOutputAssessmentAudit,
        LoggingOutputAssessmentAudit,
    )

    async with AsyncExitStack() as cleanup:
        # Initialize OpenTelemetry
        TelemetryService("agent-kms-api")

        http_pool = HttpClientPool()
        cleanup.push_async_callback(http_pool.close_all)
        configs = load_config("config.json")
        app.state.tenant_manager = TenantManager(configs, http_pool)
        app.state.http_pool = http_pool

        # Rate limiter (Redis in production, InMemory for dev)
        redis_url = os.environ.get("RATE_LIMIT_REDIS_URL")
        app.state.rate_limiter = create_rate_limiter(redis_url)
        cleanup.push_async_callback(app.state.rate_limiter.close)

        # Audit logger with configured sinks
        audit_sinks: list[AuditSink] = [FileAuditSink()]
        gcp_project = os.environ.get("GCP_PROJECT_ID")
        bigquery_assessment_audit: BigQueryOutputAssessmentAudit | None = None
        if gcp_project:
            try:
                audit_sinks.append(BigQueryAuditSink(project_id=gcp_project))
                bigquery_assessment_audit = BigQueryOutputAssessmentAudit(
                    project_id=gcp_project
                )
            except Exception:
                logger.warning("BigQuery audit sink not available, using file only")
        app.state.audit_logger = AuditLogger(sinks=audit_sinks)
        cleanup.push_async_callback(app.state.audit_logger.close)
        app.state.langgraph_v2_output_assessment_audit = (
            bigquery_assessment_audit or LoggingOutputAssessmentAudit()
        )
        if bigquery_assessment_audit is not None:
            cleanup.push_async_callback(bigquery_assessment_audit.close)

        # Config hot-reloader
        app.state.config_reloader = ConfigReloader(app, http_pool)

        logger.info("KMS started — tenants: %s", app.state.tenant_manager.tenant_ids)

        cleanup.push_async_callback(get_session_store().close)
        await cleanup.enter_async_context(postgres_lifespan(app))
        yield
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

# Middleware stack (order matters — outermost first)
app.add_middleware(TimeoutMiddleware)
max_concurrent = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "100"))
app.add_middleware(ConcurrencyLimiterMiddleware, max_concurrent_requests=max_concurrent)
app.add_middleware(AuditMiddleware)
app.add_middleware(TenantRateLimitMiddleware)

# Routers
app.include_router(router)
app.include_router(admin_router)
register_v2_routes(
    app,
    enabled=(
        os.environ.get("LANGGRAPH_V2_UAT_ENABLED") == "1"
        or os.environ.get("LANGGRAPH_V2_LINEAR_CORE_ENABLED") == "1"
    ),
)
