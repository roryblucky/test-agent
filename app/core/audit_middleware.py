"""Audit middleware — captures request lifecycle and emits audit records.

Automatically logs every API request with user identity, tenant context,
timing, and outcome.  Audit writes are fire-and-forget (non-blocking).
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import Request, Response
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.audit import AuditLogger, AuditRecord

logger = logging.getLogger(__name__)

# Only audit API routes
_AUDITABLE_PREFIXES = ("/api/v1/query",)


class AuditMiddleware(BaseHTTPMiddleware):
    """Capture request metadata and emit audit records on completion."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        """Wrap request lifecycle with audit logging."""
        path = request.url.path

        # Skip non-auditable paths
        if not any(path.startswith(p) for p in _AUDITABLE_PREFIXES):
            return await call_next(request)

        audit_logger: AuditLogger | None = getattr(
            request.app.state, "audit_logger", None
        )
        if not audit_logger:
            return await call_next(request)

        # Pre-request: capture metadata
        start_time = time.time()
        app_id = request.headers.get("x-application-id", "unknown")
        user_id = request.headers.get("x-user-id", "anonymous")
        user_groups_raw = request.headers.get("x-user-groups", "")
        user_groups = [g.strip() for g in user_groups_raw.split(",") if g.strip()]

        # Determine action type
        is_stream = path.endswith("/stream")
        action = "query_stream" if is_stream else "query"

        # Extract query from request body (best-effort)
        query = ""
        try:
            body = await request.body()
            if body:
                import json

                data = json.loads(body)
                query = data.get("query", "")
        except Exception:
            pass

        # Get trace ID from OpenTelemetry
        span = trace.get_current_span()
        span_context = span.get_span_context() if span else None
        trace_id = (
            format(span_context.trace_id, "032x") if span_context else "no-trace"
        )

        # Client IP (masked)
        client_ip = AuditRecord.mask_ip(
            request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
        )

        # Execute request
        status = "success"
        error_detail = None
        response: Response | None = None

        try:
            response = await call_next(request)

            if response.status_code == 429:
                status = "rate_limited"
            elif response.status_code >= 400:
                status = "error"
                error_detail = f"HTTP {response.status_code}"

        except Exception as exc:
            status = "error"
            error_detail = str(exc)[:500]
            raise
        finally:
            duration_ms = int((time.time() - start_time) * 1000)

            record = AuditRecord(
                timestamp=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)
                ),
                trace_id=trace_id,
                app_id=app_id,
                user_id=user_id,
                user_groups=user_groups,
                action=action,
                query=query[:2000],  # Truncate for storage
                query_hash=AuditRecord.hash_query(query),
                response_length=0,  # Populated below if available
                steps_executed=[],  # Populated from FlowContext.metadata
                total_tokens=0,
                duration_ms=duration_ms,
                status=status,
                error_detail=error_detail,
                client_ip=client_ip,
            )

            # Fire-and-forget: don't block the response
            asyncio.create_task(_safe_audit_log(audit_logger, record))

        return response  # type: ignore[return-value]


async def _safe_audit_log(audit_logger: AuditLogger, record: AuditRecord) -> None:
    """Write audit record, catching all exceptions."""
    try:
        await audit_logger.log(record)
    except Exception:
        logger.exception("Failed to emit audit record for trace_id=%s", record.trace_id)
