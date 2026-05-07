"""Per-tenant rate limiting middleware.

Intercepts API requests, resolves the tenant from ``X-Application-Id``,
and applies rate limits from the tenant's ``rateLimitConfig``.

Returns ``429 Too Many Requests`` with IETF-standard rate-limit headers
when limits are exceeded.
"""

from __future__ import annotations

import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Paths that bypass rate limiting
_EXEMPT_PREFIXES = ("/health", "/docs", "/openapi.json", "/admin")


class TenantRateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-tenant, per-endpoint rate limits before route handlers."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        """Check rate limits and either proceed or return 429."""
        path = request.url.path

        # Skip non-API or exempt paths
        if not path.startswith("/api/") or any(
            path.startswith(p) for p in _EXEMPT_PREFIXES
        ):
            return await call_next(request)

        # Extract tenant ID from header
        app_id = request.headers.get("x-application-id")
        if not app_id:
            # Let the route handler deal with missing header (returns 422)
            return await call_next(request)

        # Resolve tenant config
        tenant_manager = getattr(request.app.state, "tenant_manager", None)
        rate_limiter = getattr(request.app.state, "rate_limiter", None)
        if not tenant_manager or not rate_limiter:
            return await call_next(request)

        try:
            tenant_cfg = tenant_manager.get_tenant_config(app_id)
        except Exception:
            return await call_next(request)

        rl_config = tenant_cfg.rate_limit_config
        if not rl_config:
            return await call_next(request)

        # Determine endpoint type and corresponding policy
        is_stream = path.endswith("/stream")
        endpoint_type = "stream" if is_stream else "query"
        policy = rl_config.stream_policy if is_stream else rl_config.query_policy

        # 1. Check monthly token quota
        token_result = await rate_limiter.check_token_quota(
            app_id, rl_config.tokens_per_month
        )
        if not token_result.allowed:
            logger.warning(
                "Tenant %s exceeded monthly token quota (%d)",
                app_id,
                rl_config.tokens_per_month,
            )
            return _build_429(
                "Monthly token quota exceeded",
                token_result,
            )

        # 2. Check request rate (RPM / RPD)
        rate_result = await rate_limiter.check_request(
            app_id,
            endpoint_type,
            policy.requests_per_minute,
            policy.requests_per_day,
        )
        if not rate_result.allowed:
            logger.warning(
                "Tenant %s rate-limited on %s (remaining=0)",
                app_id,
                endpoint_type,
            )
            return _build_429("Rate limit exceeded", rate_result)

        # 3. Check concurrency
        concurrency_result = await rate_limiter.check_concurrency(
            app_id, endpoint_type, policy.concurrent_requests
        )
        if not concurrency_result.allowed:
            logger.warning(
                "Tenant %s concurrency limit on %s (%d)",
                app_id,
                endpoint_type,
                policy.concurrent_requests,
            )
            return _build_429(
                "Too many concurrent requests",
                concurrency_result,
            )

        # Execute request, then release concurrency slot
        try:
            response = await call_next(request)

            # Add rate-limit headers to successful responses
            response.headers["X-RateLimit-Limit"] = str(rate_result.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_result.remaining)
            response.headers["X-RateLimit-Reset"] = str(int(rate_result.reset_at))

            return response
        finally:
            await rate_limiter.release_concurrency(app_id, endpoint_type)


def _build_429(
    message: str,
    result: object,
) -> Response:
    """Build a 429 Too Many Requests response with standard headers."""
    import json

    from app.core.rate_limiter import RateLimitResult

    assert isinstance(result, RateLimitResult)

    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": str(int(result.reset_at)),
    }
    if result.retry_after is not None:
        headers["Retry-After"] = str(result.retry_after)

    body = json.dumps({"detail": message})
    return Response(
        content=body,
        status_code=429,
        media_type="application/json",
        headers=headers,
    )
