"""Per-tenant rate limiting with sliding window algorithm.

Supports two backends:
- ``RedisRateLimiter``   — production (atomic via Lua scripts)
- ``InMemoryRateLimiter`` — local development

Rate limits are checked per (app_id, endpoint_type) pair.
Token quotas are tracked per (app_id, month).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Outcome of a rate-limit check."""

    allowed: bool
    remaining: int
    limit: int
    reset_at: float  # Unix timestamp
    retry_after: int | None = None  # seconds, only set when denied


class BaseRateLimiter(ABC):
    """Abstract rate limiter interface."""

    @abstractmethod
    async def check_request(
        self,
        app_id: str,
        endpoint_type: str,
        rpm_limit: int,
        rpd_limit: int,
    ) -> RateLimitResult:
        """Check and consume a request against per-minute and per-day limits."""

    @abstractmethod
    async def check_concurrency(
        self,
        app_id: str,
        endpoint_type: str,
        max_concurrent: int,
    ) -> RateLimitResult:
        """Check concurrent request count. Call release_concurrency on completion."""

    @abstractmethod
    async def release_concurrency(
        self, app_id: str, endpoint_type: str
    ) -> None:
        """Release a concurrency slot after request completes."""

    @abstractmethod
    async def record_token_usage(self, app_id: str, tokens: int) -> None:
        """Record token consumption for monthly quota tracking."""

    @abstractmethod
    async def check_token_quota(
        self, app_id: str, monthly_limit: int
    ) -> RateLimitResult:
        """Check if the tenant has exceeded their monthly token quota."""


class InMemoryRateLimiter(BaseRateLimiter):
    """Dict-backed sliding window rate limiter for local development."""

    def __init__(self) -> None:
        # Sliding window: deque of timestamps per (app_id, endpoint, window)
        self._minute_windows: dict[str, deque[float]] = defaultdict(deque)
        self._day_windows: dict[str, deque[float]] = defaultdict(deque)
        self._concurrent: dict[str, int] = defaultdict(int)
        self._token_usage: dict[str, int] = defaultdict(int)  # key: app_id:YYYY-MM

    async def check_request(
        self,
        app_id: str,
        endpoint_type: str,
        rpm_limit: int,
        rpd_limit: int,
    ) -> RateLimitResult:
        """Check sliding window for per-minute and per-day limits."""
        now = time.time()
        minute_key = f"{app_id}:{endpoint_type}:rpm"
        day_key = f"{app_id}:{endpoint_type}:rpd"

        # Clean expired entries
        minute_window = self._minute_windows[minute_key]
        cutoff_minute = now - 60
        while minute_window and minute_window[0] < cutoff_minute:
            minute_window.popleft()

        day_window = self._day_windows[day_key]
        cutoff_day = now - 86400
        while day_window and day_window[0] < cutoff_day:
            day_window.popleft()

        # Check limits
        if len(minute_window) >= rpm_limit:
            reset_at = minute_window[0] + 60
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=rpm_limit,
                reset_at=reset_at,
                retry_after=max(1, int(reset_at - now)),
            )

        if len(day_window) >= rpd_limit:
            reset_at = day_window[0] + 86400
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=rpd_limit,
                reset_at=reset_at,
                retry_after=max(1, int(reset_at - now)),
            )

        # Consume
        minute_window.append(now)
        day_window.append(now)

        remaining = min(rpm_limit - len(minute_window), rpd_limit - len(day_window))
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=rpm_limit,
            reset_at=now + 60,
        )

    async def check_concurrency(
        self,
        app_id: str,
        endpoint_type: str,
        max_concurrent: int,
    ) -> RateLimitResult:
        """Check concurrent request slots."""
        key = f"{app_id}:{endpoint_type}"
        current = self._concurrent[key]

        if current >= max_concurrent:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=max_concurrent,
                reset_at=time.time() + 1,
                retry_after=1,
            )

        self._concurrent[key] = current + 1
        return RateLimitResult(
            allowed=True,
            remaining=max_concurrent - current - 1,
            limit=max_concurrent,
            reset_at=time.time() + 1,
        )

    async def release_concurrency(
        self, app_id: str, endpoint_type: str
    ) -> None:
        """Release a concurrency slot."""
        key = f"{app_id}:{endpoint_type}"
        self._concurrent[key] = max(0, self._concurrent[key] - 1)

    async def record_token_usage(self, app_id: str, tokens: int) -> None:
        """Record token consumption for the current month."""
        month_key = f"{app_id}:{datetime.now(tz=UTC).strftime('%Y-%m')}"
        self._token_usage[month_key] += tokens

    async def check_token_quota(
        self, app_id: str, monthly_limit: int
    ) -> RateLimitResult:
        """Check monthly token quota."""
        month_key = f"{app_id}:{datetime.now(tz=UTC).strftime('%Y-%m')}"
        used = self._token_usage.get(month_key, 0)
        remaining = max(0, monthly_limit - used)

        if used >= monthly_limit:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=monthly_limit,
                reset_at=time.time(),  # Resets at month boundary
                retry_after=None,
            )

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=monthly_limit,
            reset_at=time.time(),
        )


class RedisRateLimiter(BaseRateLimiter):
    """Redis-backed sliding window rate limiter for production.

    Uses sorted sets for sliding windows and atomic Lua scripts.
    """

    # Lua script: sliding window check-and-consume (atomic)
    _SLIDING_WINDOW_LUA = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local cutoff = now - window

    redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
    local count = redis.call('ZCARD', key)

    if count >= limit then
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local reset_at = oldest[2] and (tonumber(oldest[2]) + window) or (now + window)
        return {0, count, reset_at}
    end

    redis.call('ZADD', key, now, now .. ':' .. math.random(1000000))
    redis.call('EXPIRE', key, math.ceil(window) + 1)
    return {1, count + 1, now + window}
    """

    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._script_sha: str | None = None

    async def _ensure_script(self) -> str:
        """Load the Lua script into Redis and cache its SHA."""
        if self._script_sha is None:
            self._script_sha = await self._redis.script_load(
                self._SLIDING_WINDOW_LUA
            )
        return self._script_sha

    async def check_request(
        self,
        app_id: str,
        endpoint_type: str,
        rpm_limit: int,
        rpd_limit: int,
    ) -> RateLimitResult:
        """Atomic sliding window check via Lua script."""
        sha = await self._ensure_script()
        now = time.time()

        # Check per-minute
        rpm_key = f"ratelimit:{app_id}:{endpoint_type}:rpm"
        rpm_result = await self._redis.evalsha(
            sha, 1, rpm_key, now, 60, rpm_limit
        )

        if rpm_result[0] == 0:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=rpm_limit,
                reset_at=float(rpm_result[2]),
                retry_after=max(1, int(float(rpm_result[2]) - now)),
            )

        # Check per-day
        rpd_key = f"ratelimit:{app_id}:{endpoint_type}:rpd"
        rpd_result = await self._redis.evalsha(
            sha, 1, rpd_key, now, 86400, rpd_limit
        )

        if rpd_result[0] == 0:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=rpd_limit,
                reset_at=float(rpd_result[2]),
                retry_after=max(1, int(float(rpd_result[2]) - now)),
            )

        remaining = min(
            rpm_limit - int(rpm_result[1]),
            rpd_limit - int(rpd_result[1]),
        )
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=rpm_limit,
            reset_at=now + 60,
        )

    async def check_concurrency(
        self,
        app_id: str,
        endpoint_type: str,
        max_concurrent: int,
    ) -> RateLimitResult:
        """Atomic concurrency check using Redis INCR."""
        key = f"ratelimit:{app_id}:{endpoint_type}:concurrent"
        current = await self._redis.incr(key)
        # Set TTL as safety net (auto-expire if release never called)
        await self._redis.expire(key, 300)

        if current > max_concurrent:
            await self._redis.decr(key)
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=max_concurrent,
                reset_at=time.time() + 1,
                retry_after=1,
            )

        return RateLimitResult(
            allowed=True,
            remaining=max_concurrent - current,
            limit=max_concurrent,
            reset_at=time.time() + 1,
        )

    async def release_concurrency(
        self, app_id: str, endpoint_type: str
    ) -> None:
        """Decrement the concurrency counter."""
        key = f"ratelimit:{app_id}:{endpoint_type}:concurrent"
        await self._redis.decr(key)

    async def record_token_usage(self, app_id: str, tokens: int) -> None:
        """Atomically increment monthly token counter in Redis."""
        month = datetime.now(tz=UTC).strftime("%Y-%m")
        key = f"ratelimit:{app_id}:tokens:{month}"
        await self._redis.incrby(key, tokens)
        # Expire after ~35 days (covers the full month + buffer)
        await self._redis.expire(key, 35 * 86400)

    async def check_token_quota(
        self, app_id: str, monthly_limit: int
    ) -> RateLimitResult:
        """Check monthly token usage from Redis."""
        month = datetime.now(tz=UTC).strftime("%Y-%m")
        key = f"ratelimit:{app_id}:tokens:{month}"
        used = int(await self._redis.get(key) or 0)
        remaining = max(0, monthly_limit - used)

        if used >= monthly_limit:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=monthly_limit,
                reset_at=time.time(),
            )

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=monthly_limit,
            reset_at=time.time(),
        )

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._redis.aclose()


def create_rate_limiter(redis_url: str | None = None) -> BaseRateLimiter:
    """Factory: create the appropriate rate limiter backend.

    Uses Redis if *redis_url* is provided, otherwise falls back to
    in-memory for local development.
    """
    if redis_url:
        logger.info("Using Redis rate limiter: %s", redis_url)
        return RedisRateLimiter(redis_url)
    logger.info("Using in-memory rate limiter (dev mode)")
    return InMemoryRateLimiter()
