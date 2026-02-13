"""Redis-backed session store for conversation history.

Uses ``redis.asyncio`` for non-blocking I/O.  Each session is stored
as a JSON blob with a configurable TTL (default 1 hour).

Connection pooling is handled automatically by ``redis.asyncio`` —
the underlying ``ConnectionPool`` reuses TCP connections across calls.

Usage::

    from app.memory.redis_session_store import RedisSessionStore

    store = RedisSessionStore("redis://localhost:6379/0")
    # or with explicit options:
    store = RedisSessionStore(
        url="redis://redis-svc:6379/0",
        key_prefix="kms:session:",
        ttl=7200,
    )
"""

from __future__ import annotations

import redis.asyncio as aioredis

from app.memory.session_store import BaseSessionStore


class RedisSessionStore(BaseSessionStore):
    """Redis-backed session store with TTL-based expiry.

    Each session key is prefixed and set to expire after ``ttl`` seconds,
    preventing unbounded memory growth.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        key_prefix: str = "kms:session:",
        ttl: int = 3600,
        max_connections: int = 50,
    ) -> None:
        self._redis = aioredis.from_url(
            url,
            max_connections=max_connections,
            decode_responses=False,
        )
        self._prefix = key_prefix
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def _get_raw(self, session_id: str) -> bytes | None:
        data = await self._redis.get(self._key(session_id))
        return data  # bytes or None

    async def _set_raw(self, session_id: str, data: bytes) -> None:
        await self._redis.set(self._key(session_id), data, ex=self._ttl)

    async def close(self) -> None:
        """Close the Redis connection pool.  Call during app shutdown."""
        await self._redis.aclose()
