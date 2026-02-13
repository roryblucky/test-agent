"""Shared HTTP client pool for cloud provider connections.

Manages :class:`httpx.AsyncClient` instances keyed by provider name,
enabling TCP connection reuse across all components within a tenant.
Lifecycle is tied to the FastAPI application lifespan.
"""

from __future__ import annotations

import httpx


class HttpClientPool:
    """Manages shared ``httpx.AsyncClient`` instances per cloud provider."""

    def __init__(self) -> None:
        self._clients: dict[str, httpx.AsyncClient] = {}

    def get(
        self,
        provider: str,
        *,
        timeout: float = 60.0,
        max_connections: int = 100,
        max_keepalive: int = 20,
        proxy_url: str | None = None,
    ) -> httpx.AsyncClient:
        """Get or create a shared HTTP client for *provider*.

        The client is created lazily on first access and reused thereafter.
        """
        if provider not in self._clients:
            self._clients[provider] = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive,
                ),
                proxy=proxy_url,
            )
        return self._clients[provider]

    async def close_all(self) -> None:
        """Close all managed HTTP clients.  Call during app shutdown."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
