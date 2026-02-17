"""Shared HTTP client pool for cloud provider connections.

Manages :class:`httpx.AsyncClient` instances keyed by provider name,
enabling TCP connection reuse across all components within a tenant.
Lifecycle is tied to the FastAPI application lifespan.
"""

import httpx

try:
    from aiohttp import ClientSession
    from azure.core.pipeline.transport import AioHttpTransport, AsyncHttpTransport
except ImportError:
    ClientSession = None
    AioHttpTransport = None
    AsyncHttpTransport = None


class HttpClientPool:
    """Manages shared ``httpx.AsyncClient`` instances per cloud provider."""

    def __init__(self) -> None:
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._aiohttp_session: ClientSession | None = None

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

    def get_azure_transport(self) -> AsyncHttpTransport:
        """Get a shared Azure transport (AioHttpTransport).

        Lazily creates an underlying aiohttp.ClientSession if needed.
        """
        if AioHttpTransport is None:
            raise ImportError(
                "azure-core and aiohttp are required for Azure transport."
            )

        if self._aiohttp_session is None or self._aiohttp_session.closed:
            # Create shared session
            # Note: We can configure session limits here if needed via kwargs
            # roughly matching get() defaults if desired, but defaults are usually fine
            self._aiohttp_session = ClientSession()

        # Return a transport that uses the shared session but doesn't close it
        return AioHttpTransport(
            session=self._aiohttp_session,
            session_owner=False,
        )

    async def close_all(self) -> None:
        """Close all managed HTTP clients.  Call during app shutdown."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()
