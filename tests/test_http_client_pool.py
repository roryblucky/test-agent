"""Tests for HttpClientPool."""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.http_client_pool import HttpClientPool


@patch("app.core.http_client_pool.ClientSession")
@patch("app.core.http_client_pool.AioHttpTransport")
def test_get_azure_transport_lazy_init(mock_transport_cls, mock_session_cls):
    """Test lazy initialization of Azure transport."""
    pool = HttpClientPool()

    assert pool._aiohttp_session is None

    # Configure mock session to not look closed
    mock_session_instance = mock_session_cls.return_value
    mock_session_instance.closed = False

    # First call
    pool.get_azure_transport()
    mock_session_cls.assert_called_once()
    assert pool._aiohttp_session == mock_session_instance
    mock_transport_cls.assert_called_with(
        session=pool._aiohttp_session, session_owner=False
    )

    # Second call - reuses session
    t2 = pool.get_azure_transport()
    mock_session_cls.assert_called_once()  # Still called once
    assert t2 is not None


@pytest.mark.asyncio
async def test_close_all_closes_azure_session():
    """Test close_all closes the azure session."""
    pool = HttpClientPool()
    mock_session = AsyncMock()
    mock_session.closed = False
    pool._aiohttp_session = mock_session

    await pool.close_all()

    mock_session.close.assert_awaited_once()
