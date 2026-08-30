"""Tests for HttpClientPool."""

import unittest.mock
from unittest.mock import AsyncMock, patch

import pytest

from app.core.http_client_pool import HttpClientPool


@patch("app.core.http_client_pool.ClientSession")
@patch("app.core.http_client_pool.AioHttpTransport")
def test_get_azure_transport_lazy_init(
    mock_transport_cls: unittest.mock.MagicMock,
    mock_session_cls: unittest.mock.MagicMock,
):
    """Test lazy initialization of Azure transport."""
    pool = HttpClientPool()

    assert pool.has_azure_session is False

    # Configure mock session to not look closed
    mock_session_instance = mock_session_cls.return_value
    mock_session_instance.closed = False

    # First call
    pool.get_azure_transport()
    mock_session_cls.assert_called_once()
    assert pool.has_azure_session is True
    mock_transport_cls.assert_called_with(
        session=mock_session_instance, session_owner=False
    )

    # Second call - reuses session
    t2 = pool.get_azure_transport()
    mock_session_cls.assert_called_once()  # Still called once
    assert t2 is not None


@pytest.mark.asyncio
@patch("app.core.http_client_pool.ClientSession")
async def test_close_all_closes_azure_session(
    mock_session_cls: unittest.mock.MagicMock,
) -> None:
    """Test close_all closes the azure session."""
    pool = HttpClientPool()
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session_cls.return_value = mock_session
    pool.get_azure_transport()

    await pool.close_all()

    mock_session.close.assert_awaited_once()
