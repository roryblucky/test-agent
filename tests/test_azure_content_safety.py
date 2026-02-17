"""Tests for Azure Content Safety Moderator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import azure.ai.contentsafety  # Ensure namespace is loaded
from azure.core.exceptions import HttpResponseError

from app.config.models import AzureConfig, ModerationConfig
from app.core.http_client_pool import HttpClientPool
from app.providers.moderation.azure_content_safety import AzureContentSafetyModerator


@pytest.fixture
def mock_http_pool():
    pool = MagicMock(spec=HttpClientPool)
    pool.get.return_value = AsyncMock()  # Mock httpx client
    pool.get_azure_transport.return_value = AsyncMock()  # Mock azure transport
    return pool


@pytest.fixture
def mock_config():
    return ModerationConfig(provider="azure", threshold="medium")


@pytest.fixture
def mock_azure_config():
    return AzureConfig(
        tenantId="fake-tenant",
        clientId="fake-client",
        clientSecret="fake-secret",
        openAIEndpoint="https://fake.openai.azure.com",
        contentSafetyEndpoint="https://fake.contentsafety.azure.com",
        aiLanguageEndpoint="https://fake.language.azure.com",
    )


@patch("azure.identity.aio.ClientSecretCredential")
@patch("azure.ai.contentsafety.aio.ContentSafetyClient")
@pytest.mark.asyncio
async def test_azure_moderator_check(
    mock_client_cls, mock_cred_cls, mock_config, mock_azure_config, mock_http_pool
):
    """Test AzureContentSafetyModerator check method."""
    # Setup mocks
    mock_client_instance = AsyncMock()
    mock_client_cls.return_value = mock_client_instance

    # Mock response from analyze_text
    mock_response = MagicMock()
    # Mock categories_analysis items
    item1 = MagicMock()
    item1.category = "Hate"
    item1.severity = 0

    item2 = MagicMock()
    item2.category = "Violence"
    item2.severity = 4  # Medium -> Flagged if threshold is medium (4) or low (2)

    # items might use enum values, so ensure str() works or .value
    # In test we just mock attribute access

    mock_response.categories_analysis = [item1, item2]
    mock_client_instance.analyze_text.return_value = mock_response

    # Initialize
    moderator = AzureContentSafetyModerator(
        mock_config, mock_azure_config, mock_http_pool
    )

    # Verify transport was retrieved
    mock_http_pool.get_azure_transport.assert_called_once()

    # Run check
    result = await moderator.check("some text")

    # Verify
    assert result.is_flagged is True
    assert "Hate" in result.categories
    assert "Violence" in result.categories
    assert result.categories["Violence"] == 4.0 / 6.0


@patch("azure.identity.aio.ClientSecretCredential")
@patch("azure.ai.contentsafety.aio.ContentSafetyClient")
@pytest.mark.asyncio
async def test_azure_moderator_error(
    mock_client_cls, mock_cred_cls, mock_config, mock_azure_config, mock_http_pool
):
    """Test AzureContentSafetyModerator error handling."""
    mock_client_instance = AsyncMock()
    mock_client_cls.return_value = mock_client_instance

    # Raise error
    mock_client_instance.analyze_text.side_effect = HttpResponseError("API Error")

    moderator = AzureContentSafetyModerator(
        mock_config, mock_azure_config, mock_http_pool
    )

    result = await moderator.check("some text")

    assert result.is_flagged is True  # Fail closed
    assert "Error" in result.reason
