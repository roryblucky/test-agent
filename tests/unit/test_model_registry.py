"""Model transport configuration coverage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from pydantic_ai.providers.azure import AzureProvider

from app.config.models import ModelConfig
from app.core.http_client_pool import HttpClientPool
from app.core.model_registry import (
    _build_azure_model,  # pyright: ignore[reportPrivateUsage]
)


class _HttpPool:
    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self.http_client = http_client

    def get(self, name: str) -> httpx.AsyncClient:
        assert name == "azure"
        return self.http_client


@pytest.mark.asyncio
async def test_azure_model_disables_sdk_transport_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-01")
    http_client = httpx.AsyncClient()

    try:
        model = _build_azure_model(
            ModelConfig(provider="azure", model_name="gpt-4o"),
            {
                "azure": SimpleNamespace(
                    openai_endpoint="https://example.openai.azure.com",
                    client_secret="test-key",
                )
            },
            cast(HttpClientPool, _HttpPool(http_client)),
        )
    finally:
        await http_client.aclose()

    assert isinstance(model.provider, AzureProvider)
    assert model.provider.client.max_retries == 0
