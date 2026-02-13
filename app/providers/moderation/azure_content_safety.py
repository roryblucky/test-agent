"""Azure Content Safety moderation implementation."""

from __future__ import annotations

from app.config.models import AzureConfig, ModerationConfig
from app.core.http_client_pool import HttpClientPool
from app.models.domain import ModerationResult
from app.providers.base import BaseModerationProvider
from app.providers.factory import register_provider


@register_provider("moderation", "azure")
class AzureContentSafetyModerator(BaseModerationProvider):
    """Content moderation backed by Azure Content Safety API."""

    def __init__(
        self,
        config: ModerationConfig,
        cloud_config: AzureConfig | None,
        http_pool: HttpClientPool,
    ) -> None:
        if cloud_config is None:
            raise ValueError("Azure moderation requires azureConfig")
        self.config = config
        self.azure_config = cloud_config
        self.http_client = http_pool.get("azure")

    async def check(self, text: str) -> ModerationResult:
        """Check text against Azure Content Safety.

        TODO: Integrate with ``azure.ai.contentsafety`` SDK.
        """
        # Placeholder — replace with actual Azure Content Safety SDK call
        # from azure.ai.contentsafety import ContentSafetyClient
        # from azure.ai.contentsafety.models import AnalyzeTextOptions
        #
        # client = ContentSafetyClient(
        #     endpoint=self.azure_config.content_safety_endpoint,
        #     credential=...,
        # )
        # request = AnalyzeTextOptions(text=text, categories=self.config.categories)
        # response = client.analyze_text(request)
        # ...

        raise NotImplementedError(
            "Azure Content Safety moderation: integrate with azure-ai-contentsafety SDK"
        )
