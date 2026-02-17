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
        self.config: ModerationConfig = config
        self.azure_config = cloud_config

        # Initialize native Azure transport
        # HttpClientPool manages the underlying aiohttp session
        self.transport = http_pool.get_azure_transport()

        # Using Service Principal auth as implied by AzureConfig fields
        from azure.identity.aio import ClientSecretCredential

        self.credential = ClientSecretCredential(
            tenant_id=self.azure_config.tenant_id,
            client_id=self.azure_config.client_id,
            client_secret=self.azure_config.client_secret,
            transport=self.transport,
        )

        from azure.ai.contentsafety.aio import ContentSafetyClient

        self.client = ContentSafetyClient(
            endpoint=self.azure_config.content_safety_endpoint,
            credential=self.credential,
            transport=self.transport,
        )

    async def check(self, text: str) -> ModerationResult:
        """Check text against Azure Content Safety."""
        from azure.ai.contentsafety.models import AnalyzeTextOptions
        from azure.core.exceptions import HttpResponseError

        request = AnalyzeTextOptions(text=text)

        try:
            response = await self.client.analyze_text(request)
        except HttpResponseError as e:
            # Return safe fallback or raise depending on requirements.
            # For now return flagged with error reason to be safe.
            return ModerationResult(
                is_flagged=True,
                reason=f"Azure Content Safety Error: {str(e)}",
                categories={},
            )

        categories = {}
        is_flagged = False

        # response.categories_analysis is expected to be a list
        if response.categories_analysis:
            for item in response.categories_analysis:
                # category is enum, severity is int usually 0, 2, 4, 6
                cat_name = getattr(item.category, "value", str(item.category))
                severity = item.severity or 0

                # Normalize severity (max 7 usually, but 0/2/4/6 common) to 0-1
                # 0=Safe, 2=Low, 4=Medium, 6=High
                score = min(severity / 6.0, 1.0)
                categories[cat_name] = score

                # Simple thresholding: >= 2 is flagged (Low severity or higher)
                # Config might have threshold string like "medium"
                threshold_val = 2  # Default Low
                if self.config.threshold == "medium":
                    threshold_val = 4
                elif self.config.threshold == "high":
                    threshold_val = 6

                if severity >= threshold_val:
                    is_flagged = True

        return ModerationResult(is_flagged=is_flagged, categories=categories)
