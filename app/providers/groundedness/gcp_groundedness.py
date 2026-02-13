"""GCP groundedness checking implementation."""

from __future__ import annotations

from app.config.models import GCPConfig, GroundednessConfig
from app.core.http_client_pool import HttpClientPool
from app.models.domain import Document, GroundednessResult
from app.providers.base import BaseGroundednessProvider
from app.providers.factory import register_provider


@register_provider("groundedness", "gcp")
class GCPGroundednessChecker(BaseGroundednessProvider):
    """Groundedness checker backed by GCP Vertex AI Grounding API."""

    def __init__(
        self,
        config: GroundednessConfig,
        cloud_config: GCPConfig | None,
        http_pool: HttpClientPool,
    ) -> None:
        if cloud_config is None:
            raise ValueError("GCP groundedness checker requires gcpConfig")
        self.config = config
        self.gcp_config = cloud_config
        self.http_client = http_pool.get("gcp")

    async def check(self, answer: str, context: list[Document]) -> GroundednessResult:
        """Check answer groundedness against source documents.

        TODO: Integrate with GCP grounding / fact-checking API.
        """
        # Placeholder — replace with actual GCP grounding API call
        # This could use Vertex AI's grounding feature or a custom
        # evaluation pipeline.

        raise NotImplementedError(
            "GCP groundedness checker: integrate with Vertex AI grounding API"
        )
