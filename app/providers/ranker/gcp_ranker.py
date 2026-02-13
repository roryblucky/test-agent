"""GCP Ranking API implementation."""

from __future__ import annotations

from app.config.models import GCPConfig, RankingConfig
from app.core.http_client_pool import HttpClientPool
from app.models.domain import Document
from app.providers.base import BaseRankerProvider
from app.providers.factory import register_provider


@register_provider("ranker", "gcp")
class GCPRanker(BaseRankerProvider):
    """Re-ranker backed by GCP Ranking API (Discovery Engine)."""

    def __init__(
        self,
        config: RankingConfig,
        cloud_config: GCPConfig | None,
        http_pool: HttpClientPool,
    ) -> None:
        if cloud_config is None:
            raise ValueError("GCP ranker requires gcpConfig")
        self.config = config
        self.gcp_config = cloud_config
        self.http_client = http_pool.get("gcp")

    async def rank(
        self, query: str, documents: list[Document], top_n: int = 5
    ) -> list[Document]:
        """Re-rank documents using GCP Ranking API.

        TODO: Integrate with ``google.cloud.discoveryengine`` ranking service.
        """
        # Placeholder — replace with actual Ranking API call
        # from google.cloud import discoveryengine
        #
        # client = discoveryengine.RankServiceAsyncClient()
        # records = [discoveryengine.RankingRecord(id=d.id, content=d.content) for d in documents]
        # request = discoveryengine.RankRequest(
        #     ranking_config=f"projects/{self.gcp_config.project_id}/...",
        #     model=self.config.model,
        #     query=query,
        #     records=records,
        #     top_n=top_n or self.config.top_n,
        # )
        # response = await client.rank(request=request)
        # ...

        raise NotImplementedError(
            "GCP ranker: integrate with discoveryengine RankService SDK"
        )
