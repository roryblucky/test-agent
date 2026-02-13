"""GCP Vertex AI Search retriever implementation."""

from __future__ import annotations

from app.config.models import GCPConfig, RetrieverConfig
from app.core.http_client_pool import HttpClientPool
from app.models.domain import Document
from app.providers.base import BaseRetrieverProvider
from app.providers.factory import register_provider


@register_provider("retriever", "gcp")
class GCPVertexSearchRetriever(BaseRetrieverProvider):
    """Retriever backed by GCP Vertex AI Search (Discovery Engine)."""

    def __init__(
        self,
        config: RetrieverConfig,
        cloud_config: GCPConfig | None,
        http_pool: HttpClientPool,
    ) -> None:
        if cloud_config is None:
            raise ValueError("GCP retriever requires gcpConfig")
        self.config = config
        self.gcp_config = cloud_config
        self.http_client = http_pool.get("gcp")

    async def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Search documents using Vertex AI Search API.

        TODO: Integrate with ``google.cloud.discoveryengine`` SDK.
        """
        # Placeholder — replace with actual Vertex AI Search SDK call
        # from google.cloud import discoveryengine
        #
        # client = discoveryengine.SearchServiceAsyncClient()
        # request = discoveryengine.SearchRequest(
        #     serving_config=f"projects/{self.gcp_config.project_id}/...",
        #     query=query,
        #     page_size=top_k or self.config.top_k,
        # )
        # response = await client.search(request=request)
        # return [Document(id=r.id, content=r.document.derived_struct_data["extractive_answers"][0]["content"], ...) for r in response.results]

        raise NotImplementedError(
            "GCP Vertex AI Search retriever: integrate with discoveryengine SDK"
        )
