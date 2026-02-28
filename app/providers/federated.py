"""Federated Retriever for Multi-Tenant Multi-Source Retrieval.

This module provides a unified `BaseRetrieverProvider` implementation that
wraps multiple underlying retrievers. When the agent calls `retrieve()`, it
concurrently queries all configured data sources for the tenant, merges
the results, deduplicates them, and returns a unified list of documents.

This pattern shields the LLM from the complexity of knowing which specific
database or API to query, while maximizing speed through concurrent execution.
"""

import asyncio
from typing import Sequence

from app.models.domain import Document
from app.providers.base import BaseRetrieverProvider


class FederatedRetrieverProvider(BaseRetrieverProvider):
    """A retriever that concurrently queries multiple configured sources.

    In a multi-tenant system, `TenantManager` would instantiate this class
    and pass in the specific `BaseRetrieverProvider` implementations configured
    for the current tenant (e.g., ConfluenceRetriever, VectorDBRetriever).
    """

    def __init__(self, retrievers: Sequence[BaseRetrieverProvider]):
        """Initialize with a list of active retrievers for the tenant.

        Args:
            retrievers: A list of instantiated provider objects to query.
        """
        self.retrievers = retrievers

    async def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Concurrently fetch from all underlying retrievers and merge results.

        Args:
            query: The search string.
            top_k: The number of documents requested per source.

        Returns:
            A deduplicated list of documents aggregated from all sources.
            Note: This list is typically sent to a `BaseRankerProvider`
            afterwards to establish a global ordering across sources.
        """
        if not self.retrievers:
            return []

        # 1. Fire off all retrieval tasks concurrently
        tasks = [
            retriever.retrieve(query, top_k=top_k) for retriever in self.retrievers
        ]

        # `return_exceptions=True` ensures that if one data source goes down
        # (e.g., a timeout connecting to Confluence), it doesn't crash the
        # entire retrieval process. We can elegantly handle partial failures.
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # 2. Merge and Deduplicate Results
        all_docs: list[Document] = []
        seen_ids: set[str] = set()

        for result in results_lists:
            # Skip if a specific provider raised an exception
            if isinstance(result, Exception):
                # In production, log the exception here:
                # logger.warning(f"A retriever failed: {result}")
                continue

            # result is a list[Document]
            for doc in result:
                # Deduplicate based on document ID.
                # If IDs aren't globally unique across sources, you might
                # want to prefix them with the source name (e.g., "confl:123").
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    all_docs.append(doc)

        return all_docs
