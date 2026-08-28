"""Adapters from existing tenant providers to v2 phase contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.langgraph_v2.pre_moderation import ModerationDecision, ModerationProvider
from app.langgraph_v2.reranking import Ranker, RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult, Retriever
from app.models.domain import Document
from app.providers.base import (
    BaseGroundednessProvider,
    BaseModerationProvider,
    BaseRankerProvider,
    BaseRetrieverProvider,
)


class V2RetrieverAdapter:
    """Adapt a tenant's existing retriever to the v2 retrieval contract."""

    def __init__(self, provider: BaseRetrieverProvider, *, top_k: int = 10) -> None:
        self._provider = provider
        self._top_k = top_k

    async def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve Documents and attach stable provider provenance."""
        documents = await self._provider.retrieve(query, top_k=self._top_k)
        return RetrievalResult(
            documents=documents,
            raw_payload={
                "provider": type(self._provider).__name__,
                "query": query,
                "document_count": len(documents),
            },
        )


class V2RankerAdapter:
    """Adapt a tenant's existing query-aware ranker to the v2 contract."""

    def __init__(self, provider: BaseRankerProvider, *, top_n: int = 5) -> None:
        self._provider = provider
        self._top_n = top_n

    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        """Rank Documents through the existing query-aware provider."""
        ranked = await self._provider.rank(query, documents, top_n=self._top_n)
        return RerankingResult(documents=ranked)


class V2ModerationAdapter:
    """Adapt an existing moderation result to the v2 phase decision."""

    def __init__(self, provider: BaseModerationProvider) -> None:
        self._provider = provider

    async def check(self, text: str) -> ModerationDecision:
        """Convert the existing provider's result to the v2 decision model."""
        result = await self._provider.check(text)
        return ModerationDecision(
            is_flagged=result.is_flagged,
            categories=result.categories,
            reason=result.reason,
        )


class TenantProvidersLike(Protocol):
    """Provider bundle shape exposed by TenantManager without legacy imports."""

    retriever: BaseRetrieverProvider | None
    ranker: BaseRankerProvider | None
    moderation: BaseModerationProvider | None
    groundedness: BaseGroundednessProvider | None


@dataclass(frozen=True)
class V2ProviderBundle:
    """Tenant-scoped v2 adapters used to construct a graph."""

    retriever: Retriever | None = None
    ranker: Ranker | None = None
    moderation: ModerationProvider | None = None


def adapt_tenant_providers(providers: TenantProvidersLike) -> V2ProviderBundle:
    """Convert existing tenant providers without importing legacy orchestration."""
    return V2ProviderBundle(
        retriever=(
            V2RetrieverAdapter(providers.retriever)
            if providers.retriever is not None
            else None
        ),
        ranker=(
            V2RankerAdapter(providers.ranker)
            if providers.ranker is not None
            else None
        ),
        moderation=(
            V2ModerationAdapter(providers.moderation)
            if providers.moderation is not None
            else None
        ),
    )
