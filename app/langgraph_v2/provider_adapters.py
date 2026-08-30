"""Adapters from existing tenant providers to v2 phase contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.langgraph_v2.groundedness import (
    GroundednessActor,
    GroundednessAssessment,
)
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

    def __init__(
        self, provider: BaseRetrieverProvider, *, top_k: int | None = None
    ) -> None:
        self._provider = provider
        self._top_k = top_k

    async def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve Documents and attach stable provider provenance."""
        if self._top_k is None:
            documents = await self._provider.retrieve_configured(query)
        else:
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

    def __init__(
        self, provider: BaseRankerProvider, *, top_n: int | None = None
    ) -> None:
        self._provider = provider
        self._top_n = top_n

    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        """Rank Documents through the existing query-aware provider."""
        if self._top_n is None:
            ranked = await self._provider.rank(query, documents)
        else:
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


class V2GroundednessAdapter:
    """Adapt a tenant groundedness provider to the v2 actor contract."""

    def __init__(self, provider: BaseGroundednessProvider) -> None:
        self._provider = provider

    async def evaluate(
        self, answer: str, documents: list[Document]
    ) -> GroundednessAssessment:
        """Evaluate groundedness through the tenant-configured provider."""
        result = await self._provider.check(answer, documents)
        return GroundednessAssessment(**result.model_dump(), usage={})


class TenantProvidersLike(Protocol):
    """Provider bundle shape exposed by TenantManager without legacy imports."""

    @property
    def retriever(self) -> BaseRetrieverProvider | None:
        """Return the configured retrieval provider."""
        ...

    @property
    def ranker(self) -> BaseRankerProvider | None:
        """Return the configured ranking provider."""
        ...

    @property
    def moderation(self) -> BaseModerationProvider | None:
        """Return the configured moderation provider."""
        ...

    @property
    def groundedness(self) -> BaseGroundednessProvider | None:
        """Return the configured groundedness provider."""
        ...


class MissingRetriever:
    """Explicit failure provider for a tenant without retrieval configuration."""

    async def retrieve(self, query: str) -> RetrievalResult:
        """Raise a clear configuration error instead of fabricating data."""
        del query
        raise RuntimeError("retriever provider is not configured for this tenant")


class MissingRanker:
    """Explicit failure provider for a tenant without ranking configuration."""

    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        """Raise a clear configuration error instead of fabricating ranking."""
        del query, documents
        raise RuntimeError("ranker provider is not configured for this tenant")


class MissingModeration:
    """Explicit failure provider for a tenant without moderation configuration."""

    async def check(self, text: str) -> ModerationDecision:
        """Raise a clear configuration error instead of allowing unsafe input."""
        del text
        raise RuntimeError("moderation provider is not configured for this tenant")


@dataclass(frozen=True)
class V2ProviderBundle:
    """Tenant-scoped v2 adapters used to construct a graph."""

    retriever: Retriever | None = None
    ranker: Ranker | None = None
    moderation: ModerationProvider | None = None
    groundedness: GroundednessActor | None = None


def adapt_tenant_providers(
    providers: TenantProvidersLike,
    *,
    ranker_top_n: int | None = None,
) -> V2ProviderBundle:
    """Convert existing tenant providers without importing legacy orchestration."""
    return V2ProviderBundle(
        retriever=(
            V2RetrieverAdapter(providers.retriever)
            if providers.retriever is not None
            else None
        ),
        ranker=(
            V2RankerAdapter(providers.ranker, top_n=ranker_top_n)
            if providers.ranker is not None
            else None
        ),
        moderation=(
            V2ModerationAdapter(providers.moderation)
            if providers.moderation is not None
            else None
        ),
        groundedness=(
            V2GroundednessAdapter(providers.groundedness)
            if providers.groundedness is not None
            else None
        ),
    )
