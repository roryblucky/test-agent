"""Abstract base classes for non-LLM providers.

LLM providers are handled by :mod:`app.core.model_registry` using pydantic-ai
directly.  These bases cover retrieval, ranking, moderation, and groundedness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.domain import Document, GroundednessResult, ModerationResult


class BaseRetrieverProvider(ABC):
    """Retrieve documents relevant to a query."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Retrieve relevant documents for a query."""
        ...


class BaseRankerProvider(ABC):
    """Re-rank a list of documents against a query."""

    @abstractmethod
    async def rank(
        self, query: str, documents: list[Document], top_n: int = 5
    ) -> list[Document]:
        """Rank documents by relevance to a query."""
        ...


class BaseModerationProvider(ABC):
    """Check text for policy violations."""

    @abstractmethod
    async def check(self, text: str) -> ModerationResult:
        """Check text for moderation violations."""
        ...


class BaseGroundednessProvider(ABC):
    """Check whether an answer is grounded in the source documents."""

    @abstractmethod
    async def check(self, answer: str, context: list[Document]) -> GroundednessResult:
        """Check if an answer is grounded in the provided context."""
        ...


class TenantProvidersProtocol(ABC):
    """Protocol for the collection of tenant providers.

    Used to avoid circular imports where consumers (FlowEngine, Orchestrator)
    need to reference the provider collection defined in TenantManager.
    """

    retriever: BaseRetrieverProvider | None
    ranker: BaseRankerProvider | None
    moderation: BaseModerationProvider | None
    groundedness: BaseGroundednessProvider | None
