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
    async def retrieve(self, query: str, top_k: int = 10) -> list[Document]: ...


class BaseRankerProvider(ABC):
    """Re-rank a list of documents against a query."""

    @abstractmethod
    async def rank(
        self, query: str, documents: list[Document], top_n: int = 5
    ) -> list[Document]: ...


class BaseModerationProvider(ABC):
    """Check text for policy violations."""

    @abstractmethod
    async def check(self, text: str) -> ModerationResult: ...


class BaseGroundednessProvider(ABC):
    """Check whether an answer is grounded in the source documents."""

    @abstractmethod
    async def check(
        self, answer: str, context: list[Document]
    ) -> GroundednessResult: ...
