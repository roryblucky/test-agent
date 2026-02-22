"""Long-term memory abstraction.

Defines a pluggable interface for cross-session user knowledge.
Unlike session memory (which stores raw message history),
long-term memory captures *distilled facts* about users, topics,
or organisational knowledge that persist indefinitely.

This is **not** natively supported by pydantic-ai, so we define
our own ABC.  Concrete implementations can integrate with:

- ``mem0`` (https://github.com/mem0ai/mem0)
- GCP Vertex AI Feature Store
- Redis + vector search
- PostgreSQL + pgvector
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """A single long-term memory item."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 1.0  # relevance score when retrieved


class BaseLongTermMemory(ABC):
    """Abstract base class for long-term memory providers.

    Implementations should handle storage, retrieval, and lifecycle
    of user/org knowledge that persists across all sessions.
    """

    @abstractmethod
    async def add(
        self,
        content: str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory entry.  Returns the entry ID."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Retrieve relevant memories for *query*."""

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """Delete a specific memory entry."""

    @abstractmethod
    async def get_all(
        self,
        *,
        user_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Return all memories, optionally filtered by user."""


class NoOpLongTermMemory(BaseLongTermMemory):
    """Stub implementation that does nothing.

    Use as a default when no long-term memory backend is configured.
    """

    async def add(
        self,
        content: str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a memory (not implemented)."""
        return ""

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Search memories (not implemented)."""
        return []

    async def delete(self, memory_id: str) -> None:
        """Delete a memory by its ID."""

    async def get_all(
        self,
        *,
        user_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Get all memories (not implemented)."""
        return []
