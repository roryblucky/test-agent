"""Session-scoped conversation memory.

Stores and retrieves pydantic-ai message histories per session,
enabling multi-turn conversations.  Serialisation / deserialisation
is handled entirely by pydantic-ai's ``ModelMessagesTypeAdapter``.

Implementations
---------------
- ``InMemorySessionStore``  – dictionary-backed, for dev / testing
- Swap in Redis / Firestore / etc. by subclassing ``BaseSessionStore``
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage
from pydantic_core import to_json


class BaseSessionStore(ABC):
    """Abstract session store for conversation history.

    Concrete subclasses only need to implement ``_get_raw`` / ``_set_raw``
    with whatever backend they use (Redis, Firestore, …).
    The public ``get`` / ``save`` methods handle pydantic-ai
    serialisation automatically.
    """

    # -- public API (uses pydantic-ai's ModelMessagesTypeAdapter) ---------

    async def get(self, session_id: str) -> list[ModelMessage]:
        """Retrieve message history for *session_id*, or ``[]``."""
        raw = await self._get_raw(session_id)
        if raw is None:
            return []
        return ModelMessagesTypeAdapter.validate_json(raw)

    async def save(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Persist *messages* for *session_id*."""
        raw = to_json(messages)
        await self._set_raw(session_id, raw)

    # -- backend hooks ---------------------------------------------------

    @abstractmethod
    async def _get_raw(self, session_id: str) -> bytes | None:
        """Return raw JSON bytes, or ``None`` if no history exists."""

    @abstractmethod
    async def _set_raw(self, session_id: str, data: bytes) -> None:
        """Store raw JSON bytes."""


class InMemorySessionStore(BaseSessionStore):
    """Dict-backed session store with LRU eviction — suitable for dev/testing only."""

    def __init__(self, max_sessions: int = 10000) -> None:
        self._store: dict[str, bytes] = {}
        self._max_sessions = max_sessions

    async def _get_raw(self, session_id: str) -> bytes | None:
        if session_id in self._store:
            # Move to end (most recently used)
            value = self._store.pop(session_id)
            self._store[session_id] = value
            return value
        return None

    async def _set_raw(self, session_id: str, data: bytes) -> None:
        if session_id in self._store:
            del self._store[session_id]
        elif len(self._store) >= self._max_sessions:
            # Evict least recently used (first item)
            first_key = next(iter(self._store))
            del self._store[first_key]
        self._store[session_id] = data
