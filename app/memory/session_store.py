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
    """Dict-backed session store — suitable for dev/testing only."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def _get_raw(self, session_id: str) -> bytes | None:
        return self._store.get(session_id)

    async def _set_raw(self, session_id: str, data: bytes) -> None:
        self._store[session_id] = data
