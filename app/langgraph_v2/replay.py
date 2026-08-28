"""Tenant-scoped, snapshot-only replay of persisted v2 Events."""

from __future__ import annotations

from uuid import UUID

from app.langgraph_v2.run_events import EventRecord, RunEventRepository


class PersistedEventReplay:
    """Read one immutable Event snapshot without starting or following a Run."""

    def __init__(self, repository: RunEventRepository) -> None:
        self._repository = repository

    async def snapshot(
        self,
        *,
        tenant_id: str,
        run_id: UUID,
        after_sequence: int,
    ) -> list[EventRecord]:
        """Return only events published after the requested Run-local sequence."""
        return await self._repository.list_events_after(
            tenant_id,
            run_id,
            after_sequence=after_sequence,
        )
