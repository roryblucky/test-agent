"""Instance-local ownership of directly executing LangGraph Runs."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


class RuntimeStopping(RuntimeError):
    """The local runtime no longer accepts new Run executions."""


class LocalRunRuntime:
    """Keep strong references to local Run tasks until they reach cleanup."""

    def __init__(self, *, shutdown_grace_seconds: float = 5.0) -> None:
        self._tasks: set[asyncio.Task[None]] = set()
        self._accepting = True
        self._shutdown_grace_seconds = shutdown_grace_seconds

    @property
    def accepting(self) -> bool:
        """Whether this instance can begin another directly executed Run."""
        return self._accepting

    @property
    def active_task_count(self) -> int:
        """Return the number of locally owned executions not yet cleaned up."""
        return len(self._tasks)

    def start(self, execution: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Start and strongly retain one local execution until it finishes."""
        if not self._accepting:
            close = getattr(execution, "close", None)
            if close is not None:
                close()
            raise RuntimeStopping("LangGraph v2 runtime is stopping")
        task = asyncio.create_task(execution)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def stop_and_wait_for_checkpoint_boundary(self) -> None:
        """Stop admission and allow active graph work a bounded completion window."""
        self._accepting = False
        if not self._tasks:
            return
        _, pending = await asyncio.wait(
            self._tasks,
            timeout=self._shutdown_grace_seconds,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
