from __future__ import annotations

import asyncio

import pytest

from app.langgraph_v2.runtime import LocalRunRuntime, RuntimeStopping


@pytest.mark.asyncio
async def test_runtime_keeps_execution_alive_after_its_caller_releases_it() -> None:
    runtime = LocalRunRuntime()
    started = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()

    async def execution() -> None:
        started.set()
        await release.wait()
        completed.set()

    runtime.start(execution())
    await started.wait()
    assert runtime.active_task_count == 1

    release.set()
    for _ in range(20):
        if completed.is_set() and runtime.active_task_count == 0:
            break
        await asyncio.sleep(0)

    assert completed.is_set()
    assert runtime.active_task_count == 0


@pytest.mark.asyncio
async def test_runtime_stops_admission_after_a_bounded_grace_window() -> None:
    runtime = LocalRunRuntime(shutdown_grace_seconds=0)
    cancelled = asyncio.Event()

    async def execution() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    runtime.start(execution())
    await asyncio.sleep(0)
    await runtime.stop_and_wait_for_checkpoint_boundary()

    assert runtime.accepting is False
    assert cancelled.is_set()
    assert runtime.active_task_count == 0
    with pytest.raises(RuntimeStopping):
        runtime.start(asyncio.sleep(0))
