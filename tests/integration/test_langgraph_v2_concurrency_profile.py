from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient, Response

from tests.integration.test_langgraph_v2_tracer import (
    persistent_tracer_app,
    seed_subject_conversation,
)

pytestmark = pytest.mark.skipif(
    os.getenv("LANGGRAPH_V2_WARMED_PROFILE") != "1",
    reason="set LANGGRAPH_V2_WARMED_PROFILE=1 to run the warmed concurrency profile",
)


class _EntryBarrierGraph:
    def __init__(self) -> None:
        self.target = 1
        self.entered = 0
        self.all_entered = asyncio.Event()

    def reset(self, target: int) -> None:
        self.target = target
        self.entered = 0
        self.all_entered = asyncio.Event()

    def astream(self, state: object | None, **options: Any) -> AsyncIterator[object]:
        del state, options

        async def stream() -> AsyncIterator[object]:
            self.entered += 1
            if self.entered == self.target:
                self.all_entered.set()
            await self.all_entered.wait()
            yield (
                "custom",
                {
                    "type": "done",
                    "data": {"answer": "profile answer"},
                    "checkpoint_terminal": True,
                },
            )
            yield ("updates", {"finalization": {"final_response": {}}})

        return stream()


@pytest.mark.asyncio
async def test_fifty_warmed_query_streams_enter_graph_without_application_queue(
    langgraph_v2_migrated_database_url: str,
) -> None:
    concurrency = 50
    graph = _EntryBarrierGraph()
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, graph)
    headers = {"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"}

    async with app.router.lifespan_context(app):
        for index in range(concurrency + 1):
            await seed_subject_conversation(
                app.state.langgraph_v2_postgres_pool,
                f"profile-{index}",
            )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://profile.test",
        ) as client:
            warm_response = await client.post(
                "/v2/query/stream",
                json={"query": "profile", "sessionId": f"profile-{concurrency}"},
                headers=headers,
            )
            assert warm_response.status_code == 200
            assert graph.entered == 1

            graph.reset(concurrency)
            tasks: list[asyncio.Task[Response]] = []
            async with asyncio.TaskGroup() as task_group:
                tasks = [
                    task_group.create_task(
                        client.post(
                            "/v2/query/stream",
                            json={
                                "query": "profile",
                                "sessionId": f"profile-{index}",
                            },
                            headers=headers,
                        )
                    )
                    for index in range(concurrency)
                ]
                await asyncio.wait_for(graph.all_entered.wait(), timeout=15)

            assert all(task.result().status_code == 200 for task in tasks)

    assert graph.entered == concurrency
