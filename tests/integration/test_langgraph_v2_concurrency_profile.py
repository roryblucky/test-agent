from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.contracts import V2QueryRequest
from tests.integration.test_langgraph_v2_tracer import (
    persistent_tracer_app,
    seed_subject_conversation,
    stream_request,
    v2_stream_endpoint,
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
    context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")

    async with app.router.lifespan_context(app):
        for index in range(concurrency + 1):
            await seed_subject_conversation(
                app.state.langgraph_v2_postgres_pool,
                f"profile-{index}",
            )

        endpoint = v2_stream_endpoint(app)

        async def response_for(index: int) -> Any:
            return await endpoint(
                V2QueryRequest(
                    query="profile",
                    conversation_id=f"profile-{index}",
                ),
                stream_request(app),
                request_context=context,
            )

        warm_response = await response_for(concurrency)
        async for _ in warm_response.body_iterator:
            pass
        assert graph.entered == 1

        graph.reset(concurrency)
        responses = await asyncio.wait_for(
            asyncio.gather(*(response_for(index) for index in range(concurrency))),
            timeout=15,
        )

        async def consume(response: Any) -> None:
            async for _ in response.body_iterator:
                pass

        consumers = [asyncio.create_task(consume(response)) for response in responses]
        await asyncio.wait_for(graph.all_entered.wait(), timeout=15)
        await asyncio.wait_for(asyncio.gather(*consumers), timeout=15)

    assert graph.entered == concurrency

