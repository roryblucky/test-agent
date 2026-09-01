from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.graph import LinearGraphState, build_linear_graph
from app.langgraph_v2.history import ConversationExchange
from app.langgraph_v2.question_refinement import (
    MockQuestionRefinementActor,
    PydanticAIQuestionRefinementActor,
    QuestionRefinementResult,
    V2ResolvedQuery,
)
from app.models.workflow import ResolvedQuery
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
    seed_subject_conversation,
)


def _state() -> LinearGraphState:
    return {
        "query": "compare gold and FX",
        "conversation_id": "00000000-0000-0000-0000-000000000001",
        "request_id": "request-1",
    }


@pytest.mark.asyncio
async def test_safe_query_gets_structured_refinement(
) -> None:
    graph = build_linear_graph(
        tenant_id="tenant-a",
        refinement_actor=MockQuestionRefinementActor(),
    )

    result = await graph.ainvoke(_state())

    assert "refined_query" in result
    assert result["refined_query"] == "compare gold and FX"


@pytest.mark.asyncio
async def test_refinement_failure_halts_before_later_phases(
) -> None:
    class FailingActor:
        async def refine(
            self, query: str, history: Sequence[ConversationExchange]
        ) -> QuestionRefinementResult:
            del history
            raise ValueError(f"invalid output for {query}")

    graph = build_linear_graph(
        tenant_id="tenant-a",
        refinement_actor=FailingActor(),
    )

    result = await graph.ainvoke(_state())

    assert "halted" in result
    assert result["halted"] is True


@pytest.mark.asyncio
async def test_refinement_reexecution_reinvokes_actor(
) -> None:
    class CountingActor:
        calls = 0

        async def refine(
            self, query: str, history: Sequence[ConversationExchange]
        ) -> QuestionRefinementResult:
            del history
            self.calls += 1
            return QuestionRefinementResult(
                resolved_query=V2ResolvedQuery(
                    original_query=query, standalone_query="standalone"
                )
            )

    actor = CountingActor()
    graph = build_linear_graph(
        tenant_id="tenant-a",
        refinement_actor=actor,
    )

    await graph.ainvoke(_state())
    await graph.ainvoke(_state())

    assert actor.calls == 2


@pytest.mark.asyncio
async def test_pydantic_ai_actor_returns_agent_output() -> None:
    class FakeAgent:
        async def run(self, query: str) -> object:
            return type(
                "Result",
                (),
                {
                    "output": ResolvedQuery(
                        original_query=query,
                        standalone_query="refined",
                    )
                },
            )()

    actor = PydanticAIQuestionRefinementActor(FakeAgent())  # type: ignore[arg-type]

    result = await actor.refine("raw")

    assert result.resolved_query.standalone_query == "refined"


def test_http_query_uses_injected_refinement_actor(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async def seed() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            await seed_subject_conversation(pool)

    asyncio.run(seed())

    class CountingActor:
        calls = 0

        async def refine(
            self, query: str, history: Sequence[ConversationExchange]
        ) -> QuestionRefinementResult:
            del history
            self.calls += 1
            return QuestionRefinementResult(
                resolved_query=V2ResolvedQuery(
                    original_query=query, standalone_query="http-refined"
                )
            )

    actor = CountingActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=actor,
    )

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "00000000-0000-0000-0000-000000000001"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert response.status_code == 200
    assert actor.calls == 1
    delivered = parse_sse(response.text)
    assert delivered[-1]["data"]["refined_query"] == "http-refined"


def test_new_request_does_not_inherit_prior_refinement_usage(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class UsageThenNoUsageActor:
        calls = 0

        async def refine(
            self, query: str, history: Sequence[ConversationExchange]
        ) -> QuestionRefinementResult:
            del history
            self.calls += 1
            return QuestionRefinementResult(
                resolved_query=V2ResolvedQuery(
                    original_query=query,
                    standalone_query=query,
                ),
                usage={"input_tokens": 3} if self.calls == 1 else {},
            )

    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=UsageThenNoUsageActor(),
    )
    headers = {
        "X-Application-Id": "tenant-a",
        "X-Subject-Id": "subject-a",
    }

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        first = client.post(
            "/v2/query/stream",
            json={"query": "first", "clientRequestId": "first-request"},
            headers=headers,
        )
        second = client.post(
            "/v2/query/stream",
            json={
                "query": "second",
                "sessionId": first.headers["x-conversation-id"],
                "clientRequestId": "second-request",
            },
            headers=headers,
        )

    first_done = parse_sse(first.text)[-1]
    second_done = parse_sse(second.text)[-1]
    assert first_done["data"]["metadata"]["usage"]["input_tokens"] == 3
    assert "usage" not in second_done["data"]["metadata"]
