from __future__ import annotations

import asyncio
from collections.abc import Sequence
from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.graph import build_tracer_graph
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.question_refinement import (
    MockQuestionRefinementActor,
    PydanticAIQuestionRefinementActor,
    run_question_refinement,
)
from app.langgraph_v2.run_events import RunEventRepository
from app.models.workflow import ResolvedQuery
from tests.integration.test_langgraph_v2_tracer import parse_sse, persistent_tracer_app


def _state() -> dict:
    return {
        "query": "compare gold and FX",
        "conversation_id": "conversation-1",
        "client_request_id": None,
        "events": [],
    }


@pytest.mark.asyncio
async def test_safe_query_gets_structured_refinement_and_events(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        graph = build_tracer_graph(
            phase_context=PhaseExecutionContext(
                repository=PhaseResultRepository(pool),
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            ),
            refinement_actor=MockQuestionRefinementActor(),
        )

        result = await graph.ainvoke(_state())

        assert result["refined_query"] == "compare gold and FX"
        assert [event.get("step") for event in result["events"]] == [
            "query",
            "query",
            "moderation:pre",
            "moderation:pre",
            "llm:refine_question",
            "llm:refine_question",
            "finalization",
            "finalization",
            None,
        ]


@pytest.mark.asyncio
async def test_refinement_failure_halts_before_later_phases(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class FailingActor:
        async def refine(
            self, query: str, history: Sequence[ConversationTurn]
        ) -> ResolvedQuery:
            del history
            raise ValueError(f"invalid output for {query}")

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        graph = build_tracer_graph(
            phase_context=PhaseExecutionContext(
                repository=PhaseResultRepository(pool),
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            ),
            refinement_actor=FailingActor(),
        )

        result = await graph.ainvoke(_state())

        assert result["halted"] is True
        assert result["events"][-1]["type"] == "error"
        assert all(event.get("step") != "finalization" for event in result["events"])
        assert (await runs.get_run("tenant-a", run.run_id)).status == "failed"


@pytest.mark.asyncio
async def test_refinement_replay_does_not_reinvoke_actor_or_duplicate_events(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class CountingActor:
        calls = 0

        async def refine(
            self, query: str, history: Sequence[ConversationTurn]
        ) -> ResolvedQuery:
            del history
            self.calls += 1
            return ResolvedQuery(original_query=query, standalone_query="standalone")

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        actor = CountingActor()
        graph = build_tracer_graph(
            phase_context=PhaseExecutionContext(
                repository=PhaseResultRepository(pool),
                tenant_id="tenant-a",
                run_id=run.run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            ),
            refinement_actor=actor,
        )

        await graph.ainvoke(_state())
        await graph.ainvoke(_state())

        assert actor.calls == 1
        events = await runs.list_events("tenant-a", run.run_id)
        assert [event.event_key for event in events] == [
            "phase:query:step_start:1",
            "phase:query:step_completed:1",
            "phase:pre_moderation:step_start:1",
            "phase:pre_moderation:step_completed:1",
            "phase:question_refinement:step_start:1",
            "phase:question_refinement:step_completed:1",
        ]


@pytest.mark.asyncio
async def test_refinement_replays_after_commit_before_checkpoint(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class CountingActor:
        calls = 0

        async def refine(
            self, query: str, history: Sequence[ConversationTurn]
        ) -> ResolvedQuery:
            del history
            self.calls += 1
            return ResolvedQuery(original_query=query, standalone_query="standalone")

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        actor = CountingActor()

        first_events, _, first, _ = await run_question_refinement(
            _state(), context=context, actor=actor
        )
        replayed_events, _, replayed, _ = await run_question_refinement(
            _state(), context=context, actor=actor
        )

        assert actor.calls == 1
        assert first == replayed
        assert [event.event_key for event in first_events] == [
            "phase:question_refinement:step_start:1",
            "phase:question_refinement:step_completed:1",
        ]
        assert [event.event_key for event in replayed_events] == [
            "phase:question_refinement:step_start:1",
            "phase:question_refinement:step_completed:1",
        ]


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

    assert result.standalone_query == "refined"


def test_http_query_uses_injected_refinement_actor(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async def seed() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            await ConversationMessageRepository(pool).resolve_conversation(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="conversation-1",
            )

    asyncio.run(seed())

    class CountingActor:
        calls = 0

        async def refine(
            self, query: str, history: Sequence[ConversationTurn]
        ) -> ResolvedQuery:
            del history
            self.calls += 1
            return ResolvedQuery(original_query=query, standalone_query="http-refined")

    actor = CountingActor()
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=actor,
    )

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert response.status_code == 200
    assert actor.calls == 1
    assert parse_sse(response.text)[-1]["data"]["refined_query"] == "http-refined"
