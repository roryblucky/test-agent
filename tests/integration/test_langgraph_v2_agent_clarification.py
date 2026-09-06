"""Public Agent-mode clarification coverage."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager, suppress
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config.models import FlowConfig, LangGraphRuntimeMode, LLMConfig, TenantConfig
from app.langgraph_v2.agent_runtime import build_agent_runtime
from app.langgraph_v2.api import GraphRuntimeAdapter, GraphRuntimeFactory
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import (
    AgentCheckpointStateAdapter,
    read_conversation_messages,
    thread_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.conversation_context import ConversationExchange
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.pre_moderation import ModerationDecision
from app.models.workflow import (
    IntentResult,
    QueryUnderstandingClarification,
    QueryUnderstandingClarificationQuestion,
    QueryUnderstandingOutput,
    ResolvedQuery,
)
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
    stream_request,
    v2_stream_endpoint,
)


class _ClarifyingActor:
    def __init__(self) -> None:
        self.histories: list[list[ConversationExchange]] = []

    async def understand(
        self,
        query: str,
        history: Sequence[ConversationExchange],
    ) -> QueryUnderstandingOutput:
        self.histories.append(list(history))
        return QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query=query,
                standalone_query=query,
            ),
            intent=IntentResult(intent="market_outlook", confidence=0.5),
            clarification=QueryUnderstandingClarification(
                scope="query_resolution",
                questions=[
                    QueryUnderstandingClarificationQuestion(
                        question="Which company do you mean?",
                        options=["Apple", "Microsoft"],
                    )
                ],
                reason="The company reference is ambiguous.",
            ),
        )


class _AgentTenantManager:
    def __init__(self) -> None:
        self.config = TenantConfig(
            kms_app_name="Agent Tenant",
            application_id="tenant-a",
            ad_groups=[],
            runtime_mode=LangGraphRuntimeMode.AGENT,
            llm_config=LLMConfig(models={}),
            flow_config=FlowConfig(),
        )

    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        assert tenant_id == "tenant-a"
        return self.config

    def get_providers(self, tenant_id: str) -> object:
        assert tenant_id == "tenant-a"
        return object()


class _FlaggingModeration:
    async def check(self, text: str) -> ModerationDecision:
        assert text == "blocked request"
        return ModerationDecision(is_flagged=True, reason="blocked for test")


class _InvalidClarifyingActor:
    async def understand(
        self,
        query: str,
        history: Sequence[ConversationExchange],
    ) -> QueryUnderstandingOutput:
        del history
        return QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(original_query=query, standalone_query=query),
            intent=IntentResult(intent="market_outlook", confidence=0.5),
            clarification=QueryUnderstandingClarification(
                scope="query_resolution",
                questions=[
                    QueryUnderstandingClarificationQuestion(question=f"Question {i}")
                    for i in range(4)
                ],
            ),
        )


def _agent_runtime_factory(actor: _ClarifyingActor) -> GraphRuntimeFactory:
    def factory(
        *,
        app: FastAPI,
        request_context: TrustedRequestContext,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> GraphRuntimeAdapter:
        return build_agent_runtime(
            app,
            request_context=request_context,
            checkpointer=checkpointer,
            query_understanding_actor=actor,
        )

    return factory


def test_agent_clarification_is_committed_before_one_done_event(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "00000000-0000-0000-0000-000000000061"
    actor = _ClarifyingActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=_agent_runtime_factory(actor),
    )
    app.state.tenant_manager = _AgentTenantManager()
    headers = {"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"}

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={
                "query": "What about it?",
                "sessionId": conversation_id,
                "mode": "linear",
            },
            headers=headers,
        )
        assert client.portal is not None
        messages = client.portal.call(
            lambda: read_conversation_messages(
                app.state.langgraph_v2_checkpointer,
                thread_checkpoint_config(
                    thread_id=thread_id_for(
                        "tenant-a", "subject-a", "agent", conversation_id
                    )
                ),
                state_adapter=AgentCheckpointStateAdapter(),
            )
        )

    events = parse_sse(response.text)
    done = [event for event in events if event["type"] == "done"]
    assert response.status_code == 200
    assert len(done) == 1
    assert done[0]["data"] == {
        "query": "What about it?",
        "refined_query": None,
        "intent": None,
        "answer": "Which company do you mean?",
        "documents": [],
        "moderation": None,
        "groundedness": None,
        "clarification": {
            "scope": "query_resolution",
            "questions": [
                {
                    "question": "Which company do you mean?",
                    "options": ["Apple", "Microsoft"],
                }
            ],
            "reason": "The company reference is ambiguous.",
        },
        "session_id": conversation_id,
        "metadata": {"steps_executed": ["initializer", "pre_moderation", "query_understanding", "finalize_state"]},
        "citations": [],
    }
    assert "completion_status" not in done[0]["data"]["metadata"]
    assert "termination_reason" not in done[0]["data"]["metadata"]
    assert actor.histories == [[]]
    assert [(message.type, message.text) for message in messages] == [
        ("human", "What about it?"),
        ("ai", "Which company do you mean?"),
    ]


def test_agent_follow_up_starts_a_clean_run_from_complete_clarification_pairs(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "00000000-0000-0000-0000-000000000062"
    actor = _ClarifyingActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=_agent_runtime_factory(actor),
    )
    app.state.tenant_manager = _AgentTenantManager()
    headers = {"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"}

    with TestClient(app) as client:
        first = client.post(
            "/v2/query/stream",
            json={
                "query": "What about it?",
                "sessionId": conversation_id,
                "clientRequestId": "clarify-1",
            },
            headers=headers,
        )
        second = client.post(
            "/v2/query/stream",
            json={
                "query": "Apple",
                "sessionId": conversation_id,
                "clientRequestId": "clarify-2",
            },
            headers=headers,
        )
        repeated = client.post(
            "/v2/query/stream",
            json={
                "query": "What about it?",
                "sessionId": conversation_id,
                "clientRequestId": "clarify-1",
            },
            headers=headers,
        )
        assert client.portal is not None
        messages = client.portal.call(
            lambda: read_conversation_messages(
                app.state.langgraph_v2_checkpointer,
                thread_checkpoint_config(
                    thread_id=thread_id_for(
                        "tenant-a", "subject-a", "agent", conversation_id
                    )
                ),
                state_adapter=AgentCheckpointStateAdapter(),
            )
        )

    assert first.status_code == second.status_code == repeated.status_code == 200
    assert first.headers["x-request-id"] == repeated.headers["x-request-id"]
    assert actor.histories == [
        [],
        [
            ConversationExchange(
                user="What about it?",
                assistant="Which company do you mean?",
            )
        ],
        [
            ConversationExchange(
                user="Apple",
                assistant="Which company do you mean?",
            )
        ],
    ]
    assert [message.id for message in messages].count("clarify-1:user") == 1
    assert [message.id for message in messages].count("clarify-1:assistant") == 1
    assert [message.id for message in messages].count("clarify-2:user") == 1
    assert [message.id for message in messages].count("clarify-2:assistant") == 1


def test_agent_pre_moderation_flag_skips_clarification_and_done(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "00000000-0000-0000-0000-000000000063"
    actor = _ClarifyingActor()

    def factory(
        *,
        app: FastAPI,
        request_context: TrustedRequestContext,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> GraphRuntimeAdapter:
        return build_agent_runtime(
            app,
            request_context=request_context,
            checkpointer=checkpointer,
            query_understanding_actor=actor,
            moderation_provider=_FlaggingModeration(),
        )

    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=factory,
    )
    app.state.tenant_manager = _AgentTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "blocked request", "sessionId": conversation_id},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        assert client.portal is not None
        messages = client.portal.call(
            lambda: read_conversation_messages(
                app.state.langgraph_v2_checkpointer,
                thread_checkpoint_config(
                    thread_id=thread_id_for(
                        "tenant-a", "subject-a", "agent", conversation_id
                    )
                ),
                state_adapter=AgentCheckpointStateAdapter(),
            )
        )

    assert response.status_code == 200
    assert actor.histories == []
    assert [event["type"] for event in parse_sse(response.text)] == [
        "step_start",
        "error",
    ]
    assert [(message.type, message.text) for message in messages] == [
        ("human", "blocked request"),
    ]


def test_agent_rejects_an_overlong_clarification_before_publication(
    langgraph_v2_migrated_database_url: str,
) -> None:
    actor = _InvalidClarifyingActor()

    def factory(
        *,
        app: FastAPI,
        request_context: TrustedRequestContext,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> GraphRuntimeAdapter:
        return build_agent_runtime(
            app,
            request_context=request_context,
            checkpointer=checkpointer,
            query_understanding_actor=actor,
        )

    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=factory,
    )
    app.state.tenant_manager = _AgentTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "What about it?"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    events = parse_sse(response.text)
    assert response.status_code == 200
    assert all(event["type"] != "done" for event in events)
    assert events[-1] == {
        "type": "error",
        "data": "Query Understanding clarification requires one to three questions",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("after_commit", [False, True])
async def test_agent_cancellation_at_final_checkpoint_never_publishes_done(
    langgraph_v2_migrated_database_url: str,
    after_commit: bool,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BarrierSaver(AsyncPostgresSaver):
        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Any,
            metadata: Any,
            new_versions: Any,
        ) -> RunnableConfig:
            if checkpoint.get("channel_values", {}).get("final_response") is None:
                return await super().aput(config, checkpoint, metadata, new_versions)
            if after_commit:
                result = await super().aput(config, checkpoint, metadata, new_versions)
                entered.set()
                await release.wait()
                return result
            entered.set()
            await release.wait()
            return await super().aput(config, checkpoint, metadata, new_versions)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=langgraph_v2_migrated_database_url),
            checkpointer_factory=_BarrierSaver,
        ):
            yield

    actor = _ClarifyingActor()
    app = FastAPI(lifespan=lifespan)
    app.state.tenant_manager = _AgentTenantManager()
    from app.langgraph_v2.api import register_v2_routes

    register_v2_routes(
        app,
        enabled=True,
        agent_runtime_factory=_agent_runtime_factory(actor),
    )
    conversation_id = UUID(
        "00000000-0000-0000-0000-000000000064" if after_commit
        else "00000000-0000-0000-0000-000000000065"
    )
    request_context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")

    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            payload=V2QueryRequest(query="What about it?", conversation_id=conversation_id),
            http_request=stream_request(app),
            request_context=request_context,
        )
        subscriber = response.body_iterator
        frames: list[str] = []

        async def consume() -> None:
            frames.extend([frame async for frame in subscriber])

        consumer = asyncio.create_task(consume())
        await entered.wait()
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await consumer
        await subscriber.aclose()
        messages = await read_conversation_messages(
            app.state.langgraph_v2_checkpointer,
            thread_checkpoint_config(
                thread_id=thread_id_for(
                    "tenant-a", "subject-a", "agent", str(conversation_id)
                )
            ),
            state_adapter=AgentCheckpointStateAdapter(),
        )

    assert actor.histories == [[]]
    assert all(parse_sse(frame)[0]["type"] != "done" for frame in frames)
    assert [message.type for message in messages] == (
        ["human", "ai"] if after_commit else ["human"]
    )
