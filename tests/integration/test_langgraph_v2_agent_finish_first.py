"""Public Agent Finish-first coverage."""

from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config.models import FlowConfig, LangGraphRuntimeMode, LLMConfig, TenantConfig
from app.langgraph_v2.agent_coordination import CoordinatorInput, Finish
from app.langgraph_v2.agent_runtime import build_agent_runtime
from app.langgraph_v2.agent_scope import AgentIntentPolicy, SpecialistDescriptor
from app.langgraph_v2.api import GraphRuntimeAdapter, GraphRuntimeFactory
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import (
    AgentCheckpointStateAdapter,
    read_conversation_messages,
    thread_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.conversation_context import ConversationExchange
from app.models.workflow import IntentResult, QueryUnderstandingOutput, ResolvedQuery
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)


class _UnderstandingActor:
    def __init__(self, *, intent: str = "market_outlook", standalone_query: str = "Apple outlook") -> None:
        self.intent = intent
        self.standalone_query = standalone_query
        self.histories: list[list[ConversationExchange]] = []

    async def understand(
        self, query: str, history: Sequence[ConversationExchange]
    ) -> QueryUnderstandingOutput:
        self.histories.append(list(history))
        return QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(original_query=query, standalone_query=self.standalone_query),
            intent=IntentResult(intent=self.intent, confidence=0.9),
        )


class _FinishCoordinator:
    def __init__(self) -> None:
        self.inputs: list[CoordinatorInput] = []

    async def decide(self, input: CoordinatorInput) -> Finish:
        self.inputs.append(input)
        return Finish(kind="finish")


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


def _factory(
    understanding: _UnderstandingActor, coordinator: _FinishCoordinator
) -> GraphRuntimeFactory:
    policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )

    def factory(
        *, app: FastAPI, request_context: TrustedRequestContext, checkpointer: BaseCheckpointSaver[Any]
    ) -> GraphRuntimeAdapter:
        return build_agent_runtime(
            app,
            request_context=request_context,
            checkpointer=checkpointer,
            query_understanding_actor=understanding,
            coordinator_actor=coordinator,
            intent_policies={policy.intent: policy},
        )

    return factory


def test_agent_first_finish_publishes_one_insufficient_evidence_answer(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "00000000-0000-0000-0000-000000000071"
    understanding = _UnderstandingActor()
    coordinator = _FinishCoordinator()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=_factory(understanding, coordinator),
    )
    app.state.tenant_manager = _AgentTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "What about it?", "sessionId": conversation_id},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        assert client.portal is not None
        messages = client.portal.call(
            lambda: read_conversation_messages(
                app.state.langgraph_v2_checkpointer,
                thread_checkpoint_config(
                    thread_id=thread_id_for("tenant-a", "subject-a", "agent", conversation_id)
                ),
                state_adapter=AgentCheckpointStateAdapter(),
            )
        )

    done = [event for event in parse_sse(response.text) if event["type"] == "done"]
    assert response.status_code == 200
    assert len(done) == 1
    answer = done[0]["data"]["answer"]
    assert answer == "Incomplete research: no eligible Evidence was available."
    assert answer.count("no eligible Evidence") == 1
    assert done[0]["data"]["metadata"]["completion_status"] == "incomplete"
    assert done[0]["data"]["metadata"]["termination_reason"] == "insufficient_evidence"
    assert understanding.histories == [[]]
    assert coordinator.inputs == [
        CoordinatorInput(
            standalone_query="Apple outlook",
            intent="market_outlook",
            specialist_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
        )
    ]
    assert [(message.type, message.text) for message in messages] == [
        ("human", "What about it?"),
        ("ai", answer),
    ]


def test_agent_unknown_intent_fails_before_coordinator_or_done(
    langgraph_v2_migrated_database_url: str,
) -> None:
    understanding = _UnderstandingActor(intent="unknown")
    coordinator = _FinishCoordinator()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=_factory(understanding, coordinator),
    )
    app.state.tenant_manager = _AgentTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "What about it?"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    events = parse_sse(response.text)
    assert coordinator.inputs == []
    assert all(event["type"] != "done" for event in events)
    assert events[-1] == {"type": "error", "data": "Agent Intent is not configured"}


def test_agent_blank_standalone_query_fails_before_coordinator_or_done(
    langgraph_v2_migrated_database_url: str,
) -> None:
    understanding = _UnderstandingActor(standalone_query="  ")
    coordinator = _FinishCoordinator()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=_factory(understanding, coordinator),
    )
    app.state.tenant_manager = _AgentTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "What about it?"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    events = parse_sse(response.text)
    assert coordinator.inputs == []
    assert all(event["type"] != "done" for event in events)
    assert events[-1] == {
        "type": "error",
        "data": "Query Understanding standalone query must not be blank",
    }
