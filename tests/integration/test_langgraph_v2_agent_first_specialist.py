"""Public first no-Tool Specialist Task coverage."""

from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config.models import FlowConfig, LangGraphRuntimeMode, LLMConfig, TenantConfig
from app.langgraph_v2.agent_batch import (
    DispatchBatch,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistTaskInput,
    TaskProposal,
)
from app.langgraph_v2.agent_graph import CoordinatorInput, Finish
from app.langgraph_v2.agent_runtime import build_agent_runtime
from app.langgraph_v2.agent_scope import AgentIntentPolicy, SpecialistDescriptor
from app.langgraph_v2.api import GraphRuntimeAdapter
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import thread_checkpoint_config, thread_id_for
from app.langgraph_v2.conversation_context import ConversationExchange
from app.models.workflow import IntentResult, QueryUnderstandingOutput, ResolvedQuery
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)


class _UnderstandingActor:
    async def understand(
        self, query: str, history: Sequence[ConversationExchange]
    ) -> QueryUnderstandingOutput:
        assert history == []
        return QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query=query, standalone_query="Apple outlook"
            ),
            intent=IntentResult(intent="market_outlook", confidence=0.9),
        )


class _Coordinator:
    def __init__(self) -> None:
        self.inputs: list[CoordinatorInput] = []

    async def decide(self, input: CoordinatorInput) -> DispatchBatch | Finish:
        self.inputs.append(input)
        if len(self.inputs) == 1:
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Assess Apple market outlook.",
                    ),
                ),
            )
        return Finish(kind="finish")


class _Specialist:
    def __init__(self) -> None:
        self.inputs: list[SpecialistTaskInput] = []

    async def run(self, input: SpecialistTaskInput) -> SpecialistFindingDraft:
        self.inputs.append(input)
        return SpecialistFindingDraft(summary="No-tool market finding")


class _TenantManager:
    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        assert tenant_id == "tenant-a"
        return TenantConfig(
            kms_app_name="Agent Tenant",
            application_id="tenant-a",
            ad_groups=[],
            runtime_mode=LangGraphRuntimeMode.AGENT,
            llm_config=LLMConfig(models={}),
            flow_config=FlowConfig(),
        )


def test_first_no_tool_specialist_is_accepted_before_conservative_done(
    langgraph_v2_migrated_database_url: str,
) -> None:
    coordinator = _Coordinator()
    specialist = _Specialist()
    policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )
    registry = SpecialistRegistry(
        registrations=(SpecialistRegistration(id="market-data", actor=specialist),),
        tenant_eligible_ids=frozenset({"market-data"}),
    )

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
            query_understanding_actor=_UnderstandingActor(),
            coordinator_actor=coordinator,
            specialist_registry=registry,
            intent_policies={policy.intent: policy},
        )

    conversation_id = "00000000-0000-0000-0000-000000000081"
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=factory,
    )
    app.state.tenant_manager = _TenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "What about it?", "sessionId": conversation_id},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        assert client.portal is not None
        checkpoint = client.portal.call(
            lambda: app.state.langgraph_v2_checkpointer.aget_tuple(
                thread_checkpoint_config(
                    thread_id=thread_id_for(
                        "tenant-a", "subject-a", "agent", conversation_id
                    )
                )
            )
        )

    done = [event for event in parse_sse(response.text) if event["type"] == "done"]
    assert response.status_code == 200
    assert len(done) == 1
    assert done[0]["data"]["metadata"]["termination_reason"] == "insufficient_evidence"
    assert len(coordinator.inputs) == 2
    assert specialist.inputs[0].objective == "Assess Apple market outlook."
    assert specialist.inputs[0].task_id.startswith("task_")
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    assert state["active_batch"] is None
    assert state["staged_contributions"] == {}
    assert len(state["accepted_batches"]) == 1
