"""Public first no-Tool Specialist Task coverage."""

from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import psycopg
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.agents.specialist import PydanticAISpecialistActor
from app.config.models import FlowConfig, LangGraphRuntimeMode, LLMConfig, TenantConfig
from app.langgraph_v2.agent_batch import (
    DispatchBatch,
    EvidenceToolRegistration,
    SpecialistAttempt,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistTaskInput,
    TaskProposal,
)
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    FinancialResearchReport,
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

    async def run(self, input: SpecialistTaskInput) -> SpecialistAttempt:
        self.inputs.append(input)
        return SpecialistAttempt(
            finding=SpecialistFindingDraft(summary="No-tool market finding")
        )


async def _evidence_provider(source: str, query: str) -> EvidenceEnvelope:
    assert (source, query) == ("filing", "Apple revenue")
    return EvidenceEnvelope(
        id="evidence-1",
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task_90fff3e68e9a59d229d7982b65c5fe8b",
        source=source,
        source_url="https://example.test/filing",
        title="Annual filing",
        body="BODY-SENTINEL",
        excerpt="Apple revenue grew.",
        as_of_date=date(2026, 9, 6),
        raw_provider_payload="RAW-PROVIDER-SENTINEL",
    )


def _evidence_specialist_factory(
    tools: tuple[Callable[..., object], ...], returned_evidence: list[EvidenceEnvelope]
) -> PydanticAISpecialistActor:
    calls = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        nonlocal calls
        calls += 1
        if calls == 1:
            assert [tool.name for tool in info.function_tools] == ["filing_reader"]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="filing_reader",
                        args={"source": "filing", "query": "Apple revenue"},
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "summary": "Apple revenue grew.",
                        "evidence_ids": ["evidence-1"],
                    },
                )
            ]
        )

    return PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=tools,
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        ),
        returned_evidence=returned_evidence,
    )


class _Synthesis:
    async def synthesize(self, prepared: object) -> FinancialResearchReport:
        del prepared
        return FinancialResearchReport(markdown_report="Apple revenue grew. [[E:1]]")


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
        allowed_tool_ids=frozenset({"filing_reader"}),
        allowed_sources=frozenset({"filing"}),
        allowed_queries=frozenset({"Apple revenue"}),
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


def test_evidence_backed_specialist_publishes_citation_without_checkpoint_body(
    langgraph_v2_migrated_database_url: str,
) -> None:
    policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
        allowed_tool_ids=frozenset({"filing_reader"}),
        allowed_sources=frozenset({"filing"}),
        allowed_queries=frozenset({"Apple revenue"}),
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )
    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                actor_factory=_evidence_specialist_factory,
                allowed_tool_ids=frozenset({"filing_reader"}),
            ),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
        tool_registrations=(
            EvidenceToolRegistration(
                id="filing_reader",
                provider=_evidence_provider,
                allowed_sources=frozenset({"filing"}),
                allowed_queries=frozenset({"Apple revenue"}),
            ),
        ),
        tenant_eligible_tool_ids=frozenset({"filing_reader"}),
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
            coordinator_actor=_Coordinator(),
            specialist_registry=registry,
            intent_policies={policy.intent: policy},
            synthesis_actor=_Synthesis(),
        )

    conversation_id = "00000000-0000-0000-0000-000000000082"
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=factory,
    )
    app.state.tenant_manager = _TenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={
                "query": "What about it?",
                "sessionId": conversation_id,
                "clientRequestId": "request-1",
            },
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
        checkpoint_history = client.portal.call(
            lambda: _checkpoint_history(
                app.state.langgraph_v2_checkpointer,
                thread_checkpoint_config(
                    thread_id=thread_id_for(
                        "tenant-a", "subject-a", "agent", conversation_id
                    )
                ),
            )
        )

    done = [event for event in parse_sse(response.text) if event["type"] == "done"]
    events = parse_sse(response.text)
    assert response.status_code == 200
    assert done[0]["data"]["answer"] == "Apple revenue grew. [[E:1]]"
    assert done[0]["data"]["citations"][0]["evidence_id"] == "evidence-1"
    assert any(
        event["type"] == "step_start" and event.get("step") == "specialist"
        for event in events
    )
    assert any(
        event["type"] == "progress"
        and event.get("step") == "tool"
        and event.get("data") == {
            "task_id": "task_90fff3e68e9a59d229d7982b65c5fe8b",
            "tool_id": "filing_reader",
            "status": "completed",
        }
        for event in events
    )
    assert checkpoint is not None
    assert "BODY-SENTINEL" not in repr(checkpoint.checkpoint["channel_values"])
    assert "RAW-PROVIDER-SENTINEL" not in repr(checkpoint.checkpoint["channel_values"])
    history_text = repr(checkpoint_history)
    assert "BODY-SENTINEL" not in history_text
    assert "RAW-PROVIDER-SENTINEL" not in history_text
    with psycopg.connect(langgraph_v2_migrated_database_url) as connection:
        persisted_text = "\n".join(
            repr(row)
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
            for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        )
    assert "BODY-SENTINEL" not in persisted_text
    assert "RAW-PROVIDER-SENTINEL" not in persisted_text


async def _checkpoint_history(
    checkpointer: BaseCheckpointSaver[Any], config: RunnableConfig
) -> list[object]:
    return [item async for item in checkpointer.alist(config)]
