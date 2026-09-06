"""Public first bounded Specialist Task coverage."""

from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import psycopg
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.agents.specialist import PydanticAISpecialistActor
from app.config.models import FlowConfig, LangGraphRuntimeMode, LLMConfig, TenantConfig
from app.langgraph_v2.agent_batch import (
    DispatchBatch,
    EvidenceToolRegistration,
    SpecialistActor,
    SpecialistActorFactory,
    SpecialistAttempt,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistTaskInput,
    TaskProposal,
)
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    ExpectedToolUnavailability,
    FinancialResearchReport,
    PreparedSynthesis,
    ToolUnavailabilityRecord,
    ToolUnavailable,
    ToolUnavailableReason,
)
from app.langgraph_v2.agent_graph import CoordinatorInput, Finish
from app.langgraph_v2.agent_runtime import build_agent_runtime
from app.langgraph_v2.agent_scope import AgentIntentPolicy, SpecialistDescriptor
from app.langgraph_v2.agent_skills import (
    SkillInvocation,
    SkillReference,
    SkillRegistration,
    SpecialistSkillRegistry,
)
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
    tools: tuple[Callable[..., object], ...],
    returned_evidence: list[EvidenceEnvelope],
    skill_invocation: SkillInvocation | None,
    returned_unavailability: list[ToolUnavailabilityRecord],
) -> SpecialistActor:
    return _specialist_factory(
        query="Apple revenue",
        summary="Apple revenue grew.",
        evidence_id="evidence-1",
    )(tools, returned_evidence, skill_invocation, returned_unavailability)


def _specialist_factory(
    *, query: str, summary: str, evidence_id: str
) -> SpecialistActorFactory:
    def build(
        tools: tuple[Callable[..., object], ...],
        returned_evidence: list[EvidenceEnvelope],
        skill_invocation: SkillInvocation | None,
        returned_unavailability: list[ToolUnavailabilityRecord],
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
                            args={"source": "filing", "query": query},
                        )
                    ]
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args={"summary": summary, "evidence_ids": [evidence_id]},
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
            skill_invocation=skill_invocation,
            returned_unavailability=returned_unavailability,
        )

    return build


def _skill_specialist_factory(
    tools: tuple[Callable[..., object], ...],
    returned_evidence: list[EvidenceEnvelope],
    skill_invocation: SkillInvocation | None,
    returned_unavailability: list[ToolUnavailabilityRecord],
) -> PydanticAISpecialistActor:
    assert skill_invocation is not None
    calls = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        business_tools = [
            tool.name for tool in info.function_tools if tool.name != "activate_skill"
        ]
        assert business_tools == ["filing_reader"]
        if calls == 1:
            assert "filing-analysis" in repr(messages)
            assert "FULL-SKILL-INSTRUCTIONS-SENTINEL" not in repr(messages)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="activate_skill",
                        args={"skill_name": "filing-analysis"},
                    )
                ]
            )
        if calls == 2:
            assert "FULL-SKILL-INSTRUCTIONS-SENTINEL" in repr(messages)
            assert "FULL-SKILL-REFERENCE-SENTINEL" in repr(messages)
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
        skill_invocation=skill_invocation,
        returned_unavailability=returned_unavailability,
    )


async def _empty_evidence_provider(source: str, query: str) -> EvidenceEnvelope:
    assert (source, query) == ("filing", "Apple litigation")
    return EvidenceEnvelope(
        id="evidence-empty-1",
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task_90fff3e68e9a59d229d7982b65c5fe8b",
        source=source,
        source_url="https://example.test/filing-search",
        title="Authoritative filing search",
        body="No matching records.",
        excerpt="The authoritative search returned no matching records.",
        as_of_date=date(2026, 9, 6),
        raw_provider_payload="[]",
    )


class _Synthesis:
    async def synthesize(self, prepared: object) -> FinancialResearchReport:
        del prepared
        return FinancialResearchReport(markdown_report="Apple revenue grew. [[E:1]]")


class _EmptySynthesis:
    async def synthesize(self, prepared: object) -> FinancialResearchReport:
        del prepared
        return FinancialResearchReport(
            markdown_report="No matching filing records were found. [[E:1]]"
        )


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
        and event.get("data")
        == {
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


def test_unavailable_tool_fallback_persists_a_gap_and_marks_completion_incomplete(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class SourceUnreachable(Exception):
        pass

    provider_calls = 0

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            raise SourceUnreachable("provider payload must not persist")
        return await _evidence_provider(source, query)

    def specialist_factory(
        tools: tuple[Callable[..., object], ...],
        returned_evidence: list[EvidenceEnvelope],
        skill_invocation: SkillInvocation | None,
        returned_unavailability: list[ToolUnavailabilityRecord],
    ) -> PydanticAISpecialistActor:
        del skill_invocation
        model_calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="filing_reader",
                            tool_call_id="call-1",
                            args={"source": "filing", "query": "Apple revenue"},
                        )
                    ]
                )
            if model_calls == 2:
                tool_returns = [
                    part
                    for message in messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                    if part.part_kind == "tool-return"
                ]
                assert tool_returns[0].content == ToolUnavailable(
                    reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
                    requested_coverage="Apple revenue",
                )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="filing_reader",
                            tool_call_id="call-2",
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
            returned_unavailability=returned_unavailability,
        )

    class PartialSynthesis:
        prepared: PreparedSynthesis | None = None

        async def synthesize(
            self, prepared: PreparedSynthesis
        ) -> FinancialResearchReport:
            self.prepared = prepared
            return FinancialResearchReport(
                markdown_report="Apple revenue grew. [[E:1]]"
            )

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
                actor_factory=specialist_factory,
                allowed_tool_ids=frozenset({"filing_reader"}),
            ),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
        tool_registrations=(
            EvidenceToolRegistration(
                id="filing_reader",
                provider=provider,
                allowed_sources=frozenset({"filing"}),
                allowed_queries=frozenset({"Apple revenue"}),
                expected_unavailability=(
                    ExpectedToolUnavailability(
                        exception_type=SourceUnreachable,
                        reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
                    ),
                ),
            ),
        ),
        tenant_eligible_tool_ids=frozenset({"filing_reader"}),
    )
    synthesis = PartialSynthesis()

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
            synthesis_actor=synthesis,
        )

    conversation_id = "00000000-0000-0000-0000-000000000086"
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

    done = [event for event in parse_sse(response.text) if event["type"] == "done"]
    assert response.status_code == 200
    assert provider_calls == 2
    assert done[0]["data"]["metadata"]["completion_status"] == "incomplete"
    assert done[0]["data"]["metadata"]["termination_reason"] == "partial_results"
    assert done[0]["data"]["answer"] == (
        "Apple revenue grew. [[E:1]]\n\n"
        "Incomplete research: one or more requested data sources were unavailable."
    )
    assert synthesis.prepared is not None
    assert synthesis.prepared.data_gaps[0].model_dump() == {
        "requested_coverage": "Apple revenue",
        "reason": ToolUnavailableReason.SOURCE_UNREACHABLE,
        "observed_at": synthesis.prepared.data_gaps[0].observed_at,
    }
    assert checkpoint is not None
    checkpoint_text = repr(checkpoint.checkpoint["channel_values"])
    assert "unavailable_" in checkpoint_text
    assert "filing_reader" in checkpoint_text
    assert "provider payload must not persist" not in checkpoint_text


def test_specialist_activates_a_scope_bound_skill_before_publishing_evidence(
    langgraph_v2_migrated_database_url: str,
) -> None:
    coordinator = _Coordinator()
    policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
        allowed_tool_ids=frozenset({"filing_reader"}),
        allowed_skill_names=frozenset({"filing-analysis"}),
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
                actor_factory=_skill_specialist_factory,
                allowed_tool_ids=frozenset({"filing_reader"}),
                allowed_skill_names=frozenset({"filing-analysis"}),
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
        skill_registry=SpecialistSkillRegistry(
            registrations=(
                SkillRegistration(
                    name="filing-analysis",
                    version="2026.09",
                    description="Read an eligible filing before analysis.",
                    instructions="FULL-SKILL-INSTRUCTIONS-SENTINEL",
                    references=(
                        SkillReference(
                            name="filing-guide",
                            content="FULL-SKILL-REFERENCE-SENTINEL",
                        ),
                    ),
                    required_tool_ids=frozenset({"filing_reader"}),
                ),
            ),
            tenant_eligible_names=frozenset({"filing-analysis"}),
            shared_skill_names=frozenset({"filing-analysis"}),
        ),
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
            synthesis_actor=_Synthesis(),
        )

    conversation_id = "00000000-0000-0000-0000-000000000084"
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

    done = [event for event in parse_sse(response.text) if event["type"] == "done"]
    assert response.status_code == 200
    assert done[0]["data"]["answer"] == "Apple revenue grew. [[E:1]]"
    assert done[0]["data"]["citations"][0]["evidence_id"] == "evidence-1"
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    skill_pins = next(iter(state["accepted_batches"].values()))["skill_pins"]
    pin = skill_pins[0]["pins"][0]
    assert pin["name"] == "filing-analysis"
    assert pin["version"] == "2026.09"
    assert len(pin["content_hash"]) == 64
    assert "FULL-SKILL-INSTRUCTIONS-SENTINEL" not in repr(state)
    assert "FULL-SKILL-REFERENCE-SENTINEL" not in repr(state)
    assert all(
        "filing-analysis" not in input.model_dump_json() for input in coordinator.inputs
    )


def test_authoritative_empty_result_still_publishes_evidence_backed_report(
    langgraph_v2_migrated_database_url: str,
) -> None:
    policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
        allowed_tool_ids=frozenset({"filing_reader"}),
        allowed_sources=frozenset({"filing"}),
        allowed_queries=frozenset({"Apple litigation"}),
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )
    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                actor_factory=_specialist_factory(
                    query="Apple litigation",
                    summary="The authoritative search returned no matching records.",
                    evidence_id="evidence-empty-1",
                ),
                allowed_tool_ids=frozenset({"filing_reader"}),
            ),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
        tool_registrations=(
            EvidenceToolRegistration(
                id="filing_reader",
                provider=_empty_evidence_provider,
                allowed_sources=frozenset({"filing"}),
                allowed_queries=frozenset({"Apple litigation"}),
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
            synthesis_actor=_EmptySynthesis(),
        )

    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=factory,
    )
    app.state.tenant_manager = _TenantManager()
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={
                "query": "Are there litigation records?",
                "sessionId": "00000000-0000-0000-0000-000000000083",
                "clientRequestId": "request-1",
            },
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    done = [event for event in parse_sse(response.text) if event["type"] == "done"]
    assert response.status_code == 200
    assert done[0]["data"]["metadata"]["completion_status"] == "complete"
    assert done[0]["data"]["answer"] == (
        "No matching filing records were found. [[E:1]]"
    )
    assert done[0]["data"]["citations"][0]["evidence_id"] == "evidence-empty-1"


async def _checkpoint_history(
    checkpointer: BaseCheckpointSaver[Any], config: RunnableConfig
) -> list[object]:
    return [item async for item in checkpointer.alist(config)]
