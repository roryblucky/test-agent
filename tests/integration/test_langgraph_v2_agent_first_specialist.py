"""Public first bounded Specialist Task coverage."""

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import date
from decimal import Decimal
from typing import Any, cast
from uuid import UUID

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage, UsageLimits

from app.agents.specialist import PydanticAISpecialistActor
from app.config.models import FlowConfig, LangGraphRuntimeMode, LLMConfig, TenantConfig
from app.langgraph_v2 import agent_graph
from app.langgraph_v2.agent_batch import (
    CalculationToolRegistration,
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
from app.langgraph_v2.agent_completion import IncompleteResearch
from app.langgraph_v2.agent_coordination import (
    CoordinatorInput,
    CoordinatorOutputInvalid,
    Finish,
)
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    ExpectedToolUnavailability,
    FinancialResearchReport,
    PreparedSynthesis,
    PublishedReport,
    SpecialistToolCapture,
    SynthesisActor,
    ToolUnavailable,
    ToolUnavailableReason,
)
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
from app.langgraph_v2.calculations import (
    CalculationExecutionContext,
    CalculationExecutor,
    CalculationMethod,
    CalculationRequest,
    PriceObservation,
    TrustedPriceSeries,
)
from app.langgraph_v2.checkpointing import (
    AgentCheckpointStateAdapter,
    read_conversation_messages,
    thread_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.conversation_context import ConversationExchange
from app.langgraph_v2.postgres import CheckpointerFactory
from app.langgraph_v2.specialist_retry import (
    SpecialistFailureFacts,
    SpecialistInvocationFailure,
    SpecialistModelBoundary,
    specialist_usage_limits,
)
from app.models.workflow import IntentResult, QueryUnderstandingOutput, ResolvedQuery
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
    stream_request,
    v2_stream_endpoint,
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


class _RequestScopedCoordinator:
    """Dispatch once whenever the current request has no accepted result."""

    def __init__(self) -> None:
        self.inputs: list[CoordinatorInput] = []

    async def decide(self, input: CoordinatorInput) -> DispatchBatch | Finish:
        self.inputs.append(input)
        if input.prior_results:
            return Finish(kind="finish")
        return DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Assess the current request.",
                ),
            ),
        )


class _ConversationUnderstandingActor:
    async def understand(
        self, query: str, history: Sequence[ConversationExchange]
    ) -> QueryUnderstandingOutput:
        del history
        return QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(original_query=query, standalone_query=query),
            intent=IntentResult(intent="market_outlook", confidence=0.9),
        )


class _RollingCoordinator:
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
                        objective="Establish the market premise.",
                    ),
                ),
            )
        if len(self.inputs) == 2:
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Assess the premise implications.",
                        context_task_ids=(input.prior_results[0].task_id,),
                    ),
                ),
            )
        return Finish(kind="finish")


class _MaxRollingCoordinator:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, input: CoordinatorInput) -> DispatchBatch | Finish:
        del input
        self.calls += 1
        if self.calls <= 4:
            return DispatchBatch(
                kind="dispatch",
                tasks=tuple(
                    TaskProposal(
                        specialist_id="market-data",
                        objective=(f"Assess rolling dimension {self.calls}-{index}."),
                    )
                    for index in range(8)
                ),
            )
        return Finish(kind="finish")


class _OutcomeDrivenCoordinator:
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
                        objective="Establish the market premise.",
                    ),
                ),
            )
        if len(self.inputs) == 2:
            follow_up_count = 2 if input.failed_tasks else 1
            return DispatchBatch(
                kind="dispatch",
                tasks=tuple(
                    TaskProposal(
                        specialist_id="market-data",
                        objective=f"Assess follow-up {index}.",
                    )
                    for index in range(follow_up_count)
                ),
            )
        return Finish(kind="finish")


class _InvalidCoordinator:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, input: CoordinatorInput) -> DispatchBatch:
        del input
        self.calls += 1
        raise CoordinatorOutputInvalid("invalid output")

    async def repair(
        self,
        input: CoordinatorInput,
        *,
        rejection: str,
    ) -> DispatchBatch:
        del input
        assert rejection == "Coordinator decision is invalid"
        self.calls += 1
        raise CoordinatorOutputInvalid("still invalid")


class _ContextSpecialist:
    def __init__(self) -> None:
        self.inputs: list[SpecialistTaskInput] = []

    async def run(
        self,
        input: SpecialistTaskInput,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> SpecialistAttempt:
        del usage, usage_limits
        self.inputs.append(input)
        if len(self.inputs) == 1:
            assert input.context_results == ()
            return SpecialistAttempt(
                finding=SpecialistFindingDraft(summary="The market premise.")
            )
        assert input.context_results[0].summary == "The market premise."
        return SpecialistAttempt(
            finding=SpecialistFindingDraft(summary="The premise implications.")
        )


class _ConcurrentBatchCoordinator:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, input: CoordinatorInput) -> DispatchBatch | Finish:
        del input
        self.calls += 1
        if self.calls == 1:
            return DispatchBatch(
                kind="dispatch",
                tasks=tuple(
                    TaskProposal(
                        specialist_id="market-data",
                        objective=f"Assess market dimension {index}.",
                    )
                    for index in range(8)
                ),
            )
        return Finish(kind="finish")


class _ConcurrentBatchSpecialist:
    def __init__(
        self,
        *,
        reverse_completion: bool,
        fatal_index: int | None = None,
    ) -> None:
        self.reverse_completion = reverse_completion
        self.fatal_index = fatal_index
        self.active = 0
        self.max_active = 0
        self.inputs: list[SpecialistTaskInput] = []
        self.all_entered = asyncio.Event()

    async def run(
        self,
        input: SpecialistTaskInput,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> SpecialistAttempt:
        del usage, usage_limits
        index = int(input.objective.removesuffix(".").rsplit(" ", 1)[1])
        self.inputs.append(input)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if len(self.inputs) == 8:
                self.all_entered.set()
            await asyncio.wait_for(self.all_entered.wait(), timeout=1)
            delay = index if self.reverse_completion else 7 - index
            await asyncio.sleep(delay / 1000)
            if index == self.fatal_index:
                raise RuntimeError("forced Specialist fatal failure")
            if index == 3:
                raise SpecialistInvocationFailure(
                    UsageLimitExceeded("Task request limit reached"),
                    facts=SpecialistFailureFacts(
                        boundary=SpecialistModelBoundary.AZURE_OPENAI,
                        at_model_request_boundary=True,
                        terminal_output_tool_rejected=False,
                        count_limit_exhausted=True,
                        unreturned_model_requests=0,
                        usage_limits=specialist_usage_limits(),
                    ),
                    messages=(),
                )
            return SpecialistAttempt(
                finding=SpecialistFindingDraft(
                    summary=f"Finding for market dimension {index}."
                )
            )
        finally:
            self.active -= 1


_CONCURRENT_BATCH_POLICY = AgentIntentPolicy(
    intent="market_outlook",
    description="Assess market conditions.",
    specialist_descriptors=(
        SpecialistDescriptor(id="market-data", description="Market data"),
    ),
)
_TENANT_HEADERS = {"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"}


def _concurrent_batch_app(
    database_url: str,
    *,
    coordinator: _ConcurrentBatchCoordinator,
    specialist: SpecialistActor,
    checkpointer_factory: CheckpointerFactory = AsyncPostgresSaver,
) -> FastAPI:
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
            intent_policies={_CONCURRENT_BATCH_POLICY.intent: _CONCURRENT_BATCH_POLICY},
        )

    app = persistent_linear_app(
        database_url,
        agent_runtime_factory=factory,
        checkpointer_factory=checkpointer_factory,
    )
    app.state.tenant_manager = _TenantManager()
    return app


def _post_concurrent_batch_request(
    client: TestClient,
    *,
    conversation_id: str,
    client_request_id: str,
) -> Any:
    return client.post(
        "/v2/query/stream",
        json={
            "query": "What about it?",
            "sessionId": conversation_id,
            "clientRequestId": client_request_id,
        },
        headers=_TENANT_HEADERS,
    )


def _concurrent_batch_checkpoint(
    app: FastAPI,
    client: TestClient,
    conversation_id: str,
) -> Any:
    assert client.portal is not None
    return client.portal.call(
        lambda: app.state.langgraph_v2_checkpointer.aget_tuple(
            thread_checkpoint_config(
                thread_id=thread_id_for(
                    "tenant-a", "subject-a", "agent", conversation_id
                )
            )
        )
    )


class _Specialist:
    def __init__(self) -> None:
        self.inputs: list[SpecialistTaskInput] = []

    async def run(
        self,
        input: SpecialistTaskInput,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> SpecialistAttempt:
        del usage, usage_limits
        self.inputs.append(input)
        return SpecialistAttempt(
            finding=SpecialistFindingDraft(summary="No-tool market finding")
        )


class _RetryExhaustedSpecialist:
    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self,
        input: SpecialistTaskInput,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> SpecialistAttempt:
        del input, usage_limits
        assert usage is not None
        self.calls += 1
        usage.incr(RunUsage(requests=1))
        raise SpecialistInvocationFailure(
            ModelHTTPError(429, "specialist"),
            facts=SpecialistFailureFacts(
                boundary=SpecialistModelBoundary.AZURE_OPENAI,
                at_model_request_boundary=True,
                terminal_output_tool_rejected=False,
                count_limit_exhausted=False,
                unreturned_model_requests=0,
                usage_limits=specialist_usage_limits(),
            ),
            messages=("retry-diagnostic",),
        )


class _OutcomeDrivenSpecialist(_RetryExhaustedSpecialist):
    def __init__(self, *, fail_premise: bool) -> None:
        super().__init__()
        self.fail_premise = fail_premise
        self.inputs: list[SpecialistTaskInput] = []

    async def run(
        self,
        input: SpecialistTaskInput,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> SpecialistAttempt:
        self.inputs.append(input)
        if self.fail_premise and input.objective == "Establish the market premise.":
            return await super().run(input, usage=usage, usage_limits=usage_limits)
        del usage, usage_limits
        return SpecialistAttempt(
            finding=SpecialistFindingDraft(summary=f"Finding for {input.objective}")
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
    tool_capture: SpecialistToolCapture,
    skill_invocation: SkillInvocation | None,
) -> SpecialistActor:
    return _specialist_factory(
        query="Apple revenue",
        summary="Apple revenue grew.",
        evidence_id="evidence-1",
    )(tools, tool_capture, skill_invocation)


def _specialist_factory(
    *, query: str, summary: str, evidence_id: str
) -> SpecialistActorFactory:
    def build(
        tools: tuple[Callable[..., object], ...],
        tool_capture: SpecialistToolCapture,
        skill_invocation: SkillInvocation | None,
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
            tool_capture=tool_capture,
            skill_invocation=skill_invocation,
        )

    return build


def _skill_specialist_factory(
    tools: tuple[Callable[..., object], ...],
    tool_capture: SpecialistToolCapture,
    skill_invocation: SkillInvocation | None,
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
        tool_capture=tool_capture,
        skill_invocation=skill_invocation,
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

    async def repair(
        self, prepared: object, *, validation_errors: tuple[str, ...]
    ) -> FinancialResearchReport:
        del validation_errors
        return await self.synthesize(prepared)


class _EmptySynthesis:
    async def synthesize(self, prepared: object) -> FinancialResearchReport:
        del prepared
        return FinancialResearchReport(
            markdown_report="No matching filing records were found. [[E:1]]"
        )

    async def repair(
        self, prepared: object, *, validation_errors: tuple[str, ...]
    ) -> FinancialResearchReport:
        del validation_errors
        return await self.synthesize(prepared)


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


def test_rolling_rounds_change_dispatch_shape_from_accepted_prior_results(
    langgraph_v2_migrated_database_url: str,
) -> None:
    coordinator = _RollingCoordinator()
    specialist = _ContextSpecialist()
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

    conversation_id = "00000000-0000-0000-0000-000000000096"
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
                "clientRequestId": "rolling-round-request",
            },
            headers=_TENANT_HEADERS,
        )
        checkpoint = _concurrent_batch_checkpoint(app, client, conversation_id)

    events = parse_sse(response.text)
    assert response.status_code == 200
    assert len([event for event in events if event["type"] == "done"]) == 1
    assert len(coordinator.inputs) == 3
    assert coordinator.inputs[1].prior_results[0].summary == "The market premise."
    assert len(specialist.inputs) == 2
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    rounds = state["coordination_rounds"]
    assert [round_["kind"] for round_ in rounds.values()] == [
        "dispatch",
        "dispatch",
        "finish",
    ]
    assert state["active_batch"] is None
    assert len(state["accepted_batches"]) == 2


def test_maximum_legal_rolling_path_completes_within_recursion_limit(
    langgraph_v2_migrated_database_url: str,
) -> None:
    coordinator = _MaxRollingCoordinator()
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

    conversation_id = "00000000-0000-0000-0000-000000000097"
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
                "clientRequestId": "max-rolling-request",
            },
            headers=_TENANT_HEADERS,
        )
        checkpoint = _concurrent_batch_checkpoint(app, client, conversation_id)

    events = parse_sse(response.text)
    assert response.status_code == 200
    assert len([event for event in events if event["type"] == "done"]) == 1
    assert coordinator.calls == 5
    assert len(specialist.inputs) == 32
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    rounds = list(state["coordination_rounds"].values())
    assert [round_["revision"] for round_ in rounds] == [1, 2, 3, 4, 5]
    assert [round_["kind"] for round_ in rounds] == [
        "dispatch",
        "dispatch",
        "dispatch",
        "dispatch",
        "finish",
    ]


def test_next_request_in_one_conversation_resets_rolling_coordination_state(
    langgraph_v2_migrated_database_url: str,
) -> None:
    coordinator = _RequestScopedCoordinator()
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
            query_understanding_actor=_ConversationUnderstandingActor(),
            coordinator_actor=coordinator,
            specialist_registry=registry,
            intent_policies={policy.intent: policy},
        )

    conversation_id = "00000000-0000-0000-0000-000000000101"
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=factory,
    )
    app.state.tenant_manager = _TenantManager()
    with TestClient(app) as client:
        first = client.post(
            "/v2/query/stream",
            json={
                "query": "First request",
                "sessionId": conversation_id,
                "clientRequestId": "request-101a",
            },
            headers=_TENANT_HEADERS,
        )
        second = client.post(
            "/v2/query/stream",
            json={
                "query": "Second request",
                "sessionId": conversation_id,
                "clientRequestId": "request-101b",
            },
            headers=_TENANT_HEADERS,
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert (
        len([event for event in parse_sse(first.text) if event["type"] == "done"]) == 1
    )
    assert (
        len([event for event in parse_sse(second.text) if event["type"] == "done"]) == 1
    )
    assert len(specialist.inputs) == 2
    assert [len(input.prior_results) for input in coordinator.inputs] == [0, 1, 0, 1]


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


def test_retry_exhaustion_promotes_one_failed_task_and_completes_incomplete(
    langgraph_v2_migrated_database_url: str,
) -> None:
    coordinator = _Coordinator()
    specialist = _RetryExhaustedSpecialist()
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

    conversation_id = "00000000-0000-0000-0000-000000000088"
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
    assert specialist.calls == 3
    assert len(done) == 1
    assert done[0]["data"]["metadata"]["termination_reason"] == "partial_results"
    assert "one requested task could not complete" in done[0]["data"]["answer"]
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    accepted = next(iter(state["accepted_batches"].values()))
    assert accepted["outcomes"][0]["kind"] == "failed"
    assert accepted["outcomes"][0]["task_id"].startswith("task_")
    completion = IncompleteResearch.model_validate(state["incomplete_research"])
    assert completion == IncompleteResearch(
        insufficient_evidence=True,
        failed_task_ids=(accepted["outcomes"][0]["task_id"],),
    )
    assert accepted["usage"] == {
        "model_requests": 3,
        "completed_tool_calls": 0,
        "tool_attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
        "cost_is_complete": True,
    }
    assert done[0]["data"]["metadata"]["specialist_usage"] == accepted["usage"]


def test_prior_outcome_changes_the_follow_up_execution_shape(
    langgraph_v2_migrated_database_url: str,
) -> None:
    policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )

    def run_scenario(*, fail_premise: bool, conversation_id: str) -> dict[str, object]:
        coordinator = _OutcomeDrivenCoordinator()
        specialist = _OutcomeDrivenSpecialist(fail_premise=fail_premise)
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
                    "clientRequestId": f"outcome-{fail_premise}",
                },
                headers=_TENANT_HEADERS,
            )
            checkpoint = _concurrent_batch_checkpoint(app, client, conversation_id)

        assert response.status_code == 200
        assert checkpoint is not None
        assert len(coordinator.inputs) == 3
        return checkpoint.checkpoint["channel_values"]

    success = run_scenario(
        fail_premise=False,
        conversation_id="00000000-0000-0000-0000-000000000098",
    )
    failure = run_scenario(
        fail_premise=True,
        conversation_id="00000000-0000-0000-0000-000000000099",
    )

    success_rounds = list(
        cast(dict[str, dict[str, object]], success["coordination_rounds"]).values()
    )
    failure_rounds = list(
        cast(dict[str, dict[str, object]], failure["coordination_rounds"]).values()
    )
    assert len(cast(list[object], success_rounds[1]["tasks"])) == 1
    assert len(cast(list[object], failure_rounds[1]["tasks"])) == 2


def test_rejected_candidates_leave_no_checkpoint_coordination_round(
    langgraph_v2_migrated_database_url: str,
) -> None:
    coordinator = _InvalidCoordinator()
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

    conversation_id = "00000000-0000-0000-0000-000000000100"
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=factory,
    )
    app.state.tenant_manager = _TenantManager()
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "What about it?", "sessionId": conversation_id},
            headers=_TENANT_HEADERS,
        )
        checkpoint = _concurrent_batch_checkpoint(app, client, conversation_id)

    done = [event for event in parse_sse(response.text) if event["type"] == "done"]
    assert response.status_code == 200
    assert len(done) == 1
    assert coordinator.calls == 2
    assert specialist.inputs == []
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    assert state["coordination_rounds"] == {}
    assert state["accepted_batches"] == {}
    assert state["coordination_stop_reason"] == "coordination_invalid"


def test_concurrent_mixed_batch_is_atomically_accepted_in_manifest_order(
    langgraph_v2_migrated_database_url: str,
) -> None:
    def run_batch(
        *, reverse_completion: bool, conversation_id: str
    ) -> dict[str, object]:
        coordinator = _ConcurrentBatchCoordinator()
        specialist = _ConcurrentBatchSpecialist(reverse_completion=reverse_completion)
        app = _concurrent_batch_app(
            langgraph_v2_migrated_database_url,
            coordinator=coordinator,
            specialist=specialist,
        )
        with TestClient(app) as client:
            response = _post_concurrent_batch_request(
                client,
                conversation_id=conversation_id,
                client_request_id="mixed-batch-request",
            )
            checkpoint = _concurrent_batch_checkpoint(app, client, conversation_id)

        events = parse_sse(response.text)
        assert response.status_code == 200
        assert len([event for event in events if event["type"] == "done"]) == 1
        assert coordinator.calls == 2
        assert len(specialist.inputs) == 8
        assert specialist.max_active == 8
        assert checkpoint is not None
        state = checkpoint.checkpoint["channel_values"]
        assert state["active_batch"] is None
        assert state["staged_contributions"] == {}
        accepted = next(iter(state["accepted_batches"].values()))
        assert [outcome["kind"] for outcome in accepted["outcomes"]] == [
            "succeeded",
            "succeeded",
            "succeeded",
            "failed",
            "succeeded",
            "succeeded",
            "succeeded",
            "succeeded",
        ]
        return accepted

    forward = run_batch(
        reverse_completion=False,
        conversation_id="00000000-0000-0000-0000-000000000091",
    )
    reverse = run_batch(
        reverse_completion=True,
        conversation_id="00000000-0000-0000-0000-000000000092",
    )

    assert reverse == forward


def test_calculation_aliases_are_independent_of_specialist_completion_order(
    langgraph_v2_migrated_database_url: str,
) -> None:
    evidence_bodies = tuple(f"Price series {index}." for index in range(8))
    executor = CalculationExecutor(
        tuple(
            TrustedPriceSeries(
                ref=f"series-{index}",
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(
                        as_of=date(2026, 1, 3), value=Decimal(101 + index)
                    ),
                ),
                evidence_refs=(f"evidence-{index}",),
                evidence_hashes=(
                    hashlib.sha256(evidence_bodies[index].encode("utf-8")).hexdigest(),
                ),
            )
            for index in range(8)
        )
    )

    class _ConcurrentCalculationSpecialist:
        def __init__(self, *, reverse_completion: bool) -> None:
            self.reverse_completion = reverse_completion
            self.inputs: list[SpecialistTaskInput] = []
            self.all_entered = asyncio.Event()

        async def run(
            self,
            input: SpecialistTaskInput,
            *,
            usage: RunUsage | None = None,
            usage_limits: UsageLimits | None = None,
        ) -> SpecialistAttempt:
            del usage, usage_limits
            index = int(input.objective.removesuffix(".").rsplit(" ", 1)[1])
            self.inputs.append(input)
            if len(self.inputs) == 8:
                self.all_entered.set()
            await asyncio.wait_for(self.all_entered.wait(), timeout=1)
            delay = index if self.reverse_completion else 7 - index
            await asyncio.sleep(delay / 1000)
            calculation_context = CalculationExecutionContext(
                tenant_id="tenant-a",
                request_id="calculation-order-request",
                task_id=input.task_id,
                attempt=1,
                tool_id="calculator",
            )
            return SpecialistAttempt(
                finding=SpecialistFindingDraft(
                    summary=f"Finding for market dimension {index}.",
                    evidence_ids=(f"evidence-{index}",),
                ),
                calculations=tuple(
                    executor.execute(
                        CalculationRequest(
                            method=method,
                            version="v1",
                            series_ref=f"series-{index}",
                        ),
                        context=calculation_context,
                    )
                    for method in (
                        CalculationMethod.PERIOD_RETURN,
                        CalculationMethod.MAXIMUM_DRAWDOWN,
                    )
                ),
                evidence=(
                    EvidenceEnvelope(
                        id=f"evidence-{index}",
                        tenant_id="tenant-a",
                        request_id="calculation-order-request",
                        task_id=input.task_id,
                        source="filing",
                        source_url="https://example.test/filing",
                        title=f"Price series {index}",
                        body=evidence_bodies[index],
                        excerpt=evidence_bodies[index],
                        as_of_date=date(2026, 9, 6),
                    ),
                ),
            )

    class _CalculationSynthesis:
        def __init__(self) -> None:
            self.prepared: PreparedSynthesis | None = None

        async def synthesize(
            self, prepared: PreparedSynthesis
        ) -> FinancialResearchReport:
            self.prepared = prepared
            return FinancialResearchReport(
                markdown_report=(
                    "First [[C:1]], last [[C:16]]. "
                    "[[E:1]] [[E:2]] [[E:3]] [[E:4]] "
                    "[[E:5]] [[E:6]] [[E:7]] [[E:8]]"
                )
            )

        async def repair(
            self,
            prepared: PreparedSynthesis,
            *,
            validation_errors: tuple[str, ...],
        ) -> FinancialResearchReport:
            del validation_errors
            return await self.synthesize(prepared)

    def run_batch(
        *, reverse_completion: bool, conversation_id: str
    ) -> tuple[dict[str, object], tuple[str, ...], str]:
        coordinator = _ConcurrentBatchCoordinator()
        specialist = _ConcurrentCalculationSpecialist(
            reverse_completion=reverse_completion
        )
        synthesis = _CalculationSynthesis()

        def factory(
            tools: tuple[object, ...],
            tool_capture: object,
            skill_invocation: object,
        ) -> _ConcurrentCalculationSpecialist:
            del tools, tool_capture, skill_invocation
            return specialist

        policy = AgentIntentPolicy(
            intent="market_outlook",
            description="Assess market conditions.",
            allowed_tool_ids=frozenset({"calculator"}),
            specialist_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
            as_of_date=date(2026, 9, 6),
        )
        registry = SpecialistRegistry(
            registrations=(
                SpecialistRegistration(
                    id="market-data",
                    actor_factory=factory,
                    allowed_tool_ids=frozenset({"calculator"}),
                ),
            ),
            tenant_eligible_ids=frozenset({"market-data"}),
            calculation_tool_registrations=(
                CalculationToolRegistration(id="calculator", executor=executor),
            ),
            tenant_eligible_tool_ids=frozenset({"calculator"}),
        )

        def app_factory(
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
                synthesis_actor=synthesis,
            )

        app = persistent_linear_app(
            langgraph_v2_migrated_database_url,
            agent_runtime_factory=app_factory,
        )
        app.state.tenant_manager = _TenantManager()
        with TestClient(app) as client:
            response = client.post(
                "/v2/query/stream",
                json={
                    "query": "What about it?",
                    "sessionId": conversation_id,
                    "clientRequestId": "calculation-order-request",
                },
                headers=_TENANT_HEADERS,
            )
            checkpoint = _concurrent_batch_checkpoint(app, client, conversation_id)

        done = [event for event in parse_sse(response.text) if event["type"] == "done"]
        assert response.status_code == 200
        assert len(done) == 1
        assert coordinator.calls == 2
        assert len(specialist.inputs) == 8
        assert checkpoint is not None
        assert synthesis.prepared is not None
        accepted = next(
            iter(checkpoint.checkpoint["channel_values"]["accepted_batches"].values())
        )
        artifacts = synthesis.prepared.calculation_artifacts
        assert tuple(item.alias for item in synthesis.prepared.calculations) == tuple(
            f"C:{index}" for index in range(1, 17)
        )
        first, last = artifacts[0], artifacts[-1]
        first_rendered = (
            f"{first.formatted_value} ({first.method.value.replace('_', ' ')}; "
            f"{first.unit}; {first.currency}; {first.period_start} to "
            f"{first.period_end}; assumptions: {'; '.join(first.assumptions)})"
        )
        last_rendered = (
            f"{last.formatted_value} ({last.method.value.replace('_', ' ')}; "
            f"{last.unit}; {last.currency}; {last.period_start} to "
            f"{last.period_end}; assumptions: {'; '.join(last.assumptions)})"
        )
        assert done[0]["data"]["answer"] == (
            f"First {first_rendered}, last {last_rendered}. "
            "[[E:1]] [[E:2]] [[E:3]] [[E:4]] "
            "[[E:5]] [[E:6]] [[E:7]] [[E:8]]"
        )
        return (
            accepted,
            tuple(artifact.id for artifact in artifacts),
            done[0]["data"]["answer"],
        )

    forward = run_batch(
        reverse_completion=False,
        conversation_id="00000000-0000-0000-0000-000000000a11",
    )
    reverse = run_batch(
        reverse_completion=True,
        conversation_id="00000000-0000-0000-0000-000000000a12",
    )

    assert reverse == forward


def test_batch_barrier_checkpoint_failure_never_half_accepts_a_mixed_batch(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class _BarrierFailureSaver(AsyncPostgresSaver):
        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Any,
            metadata: Any,
            new_versions: Any,
        ) -> RunnableConfig:
            if checkpoint.get("channel_values", {}).get("accepted_batches"):
                raise RuntimeError("forced batch barrier checkpoint failure")
            return await super().aput(config, checkpoint, metadata, new_versions)

    coordinator = _ConcurrentBatchCoordinator()
    specialist = _ConcurrentBatchSpecialist(reverse_completion=False)
    conversation_id = "00000000-0000-0000-0000-000000000093"
    app = _concurrent_batch_app(
        langgraph_v2_migrated_database_url,
        coordinator=coordinator,
        specialist=specialist,
        checkpointer_factory=_BarrierFailureSaver,
    )
    with TestClient(app) as client:
        response = _post_concurrent_batch_request(
            client,
            conversation_id=conversation_id,
            client_request_id="checkpoint-failure-request",
        )
        checkpoint = _concurrent_batch_checkpoint(app, client, conversation_id)

    events = parse_sse(response.text)
    assert all(event["type"] != "done" for event in events)
    assert events[-1] == {
        "type": "error",
        "data": "forced batch barrier checkpoint failure",
    }
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    assert state["accepted_batches"] == {}
    assert checkpoint.pending_writes
    barrier_task_ids = {
        write[0]
        for write in checkpoint.pending_writes
        if write[1] == "accepted_batches"
    }
    assert len(barrier_task_ids) == 1
    barrier_channels = {
        write[1] for write in checkpoint.pending_writes if write[0] in barrier_task_ids
    }
    assert {
        "accepted_batches",
        "active_batch",
        "staged_contributions",
    } <= barrier_channels


def test_fatal_specialist_failure_keeps_sibling_contributions_unaccepted(
    langgraph_v2_migrated_database_url: str,
) -> None:
    coordinator = _ConcurrentBatchCoordinator()
    specialist = _ConcurrentBatchSpecialist(
        reverse_completion=False,
        fatal_index=3,
    )
    conversation_id = "00000000-0000-0000-0000-000000000094"
    app = _concurrent_batch_app(
        langgraph_v2_migrated_database_url,
        coordinator=coordinator,
        specialist=specialist,
    )
    with TestClient(app) as client:
        response = _post_concurrent_batch_request(
            client,
            conversation_id=conversation_id,
            client_request_id="fatal-batch-request",
        )
        checkpoint = _concurrent_batch_checkpoint(app, client, conversation_id)

    events = parse_sse(response.text)
    assert all(event["type"] != "done" for event in events)
    assert events[-1] == {"type": "error", "data": "forced Specialist fatal failure"}
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    assert state["accepted_batches"] == {}
    assert checkpoint.pending_writes
    assert any(
        write[1] == "staged_contributions" for write in checkpoint.pending_writes
    )
    assert all(write[1] != "accepted_batches" for write in checkpoint.pending_writes)


@pytest.mark.asyncio
async def test_cancelling_a_concurrent_batch_never_accepts_any_sibling(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class _BlockingSpecialist:
        def __init__(self) -> None:
            self.inputs: list[SpecialistTaskInput] = []
            self.all_entered = asyncio.Event()
            self.release = asyncio.Event()

        async def run(
            self,
            input: SpecialistTaskInput,
            *,
            usage: RunUsage | None = None,
            usage_limits: UsageLimits | None = None,
        ) -> SpecialistAttempt:
            del usage, usage_limits
            self.inputs.append(input)
            if len(self.inputs) == 8:
                self.all_entered.set()
            await self.release.wait()
            return SpecialistAttempt(
                finding=SpecialistFindingDraft(summary=input.objective)
            )

    coordinator = _ConcurrentBatchCoordinator()
    specialist = _BlockingSpecialist()
    conversation_id = UUID("00000000-0000-0000-0000-000000000095")
    app = _concurrent_batch_app(
        langgraph_v2_migrated_database_url,
        coordinator=coordinator,
        specialist=specialist,
    )
    request_context = TrustedRequestContext(
        tenant_id="tenant-a", subject_id="subject-a"
    )

    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            payload=V2QueryRequest(
                query="What about it?", conversation_id=conversation_id
            ),
            http_request=stream_request(app),
            request_context=request_context,
        )
        subscriber = response.body_iterator
        frames: list[str] = []

        async def consume() -> None:
            frames.extend([frame async for frame in subscriber])

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(specialist.all_entered.wait(), timeout=1)
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await consumer
        await subscriber.aclose()
        checkpoint = await app.state.langgraph_v2_checkpointer.aget_tuple(
            thread_checkpoint_config(
                thread_id=thread_id_for(
                    "tenant-a", "subject-a", "agent", str(conversation_id)
                )
            )
        )

    assert len(specialist.inputs) == 8
    assert all('"type":"done"' not in frame for frame in frames)
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    assert state["accepted_batches"] == {}
    assert all(write[1] != "accepted_batches" for write in checkpoint.pending_writes)


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
    assert "".join(event["data"] for event in events if event["type"] == "token") == done[
        0
    ]["data"]["answer"]
    assert [event["data"] for event in events if event["type"] == "citations"] == [
        done[0]["data"]["citations"]
    ]
    event_types = [event["type"] for event in events]
    assert event_types.index("token") < event_types.index("citations") < event_types.index(
        "done"
    )
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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepare", "synthesis", "gate"])
async def test_cancellation_before_finalization_never_publishes_research_state(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    original_prepare = agent_graph.prepare_synthesis
    original_synthesize = agent_graph.synthesize_report

    def barrier_prepare(**kwargs: Any) -> PreparedSynthesis:
        prepared = original_prepare(**kwargs)
        if phase == "prepare":
            entered.set()
        return prepared

    async def barrier_synthesize(
        actor: SynthesisActor,
        prepared: PreparedSynthesis,
    ) -> PublishedReport:
        published = await original_synthesize(actor, prepared)
        if phase == "gate":
            entered.set()
            await release.wait()
        return published

    class _BarrierSynthesis:
        async def synthesize(self, prepared: PreparedSynthesis) -> FinancialResearchReport:
            del prepared
            if phase == "synthesis":
                entered.set()
            if phase in {"prepare", "synthesis"}:
                await release.wait()
            return FinancialResearchReport(
                markdown_report="Apple revenue grew. [[E:1]]"
            )

        async def repair(
            self,
            prepared: PreparedSynthesis,
            *,
            validation_errors: tuple[str, ...],
        ) -> FinancialResearchReport:
            del validation_errors
            return await self.synthesize(prepared)

    monkeypatch.setattr(agent_graph, "prepare_synthesis", barrier_prepare)
    monkeypatch.setattr(agent_graph, "synthesize_report", barrier_synthesize)
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
            synthesis_actor=_BarrierSynthesis(),
        )

    conversation_id = UUID(
        {
            "prepare": "00000000-0000-0000-0000-000000000068",
            "synthesis": "00000000-0000-0000-0000-000000000069",
            "gate": "00000000-0000-0000-0000-000000000070",
        }[phase]
    )
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=factory,
    )
    app.state.tenant_manager = _TenantManager()
    request_context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")

    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            payload=V2QueryRequest(
                query="What about it?",
                conversation_id=conversation_id,
                client_request_id="request-1",
            ),
            http_request=stream_request(app),
            request_context=request_context,
        )
        subscriber = response.body_iterator
        frames: list[str] = []

        async def consume() -> None:
            frames.extend([frame async for frame in subscriber])

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), timeout=1)
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await consumer
        release.set()
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

    events = [event for frame in frames for event in parse_sse(frame)]
    assert all(event["type"] not in {"token", "citations", "done"} for event in events)
    assert [(message.id, message.type, message.text) for message in messages] == [
        ("request-1:user", "human", "What about it?"),
    ]


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
        tool_capture: SpecialistToolCapture,
        skill_invocation: SkillInvocation | None,
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
            tool_capture=tool_capture,
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

        async def repair(
            self,
            prepared: PreparedSynthesis,
            *,
            validation_errors: tuple[str, ...],
        ) -> FinancialResearchReport:
            del validation_errors
            return await self.synthesize(prepared)

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
        "Incomplete research: requested data was unavailable:\n- Apple revenue\n\n"
        "Apple revenue grew. [[E:1]]"
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
