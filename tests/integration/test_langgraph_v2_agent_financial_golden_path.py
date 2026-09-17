"""External deterministic financial golden-path coverage for Agent mode."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, cast
from uuid import UUID

import pydantic_ai.models as pydantic_models
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage, UsageLimits

from app.agents.specialist import (
    SPECIALIST_OUTPUT_RETRIES,
    PydanticAISpecialistActor,
)
from app.agents.synthesis import (
    PydanticAISynthesisActor,
    create_synthesis_agent,
)
from app.config.models import FlowConfig, LangGraphRuntimeMode, LLMConfig, TenantConfig
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import (
    CalculationToolRegistration,
    DispatchBatch,
    EvidenceToolRegistration,
    SpecialistActor,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistTaskInput,
    TaskProposal,
    task_id_for,
)
from app.langgraph_v2.agent_coordination import (
    CoordinatorActor,
    CoordinatorInput,
    Finish,
)
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    PreparedSynthesis,
    SpecialistToolCapture,
)
from app.langgraph_v2.agent_graph import QueryUnderstandingActor, SynthesisActor
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
    CalculationExecutor,
    CalculationMethod,
    PriceObservation,
    TrustedPriceSeries,
)
from app.langgraph_v2.checkpointing import (
    AgentCheckpointStateAdapter,
    read_conversation_messages,
    thread_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.specialist_retry import (
    SpecialistFailureFacts,
    SpecialistInvocationFailure,
    SpecialistModelBoundary,
    specialist_usage_limits,
)
from app.models.workflow import IntentResult, QueryUnderstandingOutput, ResolvedQuery
from tests.integration.test_langgraph_v2_linear_core import (
    CheckpointerFactory,
    parse_sse,
    persistent_linear_app,
)

_AS_OF_DATE = date(2026, 9, 6)
_FUND_ID = "FUND-ALPHA"
_BENCHMARK_ID = "BENCHMARK-OMEGA"
_REQUEST_ID = "financial-golden-request"
_MARKET_OBJECTIVE = f"Analyze {_FUND_ID} against {_BENCHMARK_ID}."
_FUND_OBJECTIVE = f"Research {_FUND_ID} holdings and disclosures."
_NEWS_OBJECTIVE = f"Review company news selected from {_FUND_ID} research."


@dataclass(frozen=True)
class FinancialFixtureContent:
    """Optional untrusted-content fixtures for financial integration coverage."""

    evidence_bodies: Mapping[str, str] = field(default_factory=dict[str, str])
    evidence_excerpts: Mapping[str, str] = field(default_factory=dict[str, str])
    result_summaries: Mapping[str, str] = field(default_factory=dict[str, str])
    skill_references: Mapping[str, str] = field(default_factory=dict[str, str])
    raw_provider_payload: str = "RAW-PROVIDER-PAYLOAD-MUST-NOT-PERSIST"


@pytest.fixture(autouse=True)
def disable_real_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the external path deterministic even if a fixture regresses."""
    monkeypatch.setattr(pydantic_models, "ALLOW_MODEL_REQUESTS", False)


def _task_id(*, request_id: str = _REQUEST_ID, round: int, dispatch_order: int) -> str:
    return task_id_for(
        request_id=request_id,
        round=round,
        dispatch_order=dispatch_order,
    )


def _prompt_body(messages: Sequence[ModelMessage]) -> dict[str, Any]:
    prompt = next(
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if part.part_kind == "user-prompt"
    )
    assert isinstance(prompt, str)
    payload = cast(object, json.loads(prompt))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


@dataclass
class _FinancialTools:
    """Four fixed business-Tool providers and a deterministic fan-out barrier."""

    request_id: str = _REQUEST_ID
    fund_task_dispatch_order: int = 1
    synchronize_first_batch: bool = True
    content: FinancialFixtureContent = field(default_factory=FinancialFixtureContent)
    price_entered: asyncio.Event = field(default_factory=asyncio.Event)
    holdings_entered: asyncio.Event = field(default_factory=asyncio.Event)
    provider_calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    async def price_series(self, source: str, query: str) -> EvidenceEnvelope:
        assert (source, query) == (
            "market",
            f"{_FUND_ID} versus {_BENCHMARK_ID}",
        )
        self.provider_calls.append((source, query))
        self.price_entered.set()
        if self.synchronize_first_batch:
            await asyncio.wait_for(self.holdings_entered.wait(), timeout=1)
        return self._evidence(
            evidence_id="price-evidence",
            request_id=self.request_id,
            task_id=_task_id(
                request_id=self.request_id,
                round=1,
                dispatch_order=0,
            ),
            source=source,
            query=query,
            body="fixed closes: 100, 110, 99",
        )

    async def fund_holdings(self, source: str, query: str) -> EvidenceEnvelope:
        assert (source, query) == ("fund", f"{_FUND_ID} holdings")
        self.provider_calls.append((source, query))
        self.holdings_entered.set()
        if self.synchronize_first_batch:
            await asyncio.wait_for(self.price_entered.wait(), timeout=1)
        return self._evidence(
            evidence_id="holdings-evidence",
            request_id=self.request_id,
            task_id=_task_id(
                request_id=self.request_id,
                round=1,
                dispatch_order=self.fund_task_dispatch_order,
            ),
            source=source,
            query=query,
            body="fixed holdings: ALPHA 60%, BETA 40%",
        )

    async def fund_reports(self, source: str, query: str) -> EvidenceEnvelope:
        assert (source, query) == ("fund", f"{_FUND_ID} report")
        self.provider_calls.append((source, query))
        return self._evidence(
            evidence_id="report-evidence",
            request_id=self.request_id,
            task_id=_task_id(
                request_id=self.request_id,
                round=1,
                dispatch_order=self.fund_task_dispatch_order,
            ),
            source=source,
            query=query,
            body="fixed disclosure: quarterly report",
        )

    async def company_news(self, source: str, query: str) -> EvidenceEnvelope:
        assert (source, query) == ("news", f"{_FUND_ID} company news")
        self.provider_calls.append((source, query))
        return self._evidence(
            evidence_id="news-evidence",
            request_id=self.request_id,
            task_id=_task_id(
                request_id=self.request_id,
                round=2,
                dispatch_order=0,
            ),
            source=source,
            query=query,
            body="fixed news: portfolio company announced earnings",
        )

    def _evidence(
        self,
        *,
        evidence_id: str,
        request_id: str,
        task_id: str,
        source: str,
        query: str,
        body: str,
    ) -> EvidenceEnvelope:
        return EvidenceEnvelope(
            id=evidence_id,
            tenant_id="tenant-a",
            request_id=request_id,
            task_id=task_id,
            source=source,
            source_url=f"https://fixture.test/{evidence_id}",
            title=query,
            body=self.content.evidence_bodies.get(evidence_id, body),
            excerpt=self.content.evidence_excerpts.get(
                evidence_id, f"Fixture evidence for {query}."
            ),
            as_of_date=_AS_OF_DATE,
            raw_provider_payload=self.content.raw_provider_payload,
        )


@dataclass
class _FinancialUnderstanding:
    history_views: list[list[object]] = field(default_factory=list[list[object]])

    async def understand(
        self,
        query: str,
        history: Sequence[object],
    ) -> QueryUnderstandingOutput:
        self.history_views.append(list(history))
        assert _FUND_ID in query and _BENCHMARK_ID in query
        return QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query=query,
                standalone_query=f"Analyze {_FUND_ID} against {_BENCHMARK_ID} as of 2026-09-06.",
            ),
            intent=IntentResult(intent="financial_golden_path", confidence=1.0),
        )


@dataclass
class _FinancialCoordinator:
    inputs: list[CoordinatorInput] = field(default_factory=list[CoordinatorInput])
    follow_up_context_ids: list[tuple[str, ...]] = field(
        default_factory=list[tuple[str, ...]]
    )

    async def decide(self, input: CoordinatorInput) -> DispatchBatch | Finish:
        self.inputs.append(input)
        assert "skill" not in input.model_dump_json().lower()
        if not input.prior_results and not input.failed_tasks:
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-analysis",
                        objective=_MARKET_OBJECTIVE,
                    ),
                    TaskProposal(
                        specialist_id="fund-research",
                        objective=_FUND_OBJECTIVE,
                    ),
                ),
            )
        if input.failed_tasks:
            assert len(input.prior_results) == 1
            assert input.failed_tasks[0].objective == _FUND_OBJECTIVE
            return Finish(kind="finish")
        if len(input.prior_results) == 2:
            fund_result = next(
                result
                for result in input.prior_results
                if result.summary == "Fund holdings and disclosure research completed."
            )
            self.follow_up_context_ids.append((fund_result.task_id,))
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="fund-research",
                        objective=_NEWS_OBJECTIVE,
                        context_task_ids=(fund_result.task_id,),
                    ),
                ),
            )
        assert len(input.prior_results) == 3
        return Finish(kind="finish")


@dataclass
class _FailingHoldingsActor:
    """Produce the spec's stronger alternate TaskFailed outcome."""

    fallback: SpecialistActor
    failing_task_ids: frozenset[str]
    diagnostic_message: str
    emitted_diagnostics: list[str]

    async def run(
        self,
        input: SpecialistTaskInput,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> Any:
        if (
            input.objective != _FUND_OBJECTIVE
            or input.task_id not in self.failing_task_ids
        ):
            return await self.fallback.run(
                input, usage=usage, usage_limits=usage_limits
            )
        assert usage is not None
        usage.incr(RunUsage(requests=1))
        self.emitted_diagnostics.append(self.diagnostic_message)
        raise SpecialistInvocationFailure(
            ModelHTTPError(429, "fixture holdings boundary"),
            facts=SpecialistFailureFacts(
                boundary=SpecialistModelBoundary.AZURE_OPENAI,
                at_model_request_boundary=True,
                terminal_output_tool_rejected=False,
                count_limit_exhausted=False,
                unreturned_model_requests=0,
                usage_limits=specialist_usage_limits(),
            ),
            messages=(self.diagnostic_message,),
        )


@dataclass
class _FinancialSpecialists:
    """FunctionModel-backed Specialists for the two fixed roster entries."""

    alternate_holdings_failure: bool = False
    failing_task_ids: frozenset[str] = frozenset()
    failure_diagnostic: str = "fixture holdings retry"
    result_summaries: Mapping[str, str] = field(default_factory=dict[str, str])
    emitted_failure_diagnostics: list[str] = field(default_factory=list[str])
    skill_views: list[tuple[str, tuple[str, ...]]] = field(
        default_factory=list[tuple[str, tuple[str, ...]]]
    )
    activated_instruction_views: list[str] = field(default_factory=list[str])
    task_prompts: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    message_views: list[str] = field(default_factory=list[str])
    business_tool_views: list[frozenset[str]] = field(
        default_factory=list[frozenset[str]]
    )

    def market_analysis(
        self,
        tools: tuple[Callable[..., object], ...],
        tool_capture: SpecialistToolCapture,
        skill_invocation: SkillInvocation | None,
    ) -> SpecialistActor:
        return self._actor(
            role="market",
            tools=tools,
            tool_capture=tool_capture,
            skill_invocation=skill_invocation,
        )

    def fund_research(
        self,
        tools: tuple[Callable[..., object], ...],
        tool_capture: SpecialistToolCapture,
        skill_invocation: SkillInvocation | None,
    ) -> SpecialistActor:
        actor = self._actor(
            role="fund",
            tools=tools,
            tool_capture=tool_capture,
            skill_invocation=skill_invocation,
        )
        return (
            _FailingHoldingsActor(
                actor,
                failing_task_ids=self.failing_task_ids,
                diagnostic_message=self.failure_diagnostic,
                emitted_diagnostics=self.emitted_failure_diagnostics,
            )
            if self.alternate_holdings_failure
            else actor
        )

    def _actor(
        self,
        *,
        role: str,
        tools: tuple[Callable[..., object], ...],
        tool_capture: SpecialistToolCapture,
        skill_invocation: SkillInvocation | None,
    ) -> PydanticAISpecialistActor:
        calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            self.message_views.append(repr(messages))
            prompt = _prompt_body(messages)
            if calls == 1:
                self.task_prompts.append(prompt)
            objective = prompt["objective"]
            assert isinstance(objective, str)
            summaries = prompt["skill_summaries"]
            assert isinstance(summaries, list)
            summary_names = tuple(
                cast(str, item["name"])
                for item in cast(list[Mapping[str, object]], summaries)
            )
            expected_summaries = (
                ("financial-common", "market-methodology")
                if role == "market"
                else ("financial-common", "fund-disclosure")
            )
            assert summary_names == expected_summaries
            business_tools = {
                tool.name
                for tool in info.function_tools
                if tool.name != "activate_skill"
            }
            expected_tools = (
                {
                    "price_series",
                    CalculationMethod.PERIOD_RETURN,
                    CalculationMethod.ANNUALIZED_VOLATILITY,
                    CalculationMethod.MAXIMUM_DRAWDOWN,
                }
                if role == "market"
                else {"fund_holdings", "fund_reports", "company_news"}
            )
            assert business_tools == expected_tools
            self.business_tool_views.append(frozenset(business_tools))
            if calls == 1:
                self.skill_views.append((objective, summary_names))
                assert "FULL-" not in repr(messages)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="activate_skill",
                            args={
                                "skill_name": (
                                    "financial-common"
                                    if role == "market"
                                    else "fund-disclosure"
                                )
                            },
                        )
                    ]
                )

            activated_view = repr(messages)
            self.activated_instruction_views.append(activated_view)
            required_instruction = (
                "FULL-COMMON-SKILL-INSTRUCTIONS"
                if role == "market"
                else "FULL-FUND-SKILL-INSTRUCTIONS"
            )
            assert required_instruction in activated_view
            if role == "market":
                sequence = (
                    (
                        "price_series",
                        {
                            "source": "market",
                            "query": f"{_FUND_ID} versus {_BENCHMARK_ID}",
                        },
                    ),
                    (
                        CalculationMethod.PERIOD_RETURN,
                        {
                            "method": "period_return",
                            "version": "v1",
                            "series_ref": "fund-alpha-series",
                        },
                    ),
                    (
                        CalculationMethod.ANNUALIZED_VOLATILITY,
                        {
                            "method": "annualized_volatility",
                            "version": "v1",
                            "series_ref": "fund-alpha-series",
                        },
                    ),
                    (
                        CalculationMethod.MAXIMUM_DRAWDOWN,
                        {
                            "method": "maximum_drawdown",
                            "version": "v1",
                            "series_ref": "fund-alpha-series",
                        },
                    ),
                )
                if calls <= len(sequence) + 1:
                    tool_name, args = sequence[calls - 2]
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool_name, args=args)]
                    )
                evidence_ids = ["price-evidence"]
                summary = self.result_summaries.get(
                    objective, "Market analysis completed."
                )
            elif objective == _FUND_OBJECTIVE:
                sequence = (
                    (
                        "fund_holdings",
                        {"source": "fund", "query": f"{_FUND_ID} holdings"},
                    ),
                    ("fund_reports", {"source": "fund", "query": f"{_FUND_ID} report"}),
                )
                if calls <= len(sequence) + 1:
                    tool_name, args = sequence[calls - 2]
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool_name, args=args)]
                    )
                evidence_ids = ["holdings-evidence", "report-evidence"]
                summary = self.result_summaries.get(
                    objective, "Fund holdings and disclosure research completed."
                )
            else:
                assert objective == _NEWS_OBJECTIVE
                raw_context_results = cast(object, prompt["context_results"])
                assert isinstance(raw_context_results, list)
                context_results = cast(list[object], raw_context_results)
                assert len(context_results) == 1
                context_result = cast(Mapping[str, object], context_results[0])
                assert context_result["summary"] == (
                    "Fund holdings and disclosure research completed."
                )
                if calls == 2:
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name="company_news",
                                args={
                                    "source": "news",
                                    "query": f"{_FUND_ID} company news",
                                },
                            )
                        ]
                    )
                evidence_ids = ["news-evidence"]
                summary = self.result_summaries.get(
                    objective, "Company-news follow-up completed."
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args={"summary": summary, "evidence_ids": evidence_ids},
                    )
                ]
            )

        return PydanticAISpecialistActor(
            Agent(
                FunctionModel(model),
                output_type=SpecialistFindingDraft,
                tools=tools,
                tool_retries=0,
                output_retries=SPECIALIST_OUTPUT_RETRIES,
                end_strategy="early",
            ),
            tool_capture=tool_capture,
            skill_invocation=skill_invocation,
        )


class _SynthesisModelRegistry:
    def __init__(self, model: FunctionModel) -> None:
        self.model = model

    def create_agent(self, model_name: str, **kwargs: Any) -> object:
        assert model_name == "synthesis"
        return Agent(self.model, **kwargs)


@dataclass
class _FinancialSynthesis:
    prepared_inputs: list[PreparedSynthesis] = field(
        default_factory=list[PreparedSynthesis]
    )

    def actor(self) -> PydanticAISynthesisActor:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            payload = _prompt_body(messages)
            assert "calculation_artifacts" not in payload
            prepared = PreparedSynthesis.model_validate(payload)
            self.prepared_inputs.append(prepared)
            assert prepared.standalone_query == (
                f"Analyze {_FUND_ID} against {_BENCHMARK_ID} as of 2026-09-06."
            )
            aliases = {item.method: item.alias for item in prepared.calculations}
            assert set(aliases) == {
                CalculationMethod.PERIOD_RETURN,
                CalculationMethod.ANNUALIZED_VOLATILITY,
                CalculationMethod.MAXIMUM_DRAWDOWN,
            }
            evidence_markers = " ".join(
                f"[[E:{index}]]" for index in range(1, len(prepared.evidence) + 1)
            )
            report = (
                f"{_FUND_ID} period return is [[{aliases[CalculationMethod.PERIOD_RETURN]}]], "
                f"annualized volatility is [[{aliases[CalculationMethod.ANNUALIZED_VOLATILITY]}]], "
                f"and maximum drawdown is [[{aliases[CalculationMethod.MAXIMUM_DRAWDOWN]}]]. "
                f"The fixed research record is supported by {evidence_markers}."
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args={"markdown_report": report},
                    )
                ]
            )

        model_registry = _SynthesisModelRegistry(FunctionModel(model))
        return PydanticAISynthesisActor(
            create_synthesis_agent(
                cast(ModelRegistry, model_registry), model_name="synthesis"
            )
        )


class _FinancialTenantManager:
    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        assert tenant_id == "tenant-a"
        return TenantConfig(
            kms_app_name="Financial Golden Tenant",
            application_id="tenant-a",
            ad_groups=[],
            runtime_mode=LangGraphRuntimeMode.AGENT,
            llm_config=LLMConfig(models={}),
            flow_config=FlowConfig(),
        )

    def get_providers(self, tenant_id: str) -> object:
        assert tenant_id == "tenant-a"
        return object()


@dataclass
class FinancialFixture:
    alternate_holdings_failure: bool = False
    request_id: str = _REQUEST_ID
    failure_request_ids: frozenset[str] | None = None
    failure_diagnostic: str = "fixture holdings retry"
    content: FinancialFixtureContent = field(default_factory=FinancialFixtureContent)
    tools: _FinancialTools = field(default_factory=_FinancialTools)
    understanding: _FinancialUnderstanding = field(
        default_factory=_FinancialUnderstanding
    )
    coordinator: _FinancialCoordinator = field(default_factory=_FinancialCoordinator)
    specialists: _FinancialSpecialists = field(init=False)
    synthesis: _FinancialSynthesis = field(default_factory=_FinancialSynthesis)

    def __post_init__(self) -> None:
        self.configure_run(request_id=self.request_id, fund_task_dispatch_order=1)
        self.tools.content = self.content
        self.tools.synchronize_first_batch = not self.alternate_holdings_failure
        failure_request_ids = (
            self.failure_request_ids
            if self.failure_request_ids is not None
            else frozenset({self.request_id})
        )
        self.specialists = _FinancialSpecialists(
            alternate_holdings_failure=self.alternate_holdings_failure,
            failing_task_ids=frozenset(
                _task_id(request_id=request_id, round=1, dispatch_order=1)
                for request_id in failure_request_ids
            ),
            failure_diagnostic=self.failure_diagnostic,
            result_summaries=self.content.result_summaries,
        )

    def configure_run(
        self,
        *,
        request_id: str,
        fund_task_dispatch_order: int,
    ) -> None:
        """Bind the reusable fixture's Tool provenance to one scripted Run."""
        self.tools.request_id = request_id
        self.tools.fund_task_dispatch_order = fund_task_dispatch_order

    def registry(self) -> SpecialistRegistry:
        price_body = "fixed closes: 100, 110, 99"
        calculator = CalculationExecutor(
            (
                TrustedPriceSeries(
                    ref="fund-alpha-series",
                    instrument_id=_FUND_ID,
                    currency="USD",
                    unit="close",
                    observations=(
                        PriceObservation(as_of=date(2026, 9, 4), value=Decimal("100")),
                        PriceObservation(as_of=date(2026, 9, 5), value=Decimal("110")),
                        PriceObservation(as_of=_AS_OF_DATE, value=Decimal("99")),
                    ),
                    evidence_refs=("price-evidence",),
                    evidence_hashes=(
                        hashlib.sha256(price_body.encode("utf-8")).hexdigest(),
                    ),
                ),
            )
        )
        return SpecialistRegistry(
            registrations=(
                SpecialistRegistration(
                    id="market-analysis",
                    actor_factory=self.specialists.market_analysis,
                    allowed_tool_ids=frozenset(
                        {
                            "price_series",
                            CalculationMethod.PERIOD_RETURN,
                            CalculationMethod.ANNUALIZED_VOLATILITY,
                            CalculationMethod.MAXIMUM_DRAWDOWN,
                        }
                    ),
                    allowed_skill_names=frozenset({"market-methodology"}),
                ),
                SpecialistRegistration(
                    id="fund-research",
                    actor_factory=self.specialists.fund_research,
                    allowed_tool_ids=frozenset(
                        {"fund_holdings", "fund_reports", "company_news"}
                    ),
                    allowed_skill_names=frozenset({"fund-disclosure"}),
                ),
            ),
            tenant_eligible_ids=frozenset({"market-analysis", "fund-research"}),
            tool_registrations=(
                EvidenceToolRegistration(
                    id="price_series",
                    provider=self.tools.price_series,
                    allowed_sources=frozenset({"market"}),
                    allowed_queries=frozenset({f"{_FUND_ID} versus {_BENCHMARK_ID}"}),
                ),
                EvidenceToolRegistration(
                    id="fund_holdings",
                    provider=self.tools.fund_holdings,
                    allowed_sources=frozenset({"fund"}),
                    allowed_queries=frozenset({f"{_FUND_ID} holdings"}),
                ),
                EvidenceToolRegistration(
                    id="fund_reports",
                    provider=self.tools.fund_reports,
                    allowed_sources=frozenset({"fund"}),
                    allowed_queries=frozenset({f"{_FUND_ID} report"}),
                ),
                EvidenceToolRegistration(
                    id="company_news",
                    provider=self.tools.company_news,
                    allowed_sources=frozenset({"news"}),
                    allowed_queries=frozenset({f"{_FUND_ID} company news"}),
                ),
            ),
            calculation_tool_registrations=tuple(
                CalculationToolRegistration(id=method, executor=calculator)
                for method in CalculationMethod
            ),
            tenant_eligible_tool_ids=frozenset(
                {
                    "price_series",
                    "fund_holdings",
                    "fund_reports",
                    "company_news",
                    *CalculationMethod,
                }
            ),
            skill_registry=SpecialistSkillRegistry(
                registrations=(
                    SkillRegistration(
                        name="financial-common",
                        version="2026.09",
                        description="Use fixed financial fixture identifiers.",
                        instructions="FULL-COMMON-SKILL-INSTRUCTIONS",
                        required_tool_ids=frozenset({"price_series"}),
                        references=(
                            SkillReference(
                                name="common-reference",
                                content=self.content.skill_references.get(
                                    "financial-common",
                                    "FULL-COMMON-SKILL-REFERENCE",
                                ),
                            ),
                        ),
                    ),
                    SkillRegistration(
                        name="market-methodology",
                        version="2026.09",
                        description="Interpret registered market calculations.",
                        instructions="FULL-MARKET-SKILL-INSTRUCTIONS",
                    ),
                    SkillRegistration(
                        name="fund-disclosure",
                        version="2026.09",
                        description="Read fund holdings and disclosures.",
                        instructions="FULL-FUND-SKILL-INSTRUCTIONS",
                        required_tool_ids=frozenset({"fund_holdings", "fund_reports"}),
                        references=(
                            SkillReference(
                                name="fund-reference",
                                content=self.content.skill_references.get(
                                    "fund-disclosure",
                                    "FULL-FUND-SKILL-REFERENCE",
                                ),
                            ),
                        ),
                    ),
                ),
                tenant_eligible_names=frozenset(
                    {"financial-common", "market-methodology", "fund-disclosure"}
                ),
                shared_skill_names=frozenset({"financial-common"}),
            ),
        )

    def policy(self) -> AgentIntentPolicy:
        return AgentIntentPolicy(
            intent="financial_golden_path",
            description="Fixed financial fan-out/fan-in fixture.",
            specialist_descriptors=(
                SpecialistDescriptor(
                    id="market-analysis", description="Market analysis"
                ),
                SpecialistDescriptor(id="fund-research", description="Fund research"),
            ),
            allowed_tool_ids=frozenset(
                {
                    "price_series",
                    "fund_holdings",
                    "fund_reports",
                    "company_news",
                    *CalculationMethod,
                }
            ),
            allowed_skill_names=frozenset(
                {"financial-common", "market-methodology", "fund-disclosure"}
            ),
            allowed_sources=frozenset({"market", "fund", "news"}),
            allowed_queries=frozenset(
                {
                    f"{_FUND_ID} versus {_BENCHMARK_ID}",
                    f"{_FUND_ID} holdings",
                    f"{_FUND_ID} report",
                    f"{_FUND_ID} company news",
                }
            ),
            as_of_date=_AS_OF_DATE,
        )


def financial_app(
    database_url: str,
    fixture: FinancialFixture,
    *,
    query_understanding_actor: QueryUnderstandingActor | None = None,
    coordinator_actor: CoordinatorActor | None = None,
    synthesis_actor: SynthesisActor | None = None,
    checkpointer_factory: CheckpointerFactory = AsyncPostgresSaver,
) -> FastAPI:
    """Assemble the golden fixture with optional request-shape scripting."""
    registry = fixture.registry()
    policy = fixture.policy()

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
            query_understanding_actor=query_understanding_actor
            or fixture.understanding,
            coordinator_actor=coordinator_actor or fixture.coordinator,
            specialist_registry=registry,
            intent_policies={policy.intent: policy},
            synthesis_actor=synthesis_actor or fixture.synthesis.actor(),
        )

    app = persistent_linear_app(
        database_url,
        agent_runtime_factory=factory,
        checkpointer_factory=checkpointer_factory,
    )
    app.state.tenant_manager = _FinancialTenantManager()
    return app


def _post_financial_request(client: TestClient, conversation_id: UUID) -> Any:
    return client.post(
        "/v2/query/stream",
        json={
            "query": f"Analyze {_FUND_ID} versus {_BENCHMARK_ID}.",
            "sessionId": str(conversation_id),
            "clientRequestId": _REQUEST_ID,
            "mode": "linear",
        },
        headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
    )


def financial_checkpoint(
    app: FastAPI, client: TestClient, conversation_id: UUID
) -> Any:
    assert client.portal is not None
    return client.portal.call(
        lambda: app.state.langgraph_v2_checkpointer.aget_tuple(
            thread_checkpoint_config(
                thread_id=thread_id_for(
                    "tenant-a", "subject-a", "agent", str(conversation_id)
                )
            )
        )
    )


def financial_messages(
    app: FastAPI, client: TestClient, conversation_id: UUID
) -> list[Any]:
    assert client.portal is not None
    return client.portal.call(
        lambda: read_conversation_messages(
            app.state.langgraph_v2_checkpointer,
            thread_checkpoint_config(
                thread_id=thread_id_for(
                    "tenant-a", "subject-a", "agent", str(conversation_id)
                )
            ),
            state_adapter=AgentCheckpointStateAdapter(),
        )
    )


def test_financial_golden_path_commits_one_validated_report_per_conversation(
    langgraph_v2_migrated_database_url: str,
) -> None:
    fixture = FinancialFixture()
    app = financial_app(langgraph_v2_migrated_database_url, fixture)
    conversation_ids = (
        UUID("00000000-0000-0000-0000-000000000141"),
        UUID("00000000-0000-0000-0000-000000000142"),
    )

    with TestClient(app) as client:
        responses = [
            _post_financial_request(client, conversation_id)
            for conversation_id in conversation_ids
        ]
        checkpoints = [
            financial_checkpoint(app, client, conversation_id)
            for conversation_id in conversation_ids
        ]
        messages = [
            financial_messages(app, client, conversation_id)
            for conversation_id in conversation_ids
        ]

    assert fixture.tools.price_entered.is_set()
    assert fixture.tools.holdings_entered.is_set()
    assert (
        fixture.tools.provider_calls.count(
            ("market", f"{_FUND_ID} versus {_BENCHMARK_ID}")
        )
        == 2
    )
    assert fixture.tools.provider_calls.count(("fund", f"{_FUND_ID} holdings")) == 2
    assert fixture.tools.provider_calls.count(("fund", f"{_FUND_ID} report")) == 2
    assert fixture.tools.provider_calls.count(("news", f"{_FUND_ID} company news")) == 2
    assert fixture.coordinator.follow_up_context_ids == [
        (_task_id(round=1, dispatch_order=1),),
        (_task_id(round=1, dispatch_order=1),),
    ]
    assert sorted(fixture.specialists.skill_views) == sorted(
        [
            (_MARKET_OBJECTIVE, ("financial-common", "market-methodology")),
            (_FUND_OBJECTIVE, ("financial-common", "fund-disclosure")),
            (_NEWS_OBJECTIVE, ("financial-common", "fund-disclosure")),
        ]
        * 2
    )
    assert fixture.specialists.activated_instruction_views

    for response, checkpoint, conversation_messages in zip(
        responses, checkpoints, messages, strict=True
    ):
        assert response.status_code == 200
        events = parse_sse(response.text)
        done = [event for event in events if event["type"] == "done"]
        citations = [event for event in events if event["type"] == "citations"]
        assert len(done) == 1
        answer = done[0]["data"]["answer"]
        assert done[0]["data"]["metadata"].get("completion_status") == "complete"
        assert (
            done[0]["data"]["metadata"].get("termination_reason") == "evidence_backed"
        )
        assert "Incomplete research:" not in answer
        assert "[[C:" not in answer
        assert (
            "-1.0000% (period return; percent; USD; 2026-09-04 to 2026-09-06; "
            "assumptions: Positive ordered close prices.)"
        ) in answer
        assert (
            "224.4994% (annualized volatility; percent; USD; 2026-09-04 to "
            "2026-09-06; assumptions: Positive ordered close prices.; Sample "
            "standard deviation of simple returns annualized by 252.)"
        ) in answer
        assert (
            "-10.0000% (maximum drawdown; percent; USD; 2026-09-04 to "
            "2026-09-06; assumptions: Positive ordered close prices.; Drawdown "
            "is measured from each prior running peak.)"
        ) in answer
        assert all(f"[[E:{index}]]" in answer for index in range(1, 5))
        assert (
            "".join(event["data"] for event in events if event["type"] == "token")
            == answer
        )
        done_citations = done[0]["data"]["citations"]
        assert [
            (
                citation["index"],
                citation["evidence_id"],
                citation["source"],
                citation["title"],
                citation["url"],
                citation["snippet"],
            )
            for citation in done_citations
        ] == [
            (
                1,
                "price-evidence",
                "market",
                f"{_FUND_ID} versus {_BENCHMARK_ID}",
                "https://fixture.test/price-evidence",
                f"Fixture evidence for {_FUND_ID} versus {_BENCHMARK_ID}.",
            ),
            (
                2,
                "holdings-evidence",
                "fund",
                f"{_FUND_ID} holdings",
                "https://fixture.test/holdings-evidence",
                f"Fixture evidence for {_FUND_ID} holdings.",
            ),
            (
                3,
                "report-evidence",
                "fund",
                f"{_FUND_ID} report",
                "https://fixture.test/report-evidence",
                f"Fixture evidence for {_FUND_ID} report.",
            ),
            (
                4,
                "news-evidence",
                "news",
                f"{_FUND_ID} company news",
                "https://fixture.test/news-evidence",
                f"Fixture evidence for {_FUND_ID} company news.",
            ),
        ]
        assert citations == [{"type": "citations", "data": done_citations}]
        assert [(message.type, message.text) for message in conversation_messages] == [
            ("human", f"Analyze {_FUND_ID} versus {_BENCHMARK_ID}."),
            ("ai", answer),
        ]
        assert checkpoint is not None
        state = checkpoint.checkpoint["channel_values"]
        assert state["incomplete_research"] is None
        rounds = sorted(
            state["coordination_rounds"].values(), key=lambda item: item["revision"]
        )
        assert [(round_["revision"], round_["kind"]) for round_ in rounds] == [
            (1, "dispatch"),
            (2, "dispatch"),
            (3, "finish"),
        ]
        first_tasks = rounds[0]["tasks"]
        assert [task["specialist_id"] for task in first_tasks] == [
            "market-analysis",
            "fund-research",
        ]
        assert rounds[1]["tasks"] == [
            {
                "id": _task_id(round=2, dispatch_order=0),
                "objective": _NEWS_OBJECTIVE,
                "specialist_id": "fund-research",
                "context_task_ids": [_task_id(round=1, dispatch_order=1)],
            }
        ]
        first_batch = state["accepted_batches"][rounds[0]["batch_id"]]
        assert [outcome["task_id"] for outcome in first_batch["outcomes"]] == [
            _task_id(round=1, dispatch_order=0),
            _task_id(round=1, dispatch_order=1),
        ]
        calculations = first_batch["calculations"]
        assert [calculation["id"] for calculation in calculations] == sorted(
            calculation["id"] for calculation in calculations
        )
        assert {calculation["method"] for calculation in calculations} == {
            "period_return",
            "annualized_volatility",
            "maximum_drawdown",
        }
        assert {calculation["task_id"] for calculation in calculations} == {
            _task_id(round=1, dispatch_order=0)
        }
        assert "fixed closes" not in repr(state)
        assert "RAW-PROVIDER-PAYLOAD-MUST-NOT-PERSIST" not in repr(state)
        assert "FULL-" not in repr(state)


def test_failed_holdings_changes_the_next_decision_and_still_finishes_bounded(
    langgraph_v2_migrated_database_url: str,
) -> None:
    fixture = FinancialFixture(alternate_holdings_failure=True)
    app = financial_app(langgraph_v2_migrated_database_url, fixture)
    conversation_id = UUID("00000000-0000-0000-0000-000000000143")

    with TestClient(app) as client:
        response = _post_financial_request(client, conversation_id)
        checkpoint = financial_checkpoint(app, client, conversation_id)

    assert response.status_code == 200
    assert fixture.coordinator.follow_up_context_ids == []
    assert len(fixture.coordinator.inputs) == 2
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    rounds = sorted(
        state["coordination_rounds"].values(), key=lambda item: item["revision"]
    )
    assert [(round_["revision"], round_["kind"]) for round_ in rounds] == [
        (1, "dispatch"),
        (2, "finish"),
    ]
    first_batch = state["accepted_batches"][rounds[0]["batch_id"]]
    assert [outcome["kind"] for outcome in first_batch["outcomes"]] == [
        "succeeded",
        "failed",
    ]
    events = parse_sse(response.text)
    done = [event for event in events if event["type"] == "done"]
    assert len(done) == 1
    assert done[0]["data"]["metadata"]["completion_status"] == "incomplete"
    assert done[0]["data"]["metadata"]["termination_reason"] == "partial_results"
