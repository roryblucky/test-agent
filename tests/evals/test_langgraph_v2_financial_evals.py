"""Deterministic financial Agent Graph regressions run through Pydantic Evals."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal, cast

import pydantic_ai.models as pydantic_models
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from app.langgraph_v2.agent_batch import (
    AcceptedBatch,
    CalculationToolRegistration,
    DispatchBatch,
    EvidenceToolRegistration,
    SpecialistActor,
    SpecialistAttempt,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistTaskInput,
    TaskProposal,
    accept_initial_dispatch,
)
from app.langgraph_v2.agent_coordination import (
    CoordinationRound,
    CoordinatorInput,
    Finish,
)
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    EvidenceInvocationContext,
    ExpectedToolUnavailability,
    FinancialResearchReport,
    PreparedSynthesis,
    RequestEvidenceCatalog,
    SpecialistToolCapture,
    SynthesisCandidateRejected,
    ToolUnavailableReason,
    prepare_synthesis,
    publish_report,
)
from app.langgraph_v2.agent_graph import build_agent_graph
from app.langgraph_v2.agent_scope import (
    AgentIntentPolicy,
    SpecialistDescriptor,
    resolve_research_scope,
)
from app.langgraph_v2.agent_skills import (
    SkillInvocation,
    SkillRegistration,
    SpecialistSkillRegistry,
)
from app.langgraph_v2.calculations import (
    CalculationArtifactInvalid,
    CalculationExecutionContext,
    CalculationExecutor,
    CalculationMethod,
    CalculationRequest,
    PriceObservation,
    TrustedPriceSeries,
)
from app.langgraph_v2.contracts import LiveStreamEvent, V2QueryResponse
from app.langgraph_v2.conversation_context import (
    ConversationExchange,
    conversation_message_id,
)
from app.langgraph_v2.pre_moderation import MockModerationProvider
from app.langgraph_v2.specialist_retry import (
    SpecialistFailureFacts,
    SpecialistInvocationFailure,
    SpecialistModelBoundary,
    specialist_usage_limits,
)
from app.models.workflow import (
    IntentResult,
    QueryUnderstandingClarification,
    QueryUnderstandingClarificationQuestion,
    QueryUnderstandingOutput,
    ResolvedQuery,
)

_AS_OF = date(2026, 9, 6)
_TENANT_ID = "financial-evals-tenant"
_REQUEST_ID = "financial-evals-request"
_QUERY = "Analyze FUND-ALPHA against BENCHMARK-OMEGA."
_INTENT = "financial_research"
_MARKET_OBJECTIVE = "Analyze FUND-ALPHA against BENCHMARK-OMEGA."
_FUND_OBJECTIVE = "Research FUND-ALPHA holdings and disclosures."
_NEWS_OBJECTIVE = "Review FUND-ALPHA company news."


@pytest.fixture(autouse=True)
def disable_real_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pydantic_models, "ALLOW_MODEL_REQUESTS", False)


@dataclass(frozen=True)
class TrajectoryObservation:
    decisions: tuple[str, ...] = ()
    task_shapes: tuple[tuple[str, ...], ...] = ()
    accepted_outcomes: tuple[tuple[str, ...], ...] = ()
    selected_skills: tuple[str, ...] = ()
    selected_tools: tuple[str, ...] = ()
    provider_calls: int = 0
    model_requests: int = 0
    task_invocations: int = 0
    max_task_attempts: int = 0
    max_task_tool_calls: int = 0
    reported_tool_calls: int = 0


@dataclass(frozen=True)
class FinalObservation:
    completion_status: str | None = None
    termination_reason: str | None = None
    disclosure: str | None = None
    citation_ids: tuple[str, ...] = ()
    calculation_rendered: bool = False
    calculation_rendering: str | None = None
    token_matches_answer: bool = False
    canonical_message_matches_answer: bool = False
    done_answer_matches_answer: bool = False
    clarification_has_research_metadata: bool = False
    clarification: QueryUnderstandingClarification | None = None


@dataclass(frozen=True)
class GateObservation:
    scope_rejected: bool = False
    evidence_provenance_rejected: bool = False
    calculation_provenance_rejected: bool = False
    invalid_marker_rejected: bool = False


@dataclass(frozen=True)
class FinancialObservation:
    trajectory: TrajectoryObservation = TrajectoryObservation()
    final: FinalObservation = FinalObservation()
    gates: GateObservation = GateObservation()


@dataclass(frozen=True)
class FinancialCase:
    scenario: Literal[
        "golden",
        "failed_partial",
        "tool_unavailable_partial",
        "authoritative_empty",
        "structural_limit",
        "finish_without_evidence",
        "clarification",
        "scope_permission_gate",
        "evidence_provenance_gate",
        "calculation_provenance_gate",
        "invalid_marker_gate",
    ]


class TrajectoryEvaluator(Evaluator[FinancialCase, FinancialObservation, None]):
    def evaluate(
        self, ctx: EvaluatorContext[FinancialCase, FinancialObservation, None]
    ) -> dict[str, bool]:
        expected = cast(FinancialObservation, ctx.expected_output)
        actual = ctx.output.trajectory
        return {
            "trajectory": actual == expected.trajectory,
            "strict_bounds": (
                actual.model_requests <= 12
                and actual.provider_calls <= 8
                and actual.task_invocations <= 32
                and actual.max_task_attempts <= 3
                and actual.max_task_tool_calls <= 8
                and len(actual.decisions) <= 5
                and all(len(shape) <= 8 for shape in actual.task_shapes)
            ),
        }


class FinalOutputEvaluator(Evaluator[FinancialCase, FinancialObservation, None]):
    def evaluate(
        self, ctx: EvaluatorContext[FinancialCase, FinancialObservation, None]
    ) -> bool:
        return ctx.output.final == cast(FinancialObservation, ctx.expected_output).final


class SingleStepGateEvaluator(Evaluator[FinancialCase, FinancialObservation, None]):
    def evaluate(
        self, ctx: EvaluatorContext[FinancialCase, FinancialObservation, None]
    ) -> bool:
        return ctx.output.gates == cast(FinancialObservation, ctx.expected_output).gates


class _FixtureUnavailable(Exception):
    pass


def _dispatch(*tasks: tuple[str, str]) -> DispatchBatch:
    return DispatchBatch(
        kind="dispatch",
        tasks=tuple(
            TaskProposal(specialist_id=specialist_id, objective=objective)
            for specialist_id, objective in tasks
        ),
    )


def _first_dispatch() -> DispatchBatch:
    return _dispatch(("market", _MARKET_OBJECTIVE), ("fund", _FUND_OBJECTIVE))


@dataclass
class _Understanding:
    scenario: str

    async def understand(
        self, query: str, history: Sequence[ConversationExchange]
    ) -> QueryUnderstandingOutput:
        assert not history
        if self.scenario == "clarification":
            return QueryUnderstandingOutput(
                resolved_query=ResolvedQuery(
                    original_query=query, standalone_query=query
                ),
                intent=IntentResult(intent=_INTENT, confidence=1.0),
                clarification=QueryUnderstandingClarification(
                    scope="query_resolution",
                    questions=[
                        QueryUnderstandingClarificationQuestion(
                            question="Which FUND-ALPHA period should be analyzed?",
                            options=["One month", "One year"],
                        )
                    ],
                ),
            )
        return QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(original_query=query, standalone_query=query),
            intent=IntentResult(intent=_INTENT, confidence=1.0),
        )


@dataclass
class _Coordinator:
    scenario: str
    inputs: list[CoordinatorInput] = field(
        default_factory=lambda: list[CoordinatorInput]()
    )

    async def decide(self, input: CoordinatorInput) -> DispatchBatch | Finish:
        self.inputs.append(input)
        number = len(self.inputs)
        if self.scenario in {"golden", "authoritative_empty"}:
            return (
                _first_dispatch()
                if number == 1
                else _dispatch(("news", _NEWS_OBJECTIVE))
                if number == 2
                else Finish(kind="finish")
            )
        if self.scenario in {"failed_partial", "tool_unavailable_partial"}:
            return _first_dispatch() if number == 1 else Finish(kind="finish")
        if self.scenario == "structural_limit":
            decisions = (
                _first_dispatch(),
                _dispatch(("news", _NEWS_OBJECTIVE)),
                _dispatch(("market", _MARKET_OBJECTIVE)),
                _dispatch(("fund", _FUND_OBJECTIVE)),
            )
            return (
                decisions[number - 1]
                if number <= 4
                else _dispatch(("news", _NEWS_OBJECTIVE))
            )
        assert self.scenario == "finish_without_evidence"
        return Finish(kind="finish")


@dataclass
class _Synthesis:
    prepared: list[PreparedSynthesis] = field(
        default_factory=lambda: list[PreparedSynthesis]()
    )

    async def synthesize(self, prepared: PreparedSynthesis) -> FinancialResearchReport:
        self.prepared.append(prepared)
        evidence_markers = " ".join(
            f"[[E:{index}]]" for index, _ in enumerate(prepared.evidence, start=1)
        )
        calculation_markers = " ".join(
            f"[[C:{index}]]" for index, _ in enumerate(prepared.calculations, start=1)
        )
        return FinancialResearchReport(
            markdown_report=" ".join(
                item
                for item in (
                    "Fixed financial result.",
                    calculation_markers,
                    evidence_markers,
                )
                if item
            )
        )

    async def repair(
        self,
        prepared: PreparedSynthesis,
        *,
        validation_errors: tuple[str, ...],
    ) -> FinancialResearchReport:
        del prepared
        raise AssertionError(
            f"Fixture report should pass first gate: {validation_errors}"
        )


@dataclass
class _Harness:
    scenario: str
    coordinator: _Coordinator = field(init=False)
    provider_calls: list[str] = field(default_factory=lambda: list[str]())
    tool_calls: list[str] = field(default_factory=lambda: list[str]())
    task_calls: list[str] = field(default_factory=lambda: list[str]())
    task_tool_calls: Counter[str] = field(default_factory=Counter[str])
    active_task_ids: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    evidence_counts: Counter[str] = field(default_factory=Counter[str])
    calculated: bool = False

    def __post_init__(self) -> None:
        self.coordinator = _Coordinator(self.scenario)

    @property
    def descriptors(self) -> tuple[SpecialistDescriptor, ...]:
        return (
            SpecialistDescriptor(id="market", description="Market analysis"),
            SpecialistDescriptor(id="fund", description="Fund research"),
            SpecialistDescriptor(id="news", description="Company news"),
        )

    @property
    def policy(self) -> AgentIntentPolicy:
        return AgentIntentPolicy(
            intent=_INTENT,
            description="Fixed financial evaluation scope.",
            specialist_descriptors=self.descriptors,
            allowed_tool_ids=frozenset(
                {"market-reader", "fund-reader", "news-reader", "calculator"}
            ),
            allowed_skill_names=frozenset({"financial-analysis"}),
            allowed_sources=frozenset({"market", "fund", "news"}),
            allowed_queries=frozenset(
                {
                    "FUND-ALPHA versus BENCHMARK-OMEGA",
                    "FUND-ALPHA holdings",
                    "FUND-ALPHA company news",
                }
            ),
            as_of_date=_AS_OF,
        )

    @property
    def calculator(self) -> CalculationExecutor:
        return CalculationExecutor(
            (
                TrustedPriceSeries(
                    ref="fund-alpha-series",
                    instrument_id="FUND-ALPHA",
                    currency="USD",
                    unit="price",
                    observations=(
                        PriceObservation(as_of=date(2026, 9, 4), value=Decimal("100")),
                        PriceObservation(as_of=_AS_OF, value=Decimal("110")),
                    ),
                    evidence_refs=("market-evidence-1",),
                    evidence_hashes=(
                        hashlib.sha256(b"fixed market close prices").hexdigest(),
                    ),
                ),
            )
        )

    def _provider(
        self, tool_id: str, source: str
    ) -> Callable[[str, str], Awaitable[EvidenceEnvelope]]:
        async def provider(requested_source: str, query: str) -> EvidenceEnvelope:
            assert requested_source == source
            if self.scenario == "tool_unavailable_partial" and tool_id == "fund-reader":
                raise _FixtureUnavailable("fixed source outage")
            self.provider_calls.append(tool_id)
            self.evidence_counts[tool_id] += 1
            count = self.evidence_counts[tool_id]
            body = (
                "No matching holdings records were found."
                if self.scenario == "authoritative_empty" and tool_id == "fund-reader"
                else "fixed market close prices"
            )
            return EvidenceEnvelope(
                id=f"{tool_id.removesuffix('-reader')}-evidence-{count}",
                tenant_id=_TENANT_ID,
                request_id=_REQUEST_ID,
                task_id=self.active_task_ids[tool_id],
                source=source,
                source_url=f"https://financial-evals.test/{tool_id}/{count}",
                title=query,
                body=body,
                excerpt=body,
                as_of_date=_AS_OF,
            )

        return provider

    def registry(self) -> SpecialistRegistry:
        skills = SpecialistSkillRegistry(
            registrations=(
                SkillRegistration(
                    name="financial-analysis",
                    version="v1",
                    description="Use fixed financial evidence only.",
                    instructions="Fixed financial analysis instructions.",
                    allowed_tool_ids=frozenset(
                        {"market-reader", "fund-reader", "news-reader", "calculator"}
                    ),
                ),
            ),
            tenant_eligible_names=frozenset({"financial-analysis"}),
        )
        return SpecialistRegistry(
            registrations=tuple(
                SpecialistRegistration(
                    id=role,
                    actor_factory=self._factory(role),
                    allowed_tool_ids=frozenset(tool_ids),
                    allowed_skill_names=frozenset({"financial-analysis"}),
                )
                for role, tool_ids in (
                    ("market", ("market-reader", "calculator")),
                    ("fund", ("fund-reader",)),
                    ("news", ("news-reader",)),
                )
            ),
            tenant_eligible_ids=frozenset({"market", "fund", "news"}),
            tool_registrations=tuple(
                EvidenceToolRegistration(
                    id=tool_id,
                    provider=self._provider(tool_id, source),
                    allowed_sources=frozenset({source}),
                    allowed_queries=frozenset({query}),
                    expected_unavailability=(
                        ExpectedToolUnavailability(
                            exception_type=_FixtureUnavailable,
                            reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
                        ),
                    )
                    if tool_id == "fund-reader"
                    else (),
                )
                for tool_id, source, query in (
                    (
                        "market-reader",
                        "market",
                        "FUND-ALPHA versus BENCHMARK-OMEGA",
                    ),
                    ("fund-reader", "fund", "FUND-ALPHA holdings"),
                    ("news-reader", "news", "FUND-ALPHA company news"),
                )
            ),
            calculation_tool_registrations=(
                CalculationToolRegistration(id="calculator", executor=self.calculator),
            ),
            tenant_eligible_tool_ids=frozenset(
                {"market-reader", "fund-reader", "news-reader", "calculator"}
            ),
            skill_registry=skills,
        )

    def _factory(self, role: str) -> Callable[..., SpecialistActor]:
        def factory(
            tools: tuple[Callable[..., object], ...],
            tool_capture: SpecialistToolCapture,
            skill_invocation: SkillInvocation | None,
        ) -> SpecialistActor:
            return _Specialist(self, role, tools, tool_capture, skill_invocation)

        return factory


@dataclass
class _Specialist:
    harness: _Harness
    role: str
    tools: tuple[Callable[..., object], ...]
    capture: SpecialistToolCapture
    skills: SkillInvocation | None
    task_id: str | None = None

    async def run(
        self,
        input: SpecialistTaskInput,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> SpecialistAttempt:
        assert usage is not None
        assert usage_limits == specialist_usage_limits()
        self.harness.task_calls.append(input.task_id)
        self.task_id = input.task_id
        if self.harness.scenario == "failed_partial" and self.role == "fund":
            usage.incr(RunUsage(requests=1))
            raise SpecialistInvocationFailure(
                ModelHTTPError(429, "fixed fund failure"),
                facts=SpecialistFailureFacts(
                    boundary=SpecialistModelBoundary.AZURE_OPENAI,
                    at_model_request_boundary=True,
                    terminal_output_tool_rejected=False,
                    count_limit_exhausted=False,
                    unreturned_model_requests=0,
                    usage_limits=specialist_usage_limits(),
                ),
                messages=(),
            )
        assert self.skills is not None
        self.skills.activate("financial-analysis")
        tool_calls_before = len(self.harness.tool_calls)
        tool_id, source, query = {
            "market": (
                "market-reader",
                "market",
                "FUND-ALPHA versus BENCHMARK-OMEGA",
            ),
            "fund": ("fund-reader", "fund", "FUND-ALPHA holdings"),
            "news": ("news-reader", "news", "FUND-ALPHA company news"),
        }[self.role]
        self.harness.active_task_ids[tool_id] = input.task_id
        await self._call(tool_id, source, query)
        if self.role == "market" and not self.harness.calculated:
            self.harness.calculated = True
            await self._call(
                "calculator", CalculationMethod.PERIOD_RETURN, "v1", "fund-alpha-series"
            )
        usage.incr(
            RunUsage(
                requests=1,
                tool_calls=len(self.harness.tool_calls) - tool_calls_before,
            )
        )
        return SpecialistAttempt(
            finding=SpecialistFindingDraft(
                summary=f"{self.role} fixture result",
                evidence_ids=tuple(item.id for item in self.capture.evidence),
            ),
            evidence=tuple(self.capture.evidence),
            unavailability=tuple(self.capture.unavailability),
            calculations=tuple(self.capture.calculations),
            skill_pins=self.skills.pins,
        )

    async def _call(self, tool_id: str, *args: object) -> None:
        tool = next(
            tool for tool in self.tools if getattr(tool, "__name__", None) == tool_id
        )
        self.harness.tool_calls.append(tool_id)
        assert self.task_id is not None
        self.harness.task_tool_calls[self.task_id] += 1
        await cast(Any, tool)(
            RunContext(
                deps=None,
                model=TestModel(),
                usage=RunUsage(),
                tool_call_id=f"call-{len(self.harness.tool_calls)}",
                tool_name=tool_id,
            ),
            *args,
        )


def _events(value: object) -> list[LiveStreamEvent]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        if isinstance(mapping.get("type"), str):
            return [LiveStreamEvent.model_validate(mapping)]
        if "event" in mapping:
            return _events(mapping["event"])
        return _events(next(iter(mapping.values()))) if len(mapping) == 1 else []
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)
        return [event for item in items for event in _events(item)]
    return []


async def _graph_observation(harness: _Harness) -> FinancialObservation:
    graph: Any = build_agent_graph(
        InMemorySaver(),
        query_understanding_actor=_Understanding(harness.scenario),
        coordinator_actor=harness.coordinator,
        specialist_registry=harness.registry(),
        intent_policies={_INTENT: harness.policy},
        moderation_provider=MockModerationProvider(),
        tenant_id=_TENANT_ID,
        evidence_catalog=RequestEvidenceCatalog(),
        synthesis_actor=_Synthesis(),
    )
    config = {"configurable": {"thread_id": f"financial-evals-{harness.scenario}"}}
    events: list[LiveStreamEvent] = []
    async for raw_part in graph.astream(
        {
            "query": _QUERY,
            "conversation_id": "financial-evals",
            "request_id": _REQUEST_ID,
        },
        config=config,
        stream_mode=["updates", "custom"],
    ):
        part = cast(object, raw_part)
        if isinstance(part, tuple):
            items = cast(tuple[object, ...], part)
            if len(items) == 2 and items[0] == "custom":
                events.extend(_events(items[1]))
    state = cast(Mapping[str, object], (await graph.aget_state(config)).values)
    rounds = tuple(
        sorted(
            (
                CoordinationRound.model_validate(value)
                for value in cast(
                    Mapping[str, object], state["coordination_rounds"]
                ).values()
            ),
            key=lambda round_: round_.revision,
        )
    )
    raw_batches = cast(Mapping[str, object], state["accepted_batches"])
    batches = tuple(
        AcceptedBatch.model_validate(raw_batches[round_.batch_id])
        for round_ in rounds
        if round_.kind == "dispatch" and round_.batch_id is not None
    )
    response = V2QueryResponse.model_validate(state["final_response"])
    answer = response.answer
    assert isinstance(answer, str)
    done = next(event for event in events if event.type == "done")
    done_data = cast(Mapping[str, object], done.data)
    message = cast(Sequence[object], state["conversation_messages"])[-1]
    message_content: object = getattr(message, "content")
    message_id: object = getattr(message, "id")
    assert isinstance(message_content, str)
    assert isinstance(message_id, str)
    attempts = Counter(harness.task_calls)
    disclosure = (
        answer.split("\n\n", maxsplit=1)[0]
        if answer.startswith("Incomplete research:")
        else None
    )
    calculation = re.search(r"-?\d+\.\d{4}% \(period return; [^)]*\)", answer)
    return FinancialObservation(
        trajectory=TrajectoryObservation(
            decisions=tuple(round_.kind for round_ in rounds),
            task_shapes=tuple(
                tuple(task.specialist_id for task in round_.tasks)
                for round_ in rounds
                if round_.kind == "dispatch"
            ),
            accepted_outcomes=tuple(
                tuple(outcome.kind for outcome in batch.outcomes) for batch in batches
            ),
            selected_skills=tuple(
                sorted(
                    {
                        pin.name
                        for batch in batches
                        for task_pins in batch.skill_pins
                        for pin in task_pins.pins
                    }
                )
            ),
            selected_tools=tuple(sorted(set(harness.tool_calls))),
            provider_calls=len(harness.provider_calls),
            model_requests=sum(batch.usage.model_requests for batch in batches),
            task_invocations=len(harness.task_calls),
            max_task_attempts=max(attempts.values(), default=0),
            max_task_tool_calls=max(harness.task_tool_calls.values(), default=0),
            reported_tool_calls=sum(
                batch.usage.completed_tool_calls for batch in batches
            ),
        ),
        final=FinalObservation(
            completion_status=cast(
                str | None, response.metadata.get("completion_status")
            ),
            termination_reason=cast(
                str | None, response.metadata.get("termination_reason")
            ),
            disclosure=disclosure,
            citation_ids=tuple(citation.evidence_id for citation in response.citations),
            calculation_rendered=calculation is not None and "[[C:" not in answer,
            calculation_rendering=(
                calculation.group() if calculation is not None else None
            ),
            token_matches_answer="".join(
                cast(str, event.data) for event in events if event.type == "token"
            )
            == answer,
            canonical_message_matches_answer=(
                message_content == answer
                and message_id == conversation_message_id(_REQUEST_ID, "assistant")
            ),
            done_answer_matches_answer=done_data.get("answer") == answer,
            clarification_has_research_metadata=(
                "completion_status" in response.metadata
                or "termination_reason" in response.metadata
            ),
            clarification=response.clarification,
        ),
    )


def _gate_observation(case: FinancialCase) -> FinancialObservation:
    if case.scenario == "scope_permission_gate":
        scope = resolve_research_scope(
            IntentResult(intent=_INTENT, confidence=1.0),
            {
                _INTENT: AgentIntentPolicy(
                    intent=_INTENT,
                    description="Market-only fixed scope.",
                    specialist_descriptors=(
                        SpecialistDescriptor(
                            id="market", description="Market analysis"
                        ),
                    ),
                )
            },
        )
        with pytest.raises(ValueError, match="Specialist is not eligible"):
            accept_initial_dispatch(
                _dispatch(("fund", _FUND_OBJECTIVE)),
                request_id=_REQUEST_ID,
                registry=_Harness("golden").registry(),
                scope_descriptors=scope.specialist_descriptors,
            )
        return FinancialObservation(gates=GateObservation(scope_rejected=True))
    catalog = RequestEvidenceCatalog()
    evidence = EvidenceEnvelope(
        id="market-evidence-1",
        tenant_id=_TENANT_ID,
        request_id=_REQUEST_ID,
        task_id="task-market",
        source="market",
        source_url="https://financial-evals.test/market",
        title="Market fixture",
        body="fixed market close prices",
        excerpt="fixed market close prices",
        as_of_date=_AS_OF,
    )
    if case.scenario == "evidence_provenance_gate":
        foreign = evidence.model_copy(update={"tenant_id": "other-tenant"})
        with pytest.raises(ValueError, match="Evidence provenance is not eligible"):
            catalog.accept_referenced(
                (foreign,),
                finding_evidence_ids=(foreign.id,),
                context=EvidenceInvocationContext(
                    tenant_id=_TENANT_ID,
                    request_id=_REQUEST_ID,
                    task_id="task-market",
                ),
            )
        return FinancialObservation(
            gates=GateObservation(evidence_provenance_rejected=True)
        )
    catalog.accept_referenced(
        (evidence,),
        finding_evidence_ids=(evidence.id,),
        context=EvidenceInvocationContext(
            tenant_id=_TENANT_ID, request_id=_REQUEST_ID, task_id="task-market"
        ),
    )
    if case.scenario == "calculation_provenance_gate":
        foreign = _Harness("golden").calculator.execute(
            CalculationRequest(
                method=CalculationMethod.PERIOD_RETURN,
                version="v1",
                series_ref="fund-alpha-series",
            ),
            context=CalculationExecutionContext(
                tenant_id="other-tenant",
                request_id=_REQUEST_ID,
                task_id="task-market",
                attempt=1,
                tool_id="calculator",
            ),
        )
        with pytest.raises(
            CalculationArtifactInvalid,
            match="Calculation Artifact provenance is invalid",
        ):
            prepare_synthesis(
                standalone_query=_QUERY,
                intent=_INTENT,
                accepted_evidence_ids=(evidence.id,),
                catalog=catalog,
                tenant_id=_TENANT_ID,
                request_id=_REQUEST_ID,
                accepted_calculations=(foreign,),
            )
        return FinancialObservation(
            gates=GateObservation(calculation_provenance_rejected=True)
        )
    assert case.scenario == "invalid_marker_gate"
    prepared = prepare_synthesis(
        standalone_query=_QUERY,
        intent=_INTENT,
        accepted_evidence_ids=(evidence.id,),
        catalog=catalog,
        tenant_id=_TENANT_ID,
        request_id=_REQUEST_ID,
    )
    with pytest.raises(
        SynthesisCandidateRejected, match="Evidence marker is not eligible"
    ):
        publish_report(
            FinancialResearchReport(markdown_report="Forged support [[E:2]]"), prepared
        )
    return FinancialObservation(gates=GateObservation(invalid_marker_rejected=True))


async def _run_financial_case(case: FinancialCase) -> FinancialObservation:
    if case.scenario.endswith("_gate"):
        return _gate_observation(case)
    return await _graph_observation(_Harness(case.scenario))


_PERIOD_RETURN = (
    "10.0000% (period return; percent; USD; 2026-09-04 to 2026-09-06; "
    "assumptions: Positive ordered close prices.)"
)

_CANONICAL_FINAL = FinalObservation(
    completion_status="complete",
    termination_reason="evidence_backed",
    citation_ids=("market-evidence-1", "fund-evidence-1", "news-evidence-1"),
    calculation_rendered=True,
    calculation_rendering=_PERIOD_RETURN,
    token_matches_answer=True,
    canonical_message_matches_answer=True,
    done_answer_matches_answer=True,
    clarification_has_research_metadata=True,
)

_CASES = (
    Case(
        name="canonical-golden-path",
        inputs=FinancialCase(scenario="golden"),
        expected_output=FinancialObservation(
            trajectory=TrajectoryObservation(
                decisions=("dispatch", "dispatch", "finish"),
                task_shapes=(("market", "fund"), ("news",)),
                accepted_outcomes=(("succeeded", "succeeded"), ("succeeded",)),
                selected_skills=("financial-analysis",),
                selected_tools=(
                    "calculator",
                    "fund-reader",
                    "market-reader",
                    "news-reader",
                ),
                provider_calls=3,
                model_requests=3,
                task_invocations=3,
                max_task_attempts=1,
                max_task_tool_calls=2,
                reported_tool_calls=4,
            ),
            final=_CANONICAL_FINAL,
        ),
    ),
    Case(
        name="failed-holdings-is-partial",
        inputs=FinancialCase(scenario="failed_partial"),
        expected_output=FinancialObservation(
            trajectory=TrajectoryObservation(
                decisions=("dispatch", "finish"),
                task_shapes=(("market", "fund"),),
                accepted_outcomes=(("succeeded", "failed"),),
                selected_skills=("financial-analysis",),
                selected_tools=("calculator", "market-reader"),
                provider_calls=1,
                model_requests=4,
                task_invocations=4,
                max_task_attempts=3,
                max_task_tool_calls=2,
                reported_tool_calls=2,
            ),
            final=FinalObservation(
                completion_status="incomplete",
                termination_reason="partial_results",
                disclosure=(
                    "Incomplete research: one requested task could not complete.\n"
                    "- Research FUND\\-ALPHA holdings and disclosures\\."
                ),
                citation_ids=("market-evidence-1",),
                calculation_rendered=True,
                calculation_rendering=_PERIOD_RETURN,
                token_matches_answer=True,
                canonical_message_matches_answer=True,
                done_answer_matches_answer=True,
                clarification_has_research_metadata=True,
            ),
        ),
    ),
    Case(
        name="tool-unavailable-is-partial",
        inputs=FinancialCase(scenario="tool_unavailable_partial"),
        expected_output=FinancialObservation(
            trajectory=TrajectoryObservation(
                decisions=("dispatch", "finish"),
                task_shapes=(("market", "fund"),),
                accepted_outcomes=(("succeeded", "succeeded"),),
                selected_skills=("financial-analysis",),
                selected_tools=("calculator", "fund-reader", "market-reader"),
                provider_calls=1,
                model_requests=2,
                task_invocations=2,
                max_task_attempts=1,
                max_task_tool_calls=2,
                reported_tool_calls=3,
            ),
            final=FinalObservation(
                completion_status="incomplete",
                termination_reason="partial_results",
                disclosure=(
                    "Incomplete research: requested data was unavailable:\n"
                    "- FUND\\-ALPHA holdings"
                ),
                citation_ids=("market-evidence-1",),
                calculation_rendered=True,
                calculation_rendering=_PERIOD_RETURN,
                token_matches_answer=True,
                canonical_message_matches_answer=True,
                done_answer_matches_answer=True,
                clarification_has_research_metadata=True,
            ),
        ),
    ),
    Case(
        name="authoritative-empty-evidence-is-complete",
        inputs=FinancialCase(scenario="authoritative_empty"),
        expected_output=FinancialObservation(
            trajectory=TrajectoryObservation(
                decisions=("dispatch", "dispatch", "finish"),
                task_shapes=(("market", "fund"), ("news",)),
                accepted_outcomes=(("succeeded", "succeeded"), ("succeeded",)),
                selected_skills=("financial-analysis",),
                selected_tools=(
                    "calculator",
                    "fund-reader",
                    "market-reader",
                    "news-reader",
                ),
                provider_calls=3,
                model_requests=3,
                task_invocations=3,
                max_task_attempts=1,
                max_task_tool_calls=2,
                reported_tool_calls=4,
            ),
            final=_CANONICAL_FINAL,
        ),
    ),
    Case(
        name="coordination-limit-is-incomplete",
        inputs=FinancialCase(scenario="structural_limit"),
        expected_output=FinancialObservation(
            trajectory=TrajectoryObservation(
                decisions=("dispatch", "dispatch", "dispatch", "dispatch"),
                task_shapes=(("market", "fund"), ("news",), ("market",), ("fund",)),
                accepted_outcomes=(
                    ("succeeded", "succeeded"),
                    ("succeeded",),
                    ("succeeded",),
                    ("succeeded",),
                ),
                selected_skills=("financial-analysis",),
                selected_tools=(
                    "calculator",
                    "fund-reader",
                    "market-reader",
                    "news-reader",
                ),
                provider_calls=5,
                model_requests=5,
                task_invocations=5,
                max_task_attempts=1,
                max_task_tool_calls=2,
                reported_tool_calls=6,
            ),
            final=FinalObservation(
                completion_status="incomplete",
                termination_reason="execution_limit",
                disclosure="Incomplete research: the Coordination limit ended further work.",
                citation_ids=(
                    "market-evidence-1",
                    "fund-evidence-1",
                    "news-evidence-1",
                    "market-evidence-2",
                    "fund-evidence-2",
                ),
                calculation_rendered=True,
                calculation_rendering=_PERIOD_RETURN,
                token_matches_answer=True,
                canonical_message_matches_answer=True,
                done_answer_matches_answer=True,
                clarification_has_research_metadata=True,
            ),
        ),
    ),
    Case(
        name="finish-first-is-insufficient-evidence",
        inputs=FinancialCase(scenario="finish_without_evidence"),
        expected_output=FinancialObservation(
            trajectory=TrajectoryObservation(decisions=("finish",)),
            final=FinalObservation(
                completion_status="incomplete",
                termination_reason="insufficient_evidence",
                disclosure="Incomplete research: no eligible Evidence was available.",
                token_matches_answer=True,
                canonical_message_matches_answer=True,
                done_answer_matches_answer=True,
                clarification_has_research_metadata=True,
            ),
        ),
    ),
    Case(
        name="clarification-halts-before-research",
        inputs=FinancialCase(scenario="clarification"),
        expected_output=FinancialObservation(
            final=FinalObservation(
                token_matches_answer=True,
                canonical_message_matches_answer=True,
                done_answer_matches_answer=True,
                clarification_has_research_metadata=False,
                clarification=QueryUnderstandingClarification(
                    scope="query_resolution",
                    questions=[
                        QueryUnderstandingClarificationQuestion(
                            question="Which FUND-ALPHA period should be analyzed?",
                            options=["One month", "One year"],
                        )
                    ],
                ),
            )
        ),
    ),
    Case(
        name="scope-permission-fails-closed",
        inputs=FinancialCase(scenario="scope_permission_gate"),
        expected_output=FinancialObservation(
            gates=GateObservation(scope_rejected=True)
        ),
    ),
    Case(
        name="evidence-provenance-fails-closed",
        inputs=FinancialCase(scenario="evidence_provenance_gate"),
        expected_output=FinancialObservation(
            gates=GateObservation(evidence_provenance_rejected=True)
        ),
    ),
    Case(
        name="calculation-provenance-fails-closed",
        inputs=FinancialCase(scenario="calculation_provenance_gate"),
        expected_output=FinancialObservation(
            gates=GateObservation(calculation_provenance_rejected=True)
        ),
    ),
    Case(
        name="invalid-support-marker-fails-closed",
        inputs=FinancialCase(scenario="invalid_marker_gate"),
        expected_output=FinancialObservation(
            gates=GateObservation(invalid_marker_rejected=True)
        ),
    ),
)


@pytest.mark.asyncio
async def test_financial_regression_dataset() -> None:
    report = await Dataset(
        name="financial-agent-regressions",
        cases=_CASES,
        evaluators=(
            TrajectoryEvaluator(),
            FinalOutputEvaluator(),
            SingleStepGateEvaluator(),
        ),
    ).evaluate(_run_financial_case, max_concurrency=1, progress=False)

    assert report.failures == []
    failures: list[str] = []
    for case in report.cases:
        assert case.evaluator_failures == []
        failures.extend(
            f"{case.name}: {name}"
            for name, result in case.assertions.items()
            if result.value is not True
        )
    assert not failures, failures
