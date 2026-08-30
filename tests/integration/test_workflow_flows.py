"""Integration tests for end-to-end workflow execution paths."""

from __future__ import annotations

import unittest.mock
from collections.abc import Callable
from typing import Any, Self, cast
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.usage import RunUsage

from app.api.schemas import QueryResponse
from app.config.models import (
    AgentConfig,
    FlowConfig,
    FlowStep,
    FlowStepType,
    LLMConfig,
    TenantConfig,
)
from app.models.domain import (
    Document,
    GroundednessResult,
    ModerationResult,
    RefinedQuestion,
)
from app.models.workflow import (
    AggregatedEvidenceBundle,
    IntentResult,
    NormalizedToolResultItem,
    PlannerOutput,
    QueryUnderstandingOutput,
    ResolvedQuery,
    ToolCallRecord,
    ToolObservation,
    ToolResultRecord,
)
from app.providers.base import (
    BaseGroundednessProvider,
    BaseModerationProvider,
    BaseRankerProvider,
    BaseRetrieverProvider,
)
from app.services.citation_extractor import QuoteExtractionItem, QuoteExtractionResult
from app.services.flow_engine import FlowEngine
from app.services.handlers.agent import AgentHandler
from app.services.handlers.aggregation import AggregationHandler
from app.services.handlers.analysis import AnalysisHandler
from app.services.handlers.groundedness import GroundednessHandler
from app.services.handlers.llm import LLMHandler
from app.services.handlers.moderation import ModerationHandler
from app.services.handlers.ranking import RankingHandler
from app.services.handlers.retriever import RetrieverHandler
from app.services.tenant_manager import TenantProviders


class FakeAgentStream:
    """Async stream helper that mimics the tiny slice of pydantic-ai we use."""

    def __init__(
        self,
        output: Any,
        chunks: list[str] | None = None,
        on_enter: Callable[[Any], None] | None = None,
        deps: Any | None = None,
        messages: list[Any] | None = None,
    ) -> None:
        self._output = output
        self._chunks = chunks or []
        self._on_enter = on_enter
        self._deps = deps
        self._messages = messages or []

    async def __aenter__(self) -> Self:
        if self._on_enter:
            self._on_enter(self._deps)
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def stream_output(self, debounce_by: float = 0.01):
        for chunk in self._chunks:
            yield chunk

    async def stream_text(self):
        for chunk in self._chunks:
            yield chunk

    async def get_output(self) -> Any:
        return self._output

    def usage(self) -> RunUsage:
        return RunUsage()

    def new_messages(self) -> list[Any]:
        return self._messages


class FakeAgent:
    """Minimal agent double used by the integration tests."""

    def __init__(
        self,
        output: Any,
        chunks: list[str] | None = None,
        on_enter: Callable[[Any], None] | None = None,
    ) -> None:
        self.output = output
        self.chunks = chunks or []
        self.on_enter = on_enter
        self.last_prompt: str | None = None
        self.last_deps: Any | None = None
        self.run_count = 0

    def run_stream(
        self,
        prompt: str,
        *,
        deps: Any | None = None,
        **kwargs: Any,
    ) -> FakeAgentStream:
        self.run_count += 1
        self.last_prompt = prompt
        self.last_deps = deps
        new_messages = [ModelRequest(parts=[UserPromptPart(prompt)])]
        return FakeAgentStream(
            self.output,
            chunks=self.chunks,
            on_enter=self.on_enter,
            deps=deps,
            messages=new_messages,
        )


class RecordingModerationProvider(BaseModerationProvider):
    """Moderation stub that records the texts it checks."""

    def __init__(self, flagged_texts: set[str] | None = None) -> None:
        self.flagged_texts = flagged_texts or set()
        self.checked_texts: list[str] = []

    async def check(self, text: str) -> ModerationResult:
        self.checked_texts.append(text)
        is_flagged = text in self.flagged_texts
        return ModerationResult(
            is_flagged=is_flagged,
            categories={"policy": 1.0} if is_flagged else {},
            reason="flagged" if is_flagged else None,
        )


class RecordingRetrieverProvider(BaseRetrieverProvider):
    """Retriever stub with a simple top-k config."""

    def __init__(self, docs: list[Document], top_k: int = 2) -> None:
        self.top_k = top_k
        self.docs = docs
        self.calls: list[tuple[str, int, str | None]] = []

    async def retrieve(
        self, query: str, top_k: int = 10, filter_expr: str | None = None
    ) -> list[Document]:
        self.calls.append((query, top_k, filter_expr))
        return self.docs[:top_k]

    async def retrieve_configured(
        self, query: str, filter_expr: str | None = None
    ) -> list[Document]:
        return await self.retrieve(query, self.top_k, filter_expr)


class RecordingRankerProvider(BaseRankerProvider):
    """Ranker stub that reorders documents by score."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def rank(
        self, query: str, documents: list[Document], top_n: int = 5
    ) -> list[Document]:
        self.calls.append((query, [doc.id for doc in documents]))
        return sorted(
            documents,
            key=lambda doc: doc.score or 0.0,
            reverse=True,
        )[:top_n]


class RecordingGroundednessProvider(BaseGroundednessProvider):
    """Groundedness stub that records the final answer/context pair."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def check(self, answer: str, context: list[Document]) -> GroundednessResult:
        self.calls.append((answer, [doc.id for doc in context]))
        return GroundednessResult(
            is_grounded=True,
            score=0.98,
            details="grounded",
        )


def _tool_result() -> ToolResultRecord:
    return ToolResultRecord(
        tool_call_id="search_documents:1",
        tool_name="search_documents",
        source="integration-test",
        normalized_items=[
            NormalizedToolResultItem(
                item_id="doc-1",
                title="Evidence ev-1",
                content="Evidence one for wealth analysis.",
                score=0.9,
            ),
            NormalizedToolResultItem(
                item_id="doc-2",
                title="Evidence ev-2",
                content="Evidence two for wealth analysis.",
                score=0.8,
            ),
        ],
    )


def _llm_handler(registry: Any | None = None) -> LLMHandler:
    return LLMHandler(registry or MagicMock())


def _supervisor_handler(
    *,
    tenant_id: str,
    agent_config: AgentConfig,
    providers: TenantProviders,
    agent: FakeAgent,
) -> AgentHandler:
    handler = AgentHandler(
        registry=MagicMock(),
        providers=providers,
        cfg=TenantConfig(
            kms_app_name=tenant_id,
            application_id=tenant_id,
            ad_groups=[],
            llm_config=LLMConfig(models={}),
            flow_config=FlowConfig(),
        ),
        skill_registry=None,
    )
    handler.set_agent_override("supervisor", agent_config, cast(Agent[Any, Any], agent))
    return handler


def _planner_handler(
    *,
    tenant_id: str,
    agent_config: AgentConfig,
    providers: TenantProviders,
    agent: FakeAgent,
) -> AgentHandler:
    handler = AgentHandler(
        registry=MagicMock(),
        providers=providers,
        cfg=TenantConfig(
            kms_app_name=tenant_id,
            application_id=tenant_id,
            ad_groups=[],
            llm_config=LLMConfig(models={}),
            flow_config=FlowConfig(),
        ),
        skill_registry=None,
    )
    handler.set_agent_override("planner", agent_config, cast(Agent[Any, Any], agent))
    return handler


@pytest.mark.asyncio
async def test_existing_rag_workflow_end_to_end(
    mock_emitter: unittest.mock.AsyncMock,
) -> None:
    """A classic RAG tenant still runs moderation -> refine -> retrieve -> rank -> answer."""
    retrieved_docs = [
        Document(
            id="doc-1",
            content="Document one about alpha.",
            score=0.4,
        ),
        Document(
            id="doc-2",
            content="Document two with the strongest match.",
            score=0.9,
        ),
    ]
    retriever = RecordingRetrieverProvider(retrieved_docs, top_k=2)
    ranker = RecordingRankerProvider()
    moderation = RecordingModerationProvider()
    groundedness = RecordingGroundednessProvider()

    refine_agent = FakeAgent(
        output=RefinedQuestion(
            refined_query="What is alpha in this tenant?",
            keywords=["alpha", "tenant"],
        ),
    )
    answer_agent = FakeAgent(
        output="Alpha is the strongest match.",
        chunks=["Alpha is ", "the strongest match."],
    )

    llm_handler = _llm_handler()
    llm_handler.set_agent_override(
        "refine_question", "fast", cast(Agent[Any, Any], refine_agent)
    )
    llm_handler.set_agent_override("answer", "pro", cast(Agent[Any, Any], answer_agent))

    steps = [
        FlowStep(type=FlowStepType.MODERATION, mode="pre"),
        FlowStep(type=FlowStepType.LLM, mode="refine_question", model="fast"),
        FlowStep(type=FlowStepType.RETRIEVER),
        FlowStep(type=FlowStepType.RANKING),
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
        FlowStep(type=FlowStepType.GROUNDEDNESS),
        FlowStep(type=FlowStepType.MODERATION, mode="post"),
        FlowStep(type=FlowStepType.ANALYSIS),
    ]
    tenant = TenantConfig(
        kms_app_name="rag-tenant",
        application_id="rag-tenant",
        ad_groups=[],
        llm_config=LLMConfig(models={}),
        flow_config=FlowConfig(steps=steps),
    )

    handlers = {
        FlowStepType.MODERATION: ModerationHandler(moderation),
        FlowStepType.LLM: llm_handler,
        FlowStepType.RETRIEVER: RetrieverHandler(retriever),
        FlowStepType.RANKING: RankingHandler(ranker),
        FlowStepType.GROUNDEDNESS: GroundednessHandler(groundedness),
        FlowStepType.ANALYSIS: AnalysisHandler(),
    }

    ctx = await FlowEngine(tenant, handlers).execute(
        "What is alpha?",
        emitter=mock_emitter,
        session_id="session-rag",
    )

    response = QueryResponse.from_flow_context(ctx)

    assert ctx.metadata["steps_executed"] == [
        "moderation:pre",
        "llm:refine_question",
        "retriever",
        "ranking",
        "llm:answer",
        "groundedness",
        "moderation:post",
        "analysis",
    ]
    assert ctx.refined_query == "What is alpha in this tenant?"
    assert [doc.id for doc in ctx.documents] == ["doc-1", "doc-2"]
    assert [doc.id for doc in ctx.ranked_documents] == ["doc-2", "doc-1"]
    assert ctx.llm_response == "Alpha is the strongest match."
    assert ctx.groundedness_result == GroundednessResult(
        is_grounded=True,
        score=0.98,
        details="grounded",
    )
    assert response.answer == "Alpha is the strongest match."
    assert answer_agent.last_prompt is not None
    assert "Document two with the strongest match." in answer_agent.last_prompt
    assert not hasattr(answer_agent.last_deps, "reference_data")
    assert ctx.new_messages
    persisted_request = ctx.new_messages[0]
    assert isinstance(persisted_request, ModelRequest)
    persisted_part = persisted_request.parts[0]
    assert isinstance(persisted_part, UserPromptPart)
    assert persisted_part.content == "What is alpha?"
    assert "Document two with the strongest match." not in persisted_part.content
    assert moderation.checked_texts == [
        "What is alpha?",
        "Alpha is the strongest match.",
    ]
    assert retriever.calls == [("What is alpha in this tenant?", 2, None)]
    assert ranker.calls == [
        (
            "What is alpha in this tenant?",
            ["doc-1", "doc-2"],
        )
    ]
    assert groundedness.calls == [("Alpha is the strongest match.", ["doc-2", "doc-1"])]
    assert ctx.metadata["analysis"]["documents_retrieved"] == 2
    assert ctx.metadata["analysis"]["streaming_policy"] == "token"
    mock_emitter.emit_token.assert_any_await("Alpha is ")
    mock_emitter.emit_token.assert_any_await("the strongest match.")


@pytest.mark.asyncio
async def test_planner_aggregation_and_citation_workflow_end_to_end(
    mock_emitter: unittest.mock.AsyncMock,
) -> None:
    """Planner + aggregation flows stream tokens, generate citations, and emit them in order."""
    tenant_id = "wealth-tenant"

    def _planner_setup(deps: Any) -> None:
        flow_ctx = deps.flow_context
        assert flow_ctx is not None
        flow_ctx.metadata["tenant_id"] = tenant_id
        flow_ctx.active_skills.append("wealth-skill")
        deps.activated_skill_names.append("wealth-skill")
        flow_ctx.tool_results.append(_tool_result())
        flow_ctx.tool_calls.append(
            ToolCallRecord(
                tool_call_id="search_documents:1",
                tool_name="search_documents",
                input_payload={"query": "wealth query"},
                status="success",
                result_count=2,
            )
        )
        flow_ctx.tool_observations.append(
            ToolObservation(
                tool_name="search_documents",
                status="success",
                task_status_hint="completed",
                result_count=2,
            )
        )

    planner_agent = FakeAgent(
        output=PlannerOutput(
            can_continue_to_aggregation=True,
            reason="Evidence is sufficient.",
            completed_tasks=["search_documents"],
        ),
        on_enter=_planner_setup,
    )
    refine_agent = FakeAgent(
        output=RefinedQuestion(
            refined_query="What is the market view for this asset?",
            keywords=["market", "asset"],
        ),
    )
    intent_agent = FakeAgent(
        output=IntentResult(
            intent="market_outlook",
            confidence=0.94,
        ),
    )
    answer_agent = FakeAgent(
        output="Draft wealth answer [1].",
        chunks=["Draft wealth ", "answer [1]."],
    )

    quote_agent = FakeAgent(
        output=QuoteExtractionResult(
            extractions=[
                QuoteExtractionItem(
                    citation_index=1,
                    quoted_passages=["Evidence one for wealth analysis."],
                )
            ]
        )
    )

    mock_registry = MagicMock()
    mock_registry.create_agent.return_value = quote_agent

    llm_handler = _llm_handler(mock_registry)
    llm_handler.set_agent_override(
        "refine_question", "fast", cast(Agent[Any, Any], refine_agent)
    )
    llm_handler.set_agent_override(
        "intent", "intent", cast(Agent[Any, Any], intent_agent)
    )
    llm_handler.set_agent_override("answer", "pro", cast(Agent[Any, Any], answer_agent))

    agent_config = AgentConfig(
        llm_type="pro",
        built_in_tools=["search_documents"],
    )
    steps = [
        FlowStep(type=FlowStepType.MODERATION, mode="pre"),
        FlowStep(type=FlowStepType.LLM, mode="refine_question", model="fast"),
        FlowStep(type=FlowStepType.LLM, mode="intent", model="intent"),
        FlowStep(type=FlowStepType.AGENT, mode="planner", agent_config=agent_config),
        FlowStep(type=FlowStepType.AGGREGATION),
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
        FlowStep(type=FlowStepType.MODERATION, mode="post"),
        FlowStep(type=FlowStepType.ANALYSIS),
    ]
    tenant = TenantConfig(
        kms_app_name=tenant_id,
        application_id=tenant_id,
        ad_groups=[],
        llm_config=LLMConfig(models={}),
        flow_config=FlowConfig(steps=steps),
    )

    planner_handler = _planner_handler(
        tenant_id=tenant_id,
        agent_config=agent_config,
        providers=TenantProviders(),
        agent=planner_agent,
    )
    handlers = {
        FlowStepType.MODERATION: ModerationHandler(RecordingModerationProvider()),
        FlowStepType.LLM: llm_handler,
        FlowStepType.AGENT: planner_handler,
        FlowStepType.AGGREGATION: AggregationHandler(),
        FlowStepType.ANALYSIS: AnalysisHandler(),
    }

    ctx = await FlowEngine(tenant, handlers).execute(
        "Give me the wealth view.",
        emitter=mock_emitter,
        session_id="session-wealth",
    )

    response = QueryResponse.from_flow_context(ctx)

    assert ctx.metadata["steps_executed"] == [
        "moderation:pre",
        "llm:refine_question",
        "llm:intent",
        "agent:planner",
        "aggregation",
        "llm:answer",
        "moderation:post",
        "analysis",
    ]
    assert ctx.intent == IntentResult(
        intent="market_outlook",
        confidence=0.94,
    )
    assert ctx.planner_output is not None
    assert ctx.planner_output.can_continue_to_aggregation is True
    assert ctx.planner_output.completed_tasks == ["search_documents"]
    assert ctx.aggregated_evidence is not None
    selected_evidence = ctx.aggregated_evidence.selected_evidence
    assert ctx.aggregated_evidence == AggregatedEvidenceBundle(
        user_query="Give me the wealth view.",
        standalone_query="What is the market view for this asset?",
        tenant_id=tenant_id,
        intent="market_outlook",
        active_skills=["wealth-skill"],
        selected_evidence=selected_evidence,
        missing_tasks=[],
        partial_tasks=[],
        stale_tasks=[],
        failed_tasks=[],
        excluded_evidence=[],
        synthesis_allowed=True,
        synthesis_block_reason=None,
    )
    assert [item.content for item in selected_evidence] == [
        "Evidence one for wealth analysis.",
        "Evidence two for wealth analysis.",
    ]
    assert ctx.llm_response == "Draft wealth answer [1]."
    assert response.answer == "Draft wealth answer [1]."
    assert len(response.citations) == 1
    assert response.citations[0].index == 1
    assert response.citations[0].evidence_id == "evidence:1:doc-1"

    assert answer_agent.last_prompt is not None
    assert "Evidence one for wealth analysis." in answer_agent.last_prompt
    assert not hasattr(answer_agent.last_deps, "reference_data")
    assert ctx.new_messages
    persisted_request = ctx.new_messages[0]
    assert isinstance(persisted_request, ModelRequest)
    persisted_part = persisted_request.parts[0]
    assert isinstance(persisted_part, UserPromptPart)
    assert persisted_part.content == "Give me the wealth view."
    assert "Evidence one for wealth analysis." not in persisted_part.content

    # Verify event emission sequence
    calls = mock_emitter.mock_calls
    call_names = [call[0] for call in calls]

    token_positions = [i for i, name in enumerate(call_names) if name == "emit_token"]
    citation_positions = [
        i for i, name in enumerate(call_names) if name == "emit_citations"
    ]
    completed_positions = [
        i
        for i, name in enumerate(call_names)
        if name == "emit_step_completed" and calls[i][1][0] == "llm:answer"
    ]

    assert token_positions, "Should have streamed some tokens"
    assert citation_positions, "Should have emitted citations"
    assert completed_positions, "Should have completed llm:answer step"

    for t_pos in token_positions:
        for c_pos in citation_positions:
            assert t_pos < c_pos, (
                f"Token emission at {t_pos} must occur before citation emission at {c_pos}"
            )

    for c_pos in citation_positions:
        for comp_pos in completed_positions:
            assert c_pos < comp_pos, (
                f"Citation emission at {c_pos} must occur before step completion at {comp_pos}"
            )

    assert ctx.metadata["analysis"]["streaming_policy"] == "token"
    assert ctx.metadata["analysis"]["planner_can_continue_to_aggregation"] is True
    assert planner_agent.last_deps is not None
    assert planner_agent.last_deps.available_tool_names == ["search_documents"]
    assert planner_agent.last_deps.flow_context.metadata["tenant_id"] == tenant_id


@pytest.mark.asyncio
async def test_query_understanding_planner_aggregation_answer_workflow(
    mock_emitter: unittest.mock.AsyncMock,
) -> None:
    """A combined understanding step can replace refine_question + intent."""
    tenant_id = "wealth-combined-tenant"

    def _planner_setup(deps: Any) -> None:
        flow_ctx = deps.flow_context
        assert flow_ctx is not None
        flow_ctx.metadata["tenant_id"] = tenant_id
        flow_ctx.active_skills.append("wealth-skill")
        deps.activated_skill_names.append("wealth-skill")
        flow_ctx.tool_results.append(_tool_result())
        flow_ctx.tool_calls.append(
            ToolCallRecord(
                tool_call_id="search_documents:1",
                tool_name="search_documents",
                input_payload={"query": "wealth query"},
                status="success",
                result_count=2,
            )
        )
        flow_ctx.tool_observations.append(
            ToolObservation(
                tool_name="search_documents",
                status="success",
                task_status_hint="completed",
                result_count=2,
            )
        )

    understanding_agent = FakeAgent(
        output=QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query="ignored",
                standalone_query="What is the market view for this asset?",
            ),
            intent=IntentResult(
                intent="market_outlook",
                confidence=0.94,
            ),
        ),
    )
    planner_agent = FakeAgent(
        output=PlannerOutput(
            can_continue_to_aggregation=True,
            reason="Evidence is sufficient.",
            completed_tasks=["search_documents"],
        ),
        on_enter=_planner_setup,
    )
    answer_agent = FakeAgent(
        output="Combined answer.",
        chunks=["Combined ", "answer."],
    )

    llm_handler = _llm_handler()
    llm_handler.set_agent_override(
        "query_understanding", "intent", cast(Agent[Any, Any], understanding_agent)
    )
    llm_handler.set_agent_override("answer", "pro", cast(Agent[Any, Any], answer_agent))

    agent_config = AgentConfig(
        llm_type="pro",
        built_in_tools=["search_documents"],
    )
    steps = [
        FlowStep(type=FlowStepType.LLM, mode="query_understanding", model="intent"),
        FlowStep(type=FlowStepType.AGENT, mode="planner", agent_config=agent_config),
        FlowStep(type=FlowStepType.AGGREGATION),
        FlowStep(type=FlowStepType.LLM, mode="answer", model="pro"),
        FlowStep(type=FlowStepType.ANALYSIS),
    ]
    tenant = TenantConfig(
        kms_app_name=tenant_id,
        application_id=tenant_id,
        ad_groups=[],
        llm_config=LLMConfig(models={}),
        flow_config=FlowConfig(steps=steps),
    )

    planner_handler = _planner_handler(
        tenant_id=tenant_id,
        agent_config=agent_config,
        providers=TenantProviders(),
        agent=planner_agent,
    )
    handlers = {
        FlowStepType.LLM: llm_handler,
        FlowStepType.AGENT: planner_handler,
        FlowStepType.AGGREGATION: AggregationHandler(),
        FlowStepType.ANALYSIS: AnalysisHandler(),
    }

    ctx = await FlowEngine(tenant, handlers).execute(
        "Give me the wealth view.",
        emitter=mock_emitter,
        session_id="session-combined",
    )

    assert ctx.metadata["steps_executed"] == [
        "llm:query_understanding",
        "agent:planner",
        "aggregation",
        "llm:answer",
        "analysis",
    ]
    assert "llm:refine_question" not in ctx.metadata["steps_executed"]
    assert "llm:intent" not in ctx.metadata["steps_executed"]
    assert ctx.refined_query == "What is the market view for this asset?"
    assert ctx.resolved_query == ResolvedQuery(
        original_query="Give me the wealth view.",
        standalone_query="What is the market view for this asset?",
    )
    assert ctx.intent == IntentResult(
        intent="market_outlook",
        confidence=0.94,
    )
    assert planner_agent.last_prompt is not None
    assert '"intent": "market_outlook"' in planner_agent.last_prompt
    assert "What is the market view for this asset?" in planner_agent.last_prompt
    assert ctx.aggregated_evidence is not None
    assert ctx.aggregated_evidence.synthesis_allowed is True
    assert [item.content for item in ctx.aggregated_evidence.selected_evidence] == [
        "Evidence one for wealth analysis.",
        "Evidence two for wealth analysis.",
    ]
    assert ctx.llm_response == "Combined answer."
    assert ctx.metadata["analysis"]["streaming_policy"] == "token"


@pytest.mark.asyncio
async def test_supervisor_agent_workflow_end_to_end(
    mock_emitter: unittest.mock.AsyncMock,
) -> None:
    """A single-agent open-agent flow can complete end to end with built-in tools."""
    tenant_id = "open-agent-tenant"
    agent_config = AgentConfig(
        llm_type="pro",
        built_in_tools=["search_documents", "rank_documents"],
    )

    def _supervisor_setup(deps: Any) -> None:
        flow_ctx = deps.flow_context
        assert flow_ctx is not None
        flow_ctx.metadata["tenant_id"] = tenant_id
        flow_ctx.tool_calls.append(
            ToolCallRecord(
                tool_call_id="plan_and_reason:1",
                tool_name="plan_and_reason",
                input_payload={"reasoning": "Inspect the repository."},
                status="success",
                result_count=1,
            )
        )

    supervisor_agent = FakeAgent(
        output="Supervisor answer.",
        chunks=["Supervisor ", "Supervisor answer."],
        on_enter=_supervisor_setup,
    )
    handler = _supervisor_handler(
        tenant_id=tenant_id,
        agent_config=agent_config,
        providers=TenantProviders(),
        agent=supervisor_agent,
    )
    steps = [
        FlowStep(type=FlowStepType.AGENT, agent_config=agent_config),
        FlowStep(type=FlowStepType.ANALYSIS),
    ]
    tenant = TenantConfig(
        kms_app_name=tenant_id,
        application_id=tenant_id,
        ad_groups=[],
        llm_config=LLMConfig(models={}),
        flow_config=FlowConfig(steps=steps),
    )

    ctx = await FlowEngine(
        tenant,
        {
            FlowStepType.AGENT: handler,
            FlowStepType.ANALYSIS: AnalysisHandler(),
        },
    ).execute(
        "Inspect the repository.",
        emitter=mock_emitter,
        session_id="session-open-agent",
    )

    response = QueryResponse.from_flow_context(ctx)

    assert ctx.planner_output is None
    assert ctx.llm_response == "Supervisor answer."
    assert ctx.tool_calls[0].tool_name == "plan_and_reason"
    assert supervisor_agent.last_deps is not None
    assert supervisor_agent.last_deps.available_tool_names == [
        "search_documents",
        "rank_documents",
    ]
    assert supervisor_agent.last_deps.flow_context is ctx
    assert response.answer == "Supervisor answer."
    assert ctx.metadata["analysis"]["tool_call_count"] == 1
    assert ctx.metadata["analysis"]["streaming_policy"] == "token"
    mock_emitter.emit_token.assert_any_await("Supervisor ")
    mock_emitter.emit_token.assert_any_await("answer.")
    mock_emitter.emit_step_completed.assert_any_await(
        "agent:orchestration",
        {"output_length": len("Supervisor answer.")},
    )
