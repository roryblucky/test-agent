"""First bounded Agent Research path behind shared v2 request lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, NotRequired, Protocol, TypedDict, cast

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import (  # pyright: ignore[reportMissingTypeStubs]
    add_messages,
)
from langgraph.types import Overwrite
from pydantic import BaseModel, ConfigDict

from app.langgraph_v2.agent_batch import (
    AcceptedBatch,
    AcceptedTask,
    ActiveBatch,
    BatchContribution,
    DispatchBatch,
    SpecialistRegistry,
    accept_initial_dispatch,
    execute_specialist,
    promote_batch,
)
from app.langgraph_v2.agent_completion import (
    IncompleteResearch,
    insufficient_evidence_answer,
)
from app.langgraph_v2.agent_evidence import (
    FinancialResearchReport,
    PreparedSynthesis,
    RequestEvidenceCatalog,
    prepare_synthesis,
    publish_report,
)
from app.langgraph_v2.agent_scope import (
    AgentIntentPolicy,
    ResearchScope,
    SpecialistDescriptor,
    resolve_research_scope,
)
from app.langgraph_v2.checkpointing import AgentCheckpointStateAdapter
from app.langgraph_v2.contracts import LiveStreamEvent, V2QueryResponse
from app.langgraph_v2.conversation_context import (
    DEFAULT_HISTORY_TOKEN_BUDGET,
    ConversationExchange,
    assistant_conversation_message,
    request_user_message_update,
    select_conversation_context,
)
from app.langgraph_v2.pre_moderation import ModerationProvider, run_pre_moderation
from app.langgraph_v2.stream import RequestOwnedGraph
from app.models.workflow import (
    CitationReference,
    IntentResult,
    QueryUnderstandingClarification,
    QueryUnderstandingOutput,
)


class QueryUnderstandingActor(Protocol):
    """Resolve one request with bounded complete Conversation context."""

    async def understand(
        self,
        query: str,
        history: Sequence[ConversationExchange],
    ) -> QueryUnderstandingOutput:
        """Return the typed query-understanding result for one request."""
        ...


class Finish(BaseModel):
    """Coordinator choice to end research without a business payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["finish"]


class CoordinatorInput(BaseModel):
    """Only prompt-visible input permitted for the initial Coordinator call."""

    model_config = ConfigDict(frozen=True)

    standalone_query: str
    intent: str
    specialist_descriptors: tuple[SpecialistDescriptor, ...]


CoordinatorDecision = Finish | DispatchBatch


class CoordinatorActor(Protocol):
    """Propose one bounded Coordinator decision."""

    async def decide(self, input: CoordinatorInput) -> CoordinatorDecision:
        """Return the typed Coordinator decision."""
        ...


class SynthesisActor(Protocol):
    """Turn a frozen Evidence projection into one report candidate."""

    async def synthesize(self, prepared: PreparedSynthesis) -> FinancialResearchReport:
        """Return one model-authored Markdown candidate."""
        ...


def _merge_contributions(
    current: dict[str, dict[str, Any]],
    update: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge immutable staged contributions, rejecting conflicting identities."""
    merged = {**current}
    for task_id, contribution in update.items():
        existing = merged.get(task_id)
        if existing is not None and existing != contribution:
            raise ValueError("Batch contribution conflicts")
        merged[task_id] = contribution
    return merged


def _merge_accepted_batches(
    current: dict[str, dict[str, Any]],
    update: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge immutable accepted batches, rejecting conflicting identities."""
    return _merge_contributions(current, update)


class AgentGraphState(TypedDict):
    """Persisted state needed by clarification and Finish-first paths."""

    query: str
    conversation_id: str
    request_id: str
    conversation_messages: NotRequired[Annotated[list[BaseMessage], add_messages]]
    halted: NotRequired[bool]
    standalone_query: NotRequired[str | None]
    intent: NotRequired[dict[str, Any] | None]
    research_scope: NotRequired[dict[str, Any] | None]
    active_batch: NotRequired[dict[str, Any] | None]
    staged_contributions: NotRequired[
        Annotated[dict[str, dict[str, Any]], _merge_contributions]
    ]
    accepted_batches: NotRequired[
        Annotated[dict[str, dict[str, Any]], _merge_accepted_batches]
    ]
    clarification: NotRequired[dict[str, Any] | None]
    answer: NotRequired[str | None]
    completion_status: NotRequired[str | None]
    termination_reason: NotRequired[str | None]
    final_response: NotRequired[dict[str, Any] | None]
    citations: NotRequired[list[dict[str, Any]]]


class AgentGraphStateUpdate(TypedDict, total=False):
    """Partial Agent Graph update returned by one node."""

    conversation_messages: list[BaseMessage]
    halted: bool
    standalone_query: str | None
    intent: dict[str, Any] | None
    research_scope: dict[str, Any] | None
    active_batch: dict[str, Any] | None
    staged_contributions: Any
    accepted_batches: dict[str, dict[str, Any]]
    clarification: dict[str, Any] | None
    answer: str | None
    completion_status: str | None
    termination_reason: str | None
    final_response: dict[str, Any] | None
    citations: list[dict[str, Any]]


def _emit(events: Sequence[LiveStreamEvent]) -> None:
    writer = get_stream_writer()
    for event in events:
        writer(event.to_stream_payload())


def _clarification_answer(clarification: QueryUnderstandingClarification) -> str:
    questions = clarification.questions
    if not 1 <= len(questions) <= 3:
        raise ValueError(
            "Query Understanding clarification requires one to three questions"
        )
    if any(not question.question.strip() for question in questions):
        raise ValueError(
            "Query Understanding clarification questions must not be blank"
        )
    if any(len(question.options) > 4 for question in questions):
        raise ValueError("Query Understanding clarification options exceed the limit")
    return questions[0].question


def build_agent_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    query_understanding_actor: QueryUnderstandingActor,
    coordinator_actor: CoordinatorActor,
    specialist_registry: SpecialistRegistry,
    intent_policies: Mapping[str, AgentIntentPolicy],
    moderation_provider: ModerationProvider,
    tenant_id: str,
    evidence_catalog: RequestEvidenceCatalog | None = None,
    synthesis_actor: SynthesisActor | None = None,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    checkpoint_state_adapter: AgentCheckpointStateAdapter | None = None,
) -> RequestOwnedGraph:
    """Compile clarification plus first legal Coordinator Finish path."""
    state_adapter = checkpoint_state_adapter or AgentCheckpointStateAdapter()
    catalog = evidence_catalog or RequestEvidenceCatalog()
    builder: StateGraph[AgentGraphState, None, AgentGraphState, AgentGraphState] = (
        StateGraph(AgentGraphState)
    )

    async def initializer(state: AgentGraphState) -> AgentGraphStateUpdate:
        state_adapter.validate_checkpoint_state(state)
        return {
            "conversation_messages": request_user_message_update(
                state.get("conversation_messages", []),
                request_id=state["request_id"],
                query=state["query"],
            ),
            "halted": False,
            "standalone_query": None,
            "intent": None,
            "research_scope": None,
            "active_batch": None,
            "staged_contributions": cast(Any, Overwrite({})),
            "accepted_batches": cast(Any, Overwrite({})),
            "clarification": None,
            "answer": None,
            "completion_status": None,
            "termination_reason": None,
            "final_response": None,
            "citations": [],
        }

    async def pre_moderation(state: AgentGraphState) -> AgentGraphStateUpdate:
        events, halted, _ = await run_pre_moderation(
            state,
            provider=moderation_provider,
        )
        _emit(events)
        return {"halted": halted}

    async def query_understanding(state: AgentGraphState) -> AgentGraphStateUpdate:
        history = select_conversation_context(
            state.get("conversation_messages", []),
            token_budget=history_token_budget,
            current_request_id=state["request_id"],
        )
        result = await query_understanding_actor.understand(state["query"], history)
        clarification = result.clarification
        _emit(
            (
                LiveStreamEvent(type="step_start", step="query_understanding"),
                LiveStreamEvent(
                    type="step_completed",
                    step="query_understanding",
                    data={"needs_clarification": clarification is not None},
                ),
            )
        )
        if clarification is not None:
            return {
                "standalone_query": result.resolved_query.standalone_query,
                "clarification": clarification.model_dump(mode="json"),
                "answer": _clarification_answer(clarification),
            }
        standalone_query = result.resolved_query.standalone_query.strip()
        if not standalone_query:
            raise ValueError("Query Understanding standalone query must not be blank")
        return {
            "standalone_query": standalone_query,
            "intent": result.intent.model_dump(mode="json"),
        }

    async def resolve_scope(state: AgentGraphState) -> AgentGraphStateUpdate:
        intent_value = state.get("intent")
        if not isinstance(intent_value, dict):
            raise TypeError("Agent Intent is invalid")
        intent = IntentResult.model_validate(intent_value)
        scope = resolve_research_scope(intent, intent_policies)
        return {"research_scope": scope.model_dump(mode="json")}

    async def coordinator(state: AgentGraphState) -> AgentGraphStateUpdate:
        standalone_query = state.get("standalone_query")
        scope_value = state.get("research_scope")
        if not isinstance(standalone_query, str):
            raise TypeError("Agent standalone query is invalid")
        if not isinstance(scope_value, dict):
            raise TypeError("Agent Research Scope is invalid")
        scope = ResearchScope.model_validate(scope_value)
        _emit((LiveStreamEvent(type="step_start", step="coordinator"),))
        decision = await coordinator_actor.decide(
            CoordinatorInput(
                standalone_query=standalone_query,
                intent=scope.intent,
                specialist_descriptors=scope.specialist_descriptors,
            )
        )
        _emit(
            (
                LiveStreamEvent(type="step_completed", step="coordinator"),
            )
        )
        if isinstance(decision, Finish):
            return {}
        if state.get("accepted_batches"):
            raise ValueError(
                "Coordinator cannot Dispatch after initial batch acceptance"
            )
        dispatch = DispatchBatch.model_validate(decision)
        active_batch = accept_initial_dispatch(
            dispatch,
            request_id=state["request_id"],
            registry=specialist_registry,
            scope_descriptors=scope.specialist_descriptors,
        )
        return {"active_batch": _active_batch_dump(active_batch)}

    async def execute_first_specialist(
        state: AgentGraphState,
    ) -> AgentGraphStateUpdate:
        active_value = state.get("active_batch")
        scope_value = state.get("research_scope")
        if not isinstance(active_value, dict) or not isinstance(scope_value, dict):
            raise TypeError("Agent active batch is invalid")
        active_batch = _active_batch_load(active_value)
        scope = ResearchScope.model_validate(scope_value)
        task = active_batch.tasks[0]
        _emit((LiveStreamEvent(type="step_start", step="specialist"),))
        contribution = await execute_specialist(
            task,
            batch_id=active_batch.id,
            registry=specialist_registry,
            scope_descriptors=scope.specialist_descriptors,
            catalog=catalog,
            tenant_id=tenant_id,
            request_id=state["request_id"],
            scope_tool_ids=scope.allowed_tool_ids,
            scope_sources=scope.allowed_sources,
            scope_queries=scope.allowed_queries,
            tool_telemetry=lambda tool_id, status: _emit(
                (
                    LiveStreamEvent(
                        type="progress",
                        step="tool",
                        data={"task_id": task.id, "tool_id": tool_id, "status": status},
                    ),
                )
            ),
        )
        _emit((LiveStreamEvent(type="step_completed", step="specialist"),))
        return {
            "staged_contributions": {
                contribution.task_id: contribution.model_dump(mode="json")
            }
        }

    async def batch_barrier(state: AgentGraphState) -> AgentGraphStateUpdate:
        active_value = state.get("active_batch")
        staged_value = state.get("staged_contributions", {})
        if not isinstance(active_value, dict):
            raise TypeError("Agent batch state is invalid")
        active_batch = _active_batch_load(active_value)
        contributions = {
            task_id: BatchContribution.model_validate(value)
            for task_id, value in staged_value.items()
        }
        accepted = promote_batch(active_batch, contributions)
        return {
            "accepted_batches": {accepted.id: accepted.model_dump(mode="json")},
            "staged_contributions": cast(Any, Overwrite({})),
            "active_batch": None,
        }

    async def research_completion(state: AgentGraphState) -> AgentGraphStateUpdate:
        del state
        completion = IncompleteResearch(insufficient_evidence=True)
        return {
            "answer": insufficient_evidence_answer(completion),
            "completion_status": "incomplete",
            "termination_reason": "insufficient_evidence",
        }

    async def synthesis(state: AgentGraphState) -> AgentGraphStateUpdate:
        if synthesis_actor is None:
            raise RuntimeError("Agent Synthesis actor is not configured")
        standalone_query = state.get("standalone_query")
        scope_value = state.get("research_scope")
        if not isinstance(standalone_query, str) or not isinstance(scope_value, dict):
            raise TypeError("Agent Synthesis input is invalid")
        scope = ResearchScope.model_validate(scope_value)
        evidence_ids = _accepted_evidence_ids(state)
        prepared = prepare_synthesis(
            standalone_query=standalone_query,
            intent=scope.intent,
            accepted_evidence_ids=evidence_ids,
            catalog=catalog,
            tenant_id=tenant_id,
            request_id=state["request_id"],
            as_of_date=scope.as_of_date,
            max_evidence_age_days=scope.max_evidence_age_days,
        )
        candidate = await synthesis_actor.synthesize(prepared)
        published = publish_report(candidate, prepared)
        return {
            "answer": published.answer,
            "citations": [item.model_dump(mode="json") for item in published.citations],
            "completion_status": "complete",
            "termination_reason": "evidence_backed",
        }

    async def finalize_state(state: AgentGraphState) -> AgentGraphStateUpdate:
        answer = state.get("answer")
        if not isinstance(answer, str):
            raise TypeError("Agent final answer is invalid")
        clarification_value = state.get("clarification")
        clarification = (
            QueryUnderstandingClarification.model_validate(clarification_value)
            if isinstance(clarification_value, dict)
            else None
        )
        metadata: dict[str, Any] = {
            "steps_executed": [
                "initializer",
                "pre_moderation",
                "query_understanding",
            ]
        }
        if clarification is None:
            completion_status = state.get("completion_status")
            termination_reason = state.get("termination_reason")
            if completion_status == "incomplete":
                if termination_reason != "insufficient_evidence":
                    raise TypeError("Agent research completion is invalid")
            elif (
                completion_status != "complete"
                or termination_reason != "evidence_backed"
            ):
                raise TypeError("Agent research completion is invalid")
            metadata["steps_executed"].extend(["resolve_scope", "coordinator"])
            if state.get("accepted_batches"):
                metadata["steps_executed"].extend(
                    ["execute_first_specialist", "batch_barrier", "coordinator"]
                )
            metadata["steps_executed"].append(
                "synthesis"
                if completion_status == "complete"
                else "research_completion"
            )
            metadata["completion_status"] = completion_status
            metadata["termination_reason"] = termination_reason
        metadata["steps_executed"].append("finalize_state")
        raw_citations = state.get("citations", [])
        response = V2QueryResponse(
            query=state["query"],
            answer=answer,
            clarification=clarification,
            conversation_id=state["conversation_id"],
            metadata=metadata,
            citations=[
                CitationReference.model_validate(citation) for citation in raw_citations
            ],
        )
        return {
            "final_response": response.model_dump(mode="json", by_alias=True),
            "conversation_messages": [
                assistant_conversation_message(state["request_id"], answer)
            ],
        }

    async def publish(state: AgentGraphState) -> AgentGraphStateUpdate:
        final_response = state.get("final_response")
        if not isinstance(final_response, dict):
            raise TypeError("Agent final response is invalid")
        response = V2QueryResponse.model_validate(final_response)
        _emit(
            (
                LiveStreamEvent(
                    type="done",
                    data=response.model_dump(by_alias=True),
                    checkpoint_terminal=True,
                ),
            )
        )
        return {}

    def next_after_pre_moderation(state: AgentGraphState) -> str:
        return "end" if state.get("halted", False) else "query_understanding"

    def next_after_query_understanding(state: AgentGraphState) -> str:
        return (
            "finalize_state"
            if state.get("clarification") is not None
            else "resolve_scope"
        )

    def next_after_coordinator(state: AgentGraphState) -> str:
        if state.get("active_batch") is not None:
            return "execute_first_specialist"
        return "synthesis" if _accepted_evidence_ids(state) else "research_completion"

    builder.add_node("initializer", initializer)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("pre_moderation", pre_moderation)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("query_understanding", query_understanding)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("resolve_scope", resolve_scope)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("coordinator", coordinator)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("execute_first_specialist", execute_first_specialist)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("batch_barrier", batch_barrier)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("research_completion", research_completion)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("synthesis", synthesis)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("finalize_state", finalize_state)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("publish", publish)  # pyright: ignore[reportUnknownMemberType]
    builder.add_edge(START, "initializer")
    builder.add_edge("initializer", "pre_moderation")
    builder.add_conditional_edges(
        "pre_moderation",
        next_after_pre_moderation,
        {"query_understanding": "query_understanding", "end": END},
    )
    builder.add_conditional_edges(
        "query_understanding",
        next_after_query_understanding,
        {"resolve_scope": "resolve_scope", "finalize_state": "finalize_state"},
    )
    builder.add_edge("resolve_scope", "coordinator")
    builder.add_conditional_edges(
        "coordinator",
        next_after_coordinator,
        {
            "execute_first_specialist": "execute_first_specialist",
            "research_completion": "research_completion",
            "synthesis": "synthesis",
        },
    )
    builder.add_edge("execute_first_specialist", "batch_barrier")
    builder.add_edge("batch_barrier", "coordinator")
    builder.add_edge("research_completion", "finalize_state")
    builder.add_edge("synthesis", "finalize_state")
    builder.add_edge("finalize_state", "publish")
    builder.add_edge("publish", END)
    return cast(
        RequestOwnedGraph,
        builder.compile(  # pyright: ignore[reportUnknownMemberType]
            checkpointer=checkpointer
        ),
    )


def _accepted_evidence_ids(state: AgentGraphState) -> tuple[str, ...]:
    """Return accepted Evidence IDs in deterministic batch and Task order."""
    accepted_batches = state.get("accepted_batches", {})
    evidence_ids: list[str] = []
    for raw_batch in accepted_batches.values():
        accepted = AcceptedBatch.model_validate(raw_batch)
        for outcome in accepted.outcomes:
            evidence_ids.extend(outcome.result.evidence_ids)
    return tuple(evidence_ids)


def _active_batch_dump(batch: ActiveBatch) -> dict[str, Any]:
    """Serialize the scalar active-batch manifest for checkpoint state."""
    return {
        "id": batch.id,
        "tasks": [
            {
                "id": task.id,
                "objective": task.objective,
                "specialist_id": task.specialist_id,
            }
            for task in batch.tasks
        ],
    }


def _active_batch_load(value: Mapping[str, object]) -> ActiveBatch:
    """Validate one scalar active-batch manifest from checkpoint state."""
    raw_tasks = value.get("tasks")
    batch_id = value.get("id")
    if not isinstance(batch_id, str) or not isinstance(raw_tasks, list):
        raise TypeError("Agent active batch is invalid")
    tasks: list[AcceptedTask] = []
    for raw_task in cast(list[object], raw_tasks):
        if not isinstance(raw_task, Mapping):
            raise TypeError("Agent active batch is invalid")
        record = cast(Mapping[str, object], raw_task)
        task_id = record.get("id")
        objective = record.get("objective")
        specialist_id = record.get("specialist_id")
        if (
            not isinstance(task_id, str)
            or not isinstance(objective, str)
            or not isinstance(specialist_id, str)
        ):
            raise TypeError("Agent active batch is invalid")
        tasks.append(
            AcceptedTask(
                id=task_id,
                objective=objective,
                specialist_id=specialist_id,
            )
        )
    return ActiveBatch(id=batch_id, tasks=tuple(tasks))
