"""First bounded Agent Research path behind shared v2 request lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, NotRequired, Protocol, TypedDict, cast

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
from langgraph.types import Overwrite, Send
from pydantic import ValidationError

from app.langgraph_v2.agent_batch import (
    AcceptedBatch,
    AcceptedTask,
    ActiveBatch,
    BatchContribution,
    SpecialistRegistry,
    SpecialistUsage,
    TaskSucceeded,
    execute_specialist,
    promote_batch,
    validate_active_batch_manifest,
    validate_promoted_calculation_contribution,
)
from app.langgraph_v2.agent_completion import (
    FailedTaskDisclosure,
    IncompleteResearch,
    completion_termination_reason,
    insufficient_evidence_answer,
    render_incomplete_research,
)
from app.langgraph_v2.agent_coordination import (
    AcceptedCoordinationDispatch,
    CoordinationInvariantError,
    CoordinationRound,
    CoordinationStopped,
    CoordinatorActor,
    StructuralStopReason,
    decide_coordination_round,
    materialize_specialist_context,
    project_coordinator_input,
    validate_active_batch_coordination_round,
    validate_coordination_rounds,
)
from app.langgraph_v2.agent_evidence import (
    DataGapView,
    EvidenceInvocationContext,
    RequestEvidenceCatalog,
    SynthesisActor,
    prepare_synthesis,
    synthesize_report,
)
from app.langgraph_v2.agent_scope import (
    AgentIntentPolicy,
    ResearchScope,
    resolve_research_scope,
)
from app.langgraph_v2.agent_termination import COORDINATION_STOP_REASONS
from app.langgraph_v2.calculations import (
    CalculationArtifact,
    accepted_calculation_artifacts,
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
from app.langgraph_v2.specialist_retry import SpecialistExecutionDiagnostics
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


def _merge_staged_contributions(
    current: dict[str, dict[str, Any]],
    update: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge immutable staged Task contributions."""
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
    """Merge immutable accepted Batch records."""
    merged = {**current}
    for batch_id, accepted_batch in update.items():
        existing = merged.get(batch_id)
        if existing is not None and existing != accepted_batch:
            raise ValueError("Accepted Batch conflicts")
        merged[batch_id] = accepted_batch
    return merged


def _merge_coordination_rounds(
    current: dict[str, dict[str, Any]],
    update: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge immutable accepted Coordinator Decisions by stable identity."""
    merged = {**current}
    for round_id, coordination_round in update.items():
        existing = merged.get(round_id)
        if existing is not None and existing != coordination_round:
            raise ValueError("Coordination Round conflicts")
        merged[round_id] = coordination_round
    return merged


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
    dispatched_task: NotRequired[dict[str, Any] | None]
    staged_contributions: NotRequired[
        Annotated[dict[str, dict[str, Any]], _merge_staged_contributions]
    ]
    accepted_batches: NotRequired[
        Annotated[dict[str, dict[str, Any]], _merge_accepted_batches]
    ]
    coordination_rounds: NotRequired[
        Annotated[dict[str, dict[str, Any]], _merge_coordination_rounds]
    ]
    coordination_request_id: NotRequired[str | None]
    coordination_finished: NotRequired[bool]
    coordination_stop_reason: NotRequired[str | None]
    clarification: NotRequired[dict[str, Any] | None]
    answer: NotRequired[str | None]
    completion_status: NotRequired[str | None]
    termination_reason: NotRequired[str | None]
    incomplete_research: NotRequired[dict[str, Any] | None]
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
    dispatched_task: dict[str, Any] | None
    staged_contributions: Any
    accepted_batches: dict[str, dict[str, Any]]
    coordination_rounds: dict[str, dict[str, Any]]
    coordination_request_id: str | None
    coordination_finished: bool
    coordination_stop_reason: str | None
    clarification: dict[str, Any] | None
    answer: str | None
    completion_status: str | None
    termination_reason: str | None
    incomplete_research: dict[str, Any] | None
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
    specialist_diagnostics = SpecialistExecutionDiagnostics()
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
            "dispatched_task": None,
            "staged_contributions": cast(Any, Overwrite({})),
            "accepted_batches": cast(Any, Overwrite({})),
            "coordination_rounds": cast(Any, Overwrite({})),
            "coordination_request_id": None,
            "coordination_finished": False,
            "coordination_stop_reason": None,
            "clarification": None,
            "answer": None,
            "completion_status": None,
            "termination_reason": None,
            "incomplete_research": None,
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
        rounds = _coordination_rounds(state)
        accepted_batches = _accepted_batches(state)
        input = project_coordinator_input(
            standalone_query=standalone_query,
            intent=scope.intent,
            specialist_descriptors=scope.specialist_descriptors,
            rounds=rounds,
            accepted_batches=accepted_batches,
        )
        _emit((LiveStreamEvent(type="step_start", step="coordinator"),))
        decision = await decide_coordination_round(
            coordinator_actor,
            input,
            request_id=state["request_id"],
            rounds=rounds,
            accepted_batches=accepted_batches,
            registry=specialist_registry,
            scope_descriptors=scope.specialist_descriptors,
        )
        _emit((LiveStreamEvent(type="step_completed", step="coordinator"),))
        if isinstance(decision, CoordinationStopped):
            return {
                "coordination_finished": True,
                "coordination_stop_reason": decision.reason,
            }
        if isinstance(decision, AcceptedCoordinationDispatch):
            return {
                "coordination_rounds": {
                    decision.round.id: decision.round.model_dump(mode="json")
                },
                "coordination_request_id": state["request_id"],
                "active_batch": _active_batch_dump(decision.active_batch),
            }
        return {
            "coordination_rounds": {decision.id: decision.model_dump(mode="json")},
            "coordination_request_id": state["request_id"],
            "coordination_finished": True,
        }

    async def execute_specialist_task(
        state: AgentGraphState,
    ) -> AgentGraphStateUpdate:
        active_value = state.get("active_batch")
        scope_value = state.get("research_scope")
        task_value = state.get("dispatched_task")
        if (
            not isinstance(active_value, dict)
            or not isinstance(scope_value, dict)
            or not isinstance(task_value, dict)
        ):
            raise TypeError("Agent active batch is invalid")
        active_batch = _active_batch_load(active_value)
        scope = ResearchScope.model_validate(scope_value)
        validate_active_batch_manifest(
            active_batch,
            request_id=state["request_id"],
            registry=specialist_registry,
            scope_descriptors=scope.specialist_descriptors,
        )
        validate_active_batch_coordination_round(
            active_batch,
            rounds=_coordination_rounds(state),
        )
        task = _accepted_task_load(task_value)
        if task not in active_batch.tasks:
            raise ValueError("Dispatched Specialist Task is not in the active batch")
        context_results = materialize_specialist_context(
            task,
            current_round=active_batch.round,
            rounds=_coordination_rounds(state),
            accepted_batches=_accepted_batches(state),
        )
        invocation_context = EvidenceInvocationContext(
            tenant_id=tenant_id,
            request_id=state["request_id"],
            task_id=task.id,
            allowed_tool_ids=scope.allowed_tool_ids,
            allowed_sources=scope.allowed_sources,
            allowed_queries=scope.allowed_queries,
        )
        _emit((LiveStreamEvent(type="step_start", step="specialist"),))
        contribution = await execute_specialist(
            task,
            batch_id=active_batch.id,
            registry=specialist_registry,
            scope_descriptors=scope.specialist_descriptors,
            catalog=catalog,
            context=invocation_context,
            scope_skill_names=scope.allowed_skill_names,
            context_results=context_results,
            diagnostics=specialist_diagnostics,
            tool_telemetry=lambda tool_id, status: _emit(
                (
                    LiveStreamEvent(
                        type="progress",
                        step="tool",
                        data={
                            "task_id": task.id,
                            "tool_id": tool_id,
                            "status": status,
                        },
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
        scope_value = state.get("research_scope")
        if not isinstance(scope_value, dict):
            raise TypeError("Agent Research Scope is invalid")
        scope = ResearchScope.model_validate(scope_value)
        validate_active_batch_coordination_round(
            active_batch,
            rounds=_coordination_rounds(state),
        )
        contributions = {
            task_id: BatchContribution.model_validate(value)
            for task_id, value in staged_value.items()
        }
        tasks_by_id = {task.id: task for task in active_batch.tasks}
        accepted = promote_batch(
            active_batch,
            contributions,
            calculation_validator=lambda contribution: (
                validate_promoted_calculation_contribution(
                    contribution,
                    task=tasks_by_id[contribution.task_id],
                    tenant_id=tenant_id,
                    request_id=state["request_id"],
                    registry=specialist_registry,
                    scope_descriptors=scope.specialist_descriptors,
                    scope_tool_ids=scope.allowed_tool_ids,
                    catalog=catalog,
                )
            ),
        )
        return {
            "accepted_batches": {accepted.id: accepted.model_dump(mode="json")},
            "staged_contributions": cast(Any, Overwrite({})),
            "active_batch": None,
        }

    async def research_completion(state: AgentGraphState) -> AgentGraphStateUpdate:
        stop_reason = _coordination_stop_reason(state)
        failed_tasks = _accepted_failed_task_disclosures(state)
        completion = IncompleteResearch(
            insufficient_evidence=not _accepted_evidence_ids(state),
            data_gaps=_accepted_data_gap_views(state),
            failed_task_ids=tuple(task.task_id for task in failed_tasks),
            structural_reasons=(stop_reason,) if stop_reason is not None else (),
        )
        return {
            "answer": insufficient_evidence_answer(
                completion,
                failed_tasks=failed_tasks,
            ),
            "completion_status": "incomplete",
            "termination_reason": completion_termination_reason(completion),
            "incomplete_research": completion.model_dump(mode="json"),
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
        data_gaps = _accepted_data_gap_views(state)
        stop_reason = _coordination_stop_reason(state)
        failed_tasks = _accepted_failed_task_disclosures(state)
        prepared = prepare_synthesis(
            standalone_query=standalone_query,
            intent=scope.intent,
            accepted_evidence_ids=evidence_ids,
            catalog=catalog,
            tenant_id=tenant_id,
            request_id=state["request_id"],
            as_of_date=scope.as_of_date,
            max_evidence_age_days=scope.max_evidence_age_days,
            data_gaps=data_gaps,
            accepted_calculations=_accepted_calculations(state),
        )
        published = await synthesize_report(synthesis_actor, prepared)
        completion = (
            IncompleteResearch(
                insufficient_evidence=False,
                data_gaps=data_gaps,
                failed_task_ids=tuple(task.task_id for task in failed_tasks),
                structural_reasons=(stop_reason,) if stop_reason is not None else (),
            )
            if data_gaps
            or failed_tasks
            or stop_reason is not None
            else None
        )
        return {
            "answer": (
                render_incomplete_research(
                    published.answer,
                    completion,
                    failed_tasks=failed_tasks,
                )
                if completion is not None
                else published.answer
            ),
            "citations": [item.model_dump(mode="json") for item in published.citations],
            "completion_status": "incomplete" if completion is not None else "complete",
            "termination_reason": (
                completion_termination_reason(completion)
                if completion is not None
                else "evidence_backed"
            ),
            "incomplete_research": (
                completion.model_dump(mode="json") if completion is not None else None
            ),
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
                if termination_reason not in {
                    "insufficient_evidence",
                    "partial_results",
                    "execution_limit",
                    "partial_results_and_execution_limit",
                }:
                    raise TypeError("Agent research completion is invalid")
            elif (
                completion_status != "complete"
                or termination_reason != "evidence_backed"
            ):
                raise TypeError("Agent research completion is invalid")
            metadata["steps_executed"].extend(["resolve_scope", "coordinator"])
            if state.get("accepted_batches"):
                metadata["steps_executed"].extend(
                    ["execute_specialist_task", "batch_barrier", "coordinator"]
                )
                metadata["specialist_usage"] = _accepted_specialist_usage(
                    state
                ).model_dump(mode="json")
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
        if response.answer is None:
            raise TypeError("Agent committed final answer is invalid")
        events = [LiveStreamEvent(type="token", data=response.answer)]
        if response.citations:
            events.append(
                LiveStreamEvent(
                    type="citations",
                    data=[citation.model_dump(mode="json") for citation in response.citations],
                )
            )
        events.append(
            LiveStreamEvent(
                type="done",
                data=response.model_dump(by_alias=True),
                checkpoint_terminal=True,
            )
        )
        _emit(tuple(events))
        return {}

    def next_after_pre_moderation(state: AgentGraphState) -> str:
        return "end" if state.get("halted", False) else "query_understanding"

    def next_after_query_understanding(state: AgentGraphState) -> str:
        return (
            "finalize_state"
            if state.get("clarification") is not None
            else "resolve_scope"
        )

    def next_after_coordinator(state: AgentGraphState) -> str | list[Send]:
        active_value = cast(object, state.get("active_batch"))
        if active_value is not None:
            if not isinstance(active_value, dict):
                raise TypeError("Agent active batch is invalid")
            scope_value = state.get("research_scope")
            if not isinstance(scope_value, dict):
                raise TypeError("Agent Research Scope is invalid")
            active_batch = _active_batch_load(cast(Mapping[str, object], active_value))
            scope = ResearchScope.model_validate(scope_value)
            validate_active_batch_manifest(
                active_batch,
                request_id=state["request_id"],
                registry=specialist_registry,
                scope_descriptors=scope.specialist_descriptors,
            )
            validate_active_batch_coordination_round(
                active_batch,
                rounds=_coordination_rounds(state),
            )
            return [
                Send(
                    "execute_specialist_task",
                    {**state, "dispatched_task": _accepted_task_dump(task)},
                )
                for task in active_batch.tasks
            ]
        if state.get("coordination_finished") is not True:
            raise CoordinationInvariantError(
                "Coordination did not reach a terminal decision"
            )
        return "synthesis" if _accepted_evidence_ids(state) else "research_completion"

    def next_after_batch_barrier(state: AgentGraphState) -> str:
        """Avoid another Coordinator decision when calculation state hit its cap."""
        if state.get("coordination_finished") is not True:
            return "coordinator"
        return "synthesis" if _accepted_evidence_ids(state) else "research_completion"

    builder.add_node("initializer", initializer)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("pre_moderation", pre_moderation)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("query_understanding", query_understanding)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("resolve_scope", resolve_scope)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("coordinator", coordinator)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("execute_specialist_task", execute_specialist_task)  # pyright: ignore[reportUnknownMemberType]
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
            "research_completion": "research_completion",
            "synthesis": "synthesis",
        },
    )
    builder.add_edge("execute_specialist_task", "batch_barrier")
    builder.add_conditional_edges(
        "batch_barrier",
        next_after_batch_barrier,
        {
            "coordinator": "coordinator",
            "research_completion": "research_completion",
            "synthesis": "synthesis",
        },
    )
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
    evidence_ids: list[str] = []
    for _, accepted in _accepted_round_batches(state):
        for outcome in accepted.outcomes:
            if isinstance(outcome, TaskSucceeded):
                evidence_ids.extend(outcome.result.evidence_ids)
    return tuple(evidence_ids)


def _accepted_data_gap_views(state: AgentGraphState) -> tuple[DataGapView, ...]:
    """Return safe Data Gap projections in deterministic accepted-state order."""
    return tuple(
        gap.view()
        for _, accepted in _accepted_round_batches(state)
        for outcome in accepted.outcomes
        if isinstance(outcome, TaskSucceeded)
        for gap in outcome.result.data_gaps
    )


def _accepted_failed_task_disclosures(
    state: AgentGraphState,
) -> tuple[FailedTaskDisclosure, ...]:
    """Pair failed Task IDs and accepted canonical objectives at publication time."""
    disclosures: list[FailedTaskDisclosure] = []
    for round_, accepted in _accepted_round_batches(state):
        outcomes = {outcome.task_id: outcome for outcome in accepted.outcomes}
        disclosures.extend(
            FailedTaskDisclosure(task_id=task.id, objective=task.objective)
            for task in round_.tasks
            if not isinstance(outcomes[task.id], TaskSucceeded)
        )
    return tuple(disclosures)


def _accepted_specialist_usage(state: AgentGraphState) -> SpecialistUsage:
    """Aggregate accepted Task accounting without exposing retry diagnostics."""
    usage = SpecialistUsage()
    for _, accepted in _accepted_round_batches(state):
        usage = usage.add(accepted.usage)
    return usage


def _coordination_rounds(state: AgentGraphState) -> tuple[CoordinationRound, ...]:
    """Load immutable Coordinator Decisions in revision order from checkpoint state."""
    raw_rounds = cast(object, state.get("coordination_rounds", {}))
    if not isinstance(raw_rounds, dict):
        raise TypeError("Agent Coordination Rounds are invalid")
    rounds = tuple(
        CoordinationRound.model_validate(value)
        for value in cast(Mapping[str, object], raw_rounds).values()
    )
    request_id = cast(object, state.get("request_id"))
    if not isinstance(request_id, str):
        raise TypeError("Agent request ID is invalid")
    return validate_coordination_rounds(rounds, request_id=request_id)


def _coordination_stop_reason(
    state: AgentGraphState,
) -> StructuralStopReason | None:
    """Load only the finite, code-owned structural terminal reasons."""
    value = state.get("coordination_stop_reason")
    if value is None:
        return None
    if value not in COORDINATION_STOP_REASONS:
        raise CoordinationInvariantError("Coordination stop reason is invalid")
    return cast(StructuralStopReason, value)


def _accepted_calculations(state: AgentGraphState) -> tuple[CalculationArtifact, ...]:
    """Return unique Artifacts in Round, manifest Task, then Artifact ID order."""
    ordered: list[CalculationArtifact] = []
    for round_, accepted in _accepted_round_batches(state):
        by_task: dict[str, list[CalculationArtifact]] = {}
        for artifact in accepted.calculations:
            by_task.setdefault(artifact.task_id, []).append(artifact)
        for task in round_.tasks:
            ordered.extend(sorted(by_task.get(task.id, ()), key=lambda item: item.id))
    return accepted_calculation_artifacts(tuple(ordered))


def _accepted_batches(state: AgentGraphState) -> dict[str, AcceptedBatch]:
    """Load immutable promoted batches without relying on checkpoint map insertion."""
    raw_batches = cast(object, state.get("accepted_batches", {}))
    if not isinstance(raw_batches, dict):
        raise TypeError("Agent Accepted Batches are invalid")
    return {
        batch_id: AcceptedBatch.model_validate(value)
        for batch_id, value in cast(Mapping[str, object], raw_batches).items()
    }


def _accepted_round_batches(
    state: AgentGraphState,
) -> tuple[tuple[CoordinationRound, AcceptedBatch], ...]:
    """Pair every dispatch manifest with its exact accepted batch outcomes."""
    accepted_batches = _accepted_batches(state)
    dispatch_rounds = tuple(
        round_ for round_ in _coordination_rounds(state) if round_.kind == "dispatch"
    )
    missing_batch_rounds = tuple(
        round_
        for round_ in dispatch_rounds
        if round_.batch_id is None or round_.batch_id not in accepted_batches
    )
    if missing_batch_rounds:
        raise CoordinationInvariantError("Accepted Batch is missing")
    ordered: list[tuple[CoordinationRound, AcceptedBatch]] = []
    for round_ in dispatch_rounds:
        if round_.batch_id is None or round_.batch_id not in accepted_batches:
            raise CoordinationInvariantError("Accepted Batch is missing")
        accepted = accepted_batches[round_.batch_id]
        if {outcome.task_id for outcome in accepted.outcomes} != {
            task.id for task in round_.tasks
        }:
            raise CoordinationInvariantError("Accepted Batch manifest is invalid")
        ordered.append((round_, accepted))
    return tuple(ordered)


def _active_batch_dump(batch: ActiveBatch) -> dict[str, Any]:
    """Serialize the scalar active-batch manifest for checkpoint state."""
    return batch.model_dump(mode="json")


def _active_batch_load(value: Mapping[str, object]) -> ActiveBatch:
    """Validate one scalar active-batch manifest from checkpoint state."""
    try:
        return ActiveBatch.model_validate(value)
    except ValidationError as error:
        raise TypeError("Agent active batch is invalid") from error


def _accepted_task_dump(task: AcceptedTask) -> dict[str, Any]:
    """Serialize one Send branch's trusted Task manifest entry."""
    return task.model_dump(mode="json")


def _accepted_task_load(value: Mapping[str, object]) -> AcceptedTask:
    """Validate one Send branch's trusted Task manifest entry."""
    try:
        return AcceptedTask.model_validate(value)
    except ValidationError as error:
        raise TypeError("Agent dispatched Task is invalid") from error
