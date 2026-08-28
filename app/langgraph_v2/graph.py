"""Minimal typed LangGraph used by the v2 tracer."""

from __future__ import annotations

import unicodedata
from typing import Any, NotRequired, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph_v2.answer import AnswerActor, AnswerCancelled, run_answer
from app.langgraph_v2.artifacts import ArtifactRef
from app.langgraph_v2.contracts import TracerQueryResponse, TracerStreamEvent
from app.langgraph_v2.finalization import finalize_in_memory, run_finalization
from app.langgraph_v2.groundedness import GroundednessActor, run_groundedness
from app.langgraph_v2.history import ConversationTurn, select_sliding_window_history
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultInput
from app.langgraph_v2.post_moderation import run_post_moderation
from app.langgraph_v2.pre_moderation import (
    MockModerationProvider,
    ModerationDecision,
    ModerationProvider,
    pre_moderation_events,
    run_pre_moderation,
)
from app.langgraph_v2.question_refinement import (
    MockQuestionRefinementActor,
    QuestionRefinementActor,
    refinement_events,
    run_question_refinement,
)
from app.langgraph_v2.reranking import MockRanker, Ranker, run_reranking
from app.langgraph_v2.retrieval import MockRetriever, Retriever, run_retrieval
from app.langgraph_v2.run_events import EventInput, EventRecord
from app.models.domain import GroundednessResult
from app.models.workflow import CitationReference


class TracerState(TypedDict):
    """Typed state for the ingress-to-finalization tracer graph."""

    query: str
    conversation_id: str
    client_request_id: str | None
    events: list[dict[str, Any]]
    history: NotRequired[list[ConversationTurn]]
    halted: NotRequired[bool]
    moderation: NotRequired[dict[str, Any]]
    refined_query: NotRequired[str]
    refinement_error: NotRequired[str]
    retrieval_error: NotRequired[str]
    reranking_error: NotRequired[str]
    artifact_refs: NotRequired[list[ArtifactRef]]
    ranked_refs: NotRequired[list[ArtifactRef]]
    answer: NotRequired[str]
    answer_usage: NotRequired[dict[str, Any]]
    citations: NotRequired[list[CitationReference]]
    answer_error: NotRequired[str]
    groundedness: NotRequired[GroundednessResult]
    groundedness_error: NotRequired[str]
    post_moderation: NotRequired[dict[str, Any]]
    post_moderation_error: NotRequired[str]
    final_response: NotRequired[TracerQueryResponse]


class TracerStateUpdate(TypedDict, total=False):
    """Partial state update returned by one tracer node."""

    events: list[dict[str, Any]]
    history: list[ConversationTurn]
    halted: bool
    moderation: dict[str, Any]
    refined_query: str
    refinement_error: str
    retrieval_error: str
    reranking_error: str
    artifact_refs: list[ArtifactRef]
    ranked_refs: list[ArtifactRef]
    answer: str
    answer_usage: dict[str, Any]
    citations: list[CitationReference]
    answer_error: str
    groundedness: GroundednessResult
    groundedness_error: str
    post_moderation: dict[str, Any]
    post_moderation_error: str
    final_response: TracerQueryResponse


async def _query(
    state: TracerState,
    *,
    phase_context: PhaseExecutionContext | None = None,
) -> TracerStateUpdate:
    query = state["query"]
    canonical = canonical_query(query)

    async def invoke() -> PhaseResultInput:
        history: list[ConversationTurn] = []
        if phase_context is not None and phase_context.message_repository is not None:
            messages = await phase_context.message_repository.list_messages(
                phase_context.tenant_id,
                state["conversation_id"],
            )
            history = select_sliding_window_history(
                messages,
                token_budget=phase_context.history_token_budget,
                current_run_id=phase_context.run_id,
            )
        return PhaseResultInput(
            phase_name="query",
            normalized_result={
                "query": canonical,
                "history_snapshot": [turn.model_dump() for turn in history],
            },
            events=(
                EventInput(
                    event_key="phase:query:step_start:1",
                    type="step_start",
                    step="query",
                ),
                EventInput(
                    event_key="phase:query:step_completed:1",
                    type="step_completed",
                    step="query",
                    data={"query": canonical},
                ),
            ),
        )

    if phase_context is None:
        phase = await invoke()
        return {
            "events": [
                _event_state(event, index)
                for index, event in enumerate(phase.events, 1)
            ],
            "history": [
                ConversationTurn.model_validate(turn)
                for turn in phase.normalized_result["history_snapshot"]
            ],
        }
    result = await phase_context.repository.get_or_invoke(
        tenant_id=phase_context.tenant_id,
        run_id=phase_context.run_id,
        owner_instance_id=phase_context.owner_instance_id,
        execution_epoch=phase_context.execution_epoch,
        phase_name="query",
        invoke=invoke,
    )
    return {
        "events": [_event_state(event, event.sequence) for event in result.events],
        "history": [
            ConversationTurn.model_validate(turn)
            for turn in result.normalized_result["history_snapshot"]
        ],
    }


def _event_state(event: EventInput | EventRecord, sequence: int) -> dict[str, Any]:
    """Convert journal or in-memory event data into graph state."""
    return TracerStreamEvent(
        event_key=event.event_key,
        type=cast(Any, event.type),
        step=event.step,
        data=event.data,
        sequence=sequence,
    ).model_dump(exclude_none=True)


def canonical_query(query: str) -> str:
    """Normalize query text without changing internal whitespace."""
    return unicodedata.normalize("NFC", query.replace("\r\n", "\n")).strip()


async def _finalize(state: TracerState) -> TracerStateUpdate:
    return finalize_in_memory(state)


async def _pre_moderation_without_journal(
    state: TracerState,
    provider: ModerationProvider,
) -> tuple[list[dict[str, Any]], ModerationDecision]:
    """Run the provider for an unconfigured in-memory graph."""
    decision = await provider.check(state["query"])
    sequence_start = len(state["events"]) + 1
    events = [
        _event_state(event, sequence_start + index)
        for index, event in enumerate(pre_moderation_events(decision))
    ]
    return events, decision


def _next_after_pre_moderation(state: TracerState) -> str:
    """Stop the graph on a flagged query before any later phase."""
    return "end" if state.get("halted", False) else "question_refinement"


def build_tracer_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    phase_context: PhaseExecutionContext | None = None,
    moderation_provider: ModerationProvider | None = None,
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    answer_actor: AnswerActor | None = None,
    groundedness_actor: GroundednessActor | None = None,
) -> CompiledStateGraph:
    """Compile the deterministic ingress-to-finalization LangGraph."""
    builder = StateGraph(TracerState)

    async def query_node(state: TracerState) -> TracerStateUpdate:
        return await _query(state, phase_context=phase_context)

    builder.add_node("query", query_node)
    selected_moderation_provider = moderation_provider or MockModerationProvider()
    selected_refinement_actor = refinement_actor or MockQuestionRefinementActor()
    selected_retriever = retriever or MockRetriever()
    selected_ranker = ranker or MockRanker()
    selected_artifact_repository = (
        phase_context.artifact_repository if phase_context is not None else None
    )

    async def pre_moderation_node(state: TracerState) -> TracerStateUpdate:
        if phase_context is None:
            events, decision = await _pre_moderation_without_journal(
                state, selected_moderation_provider
            )
        else:
            events, halted, decision = await run_pre_moderation(
                state,
                context=phase_context,
                provider=selected_moderation_provider,
            )
            return {
                "events": [
                    *state["events"],
                    *[_event_state(event, event.sequence) for event in events],
                ],
                "halted": halted,
                "moderation": decision.model_dump(exclude_none=True),
            }
        return {
            "events": [*state["events"], *events],
            "halted": decision.is_flagged,
            "moderation": decision.model_dump(exclude_none=True),
        }

    builder.add_node("pre_moderation", pre_moderation_node)

    async def question_refinement_node(state: TracerState) -> TracerStateUpdate:
        if phase_context is None:
            result = await selected_refinement_actor.refine(
                state["query"], state.get("history", [])
            )
            return {
                "events": [
                    *state["events"],
                    *[
                        _event_state(event, len(state["events"]) + index)
                        for index, event in enumerate(refinement_events(result), 1)
                    ],
                ],
                "refined_query": result.standalone_query,
            }
        events, halted, result, error = await run_question_refinement(
            state,
            context=phase_context,
            actor=selected_refinement_actor,
        )
        update: TracerStateUpdate = {
            "events": [
                *state["events"],
                *[_event_state(event, event.sequence) for event in events],
            ],
            "halted": halted,
        }
        if result is not None:
            update["refined_query"] = result.standalone_query
        if error is not None:
            update["refinement_error"] = error
        return update

    builder.add_node("question_refinement", question_refinement_node)

    async def retrieval_node(state: TracerState) -> TracerStateUpdate:
        if phase_context is None or selected_artifact_repository is None:
            return {"artifact_refs": []}
        events, refs, _, halted, error = await run_retrieval(
            state,
            context=phase_context,
            artifacts=selected_artifact_repository,
            retriever=selected_retriever,
        )
        update: TracerStateUpdate = {
            "events": [
                *state["events"],
                *[_event_state(event, event.sequence) for event in events],
            ],
            "halted": halted,
            "artifact_refs": refs,
        }
        if error is not None:
            update["retrieval_error"] = error
        return update

    builder.add_node("retrieval", retrieval_node)

    async def reranking_node(state: TracerState) -> TracerStateUpdate:
        if phase_context is None or selected_artifact_repository is None:
            return {"ranked_refs": state.get("artifact_refs", [])}
        events, refs, halted, error = await run_reranking(
            state,
            context=phase_context,
            artifacts=selected_artifact_repository,
            ranker=selected_ranker,
        )
        update: TracerStateUpdate = {
            "events": [
                *state["events"],
                *[_event_state(event, event.sequence) for event in events],
            ],
            "halted": halted,
            "ranked_refs": refs,
        }
        if error is not None:
            update["reranking_error"] = error
        return update

    builder.add_node("reranking", reranking_node)

    if answer_actor is not None and selected_artifact_repository is not None:

        async def answer_node(state: TracerState) -> TracerStateUpdate:
            events, result, halted, error = await run_answer(
                state,
                context=phase_context,
                artifacts=selected_artifact_repository,
                actor=answer_actor,
            )
            update: TracerStateUpdate = {
                "events": [
                    *state["events"],
                    *[_event_state(event, event.sequence) for event in events],
                ],
                "halted": halted,
            }
            if result is not None:
                update["answer"] = result.answer
                update["answer_usage"] = result.usage
                update["citations"] = result.citations
            if error is not None:
                update["answer_error"] = error
            if (
                not halted
                and phase_context.cancellation_check is not None
                and await phase_context.cancellation_check()
            ):
                raise AnswerCancelled("answer publication cancelled at graph boundary")
            return update

        builder.add_node("answer", answer_node)

        if groundedness_actor is not None:

            async def groundedness_node(state: TracerState) -> TracerStateUpdate:
                events, result, halted, error = await run_groundedness(
                    state,
                    context=phase_context,
                    artifacts=selected_artifact_repository,
                    actor=groundedness_actor,
                )
                update: TracerStateUpdate = {
                    "events": [
                        *state["events"],
                        *[_event_state(event, event.sequence) for event in events],
                    ],
                    "halted": halted,
                }
                if result is not None:
                    update["groundedness"] = result
                if error is not None:
                    update["groundedness_error"] = error
                return update

            builder.add_node("groundedness", groundedness_node)

        async def post_moderation_node(state: TracerState) -> TracerStateUpdate:
            events, decision, safe_answer, halted, error = await run_post_moderation(
                state,
                context=phase_context,
                provider=selected_moderation_provider,
            )
            update: TracerStateUpdate = {
                "events": [
                    *state["events"],
                    *[_event_state(event, event.sequence) for event in events],
                ],
                "halted": halted,
            }
            if decision is not None:
                update["post_moderation"] = decision.model_dump(exclude_none=True)
            if safe_answer is not None:
                update["answer"] = safe_answer
            if error is not None:
                update["post_moderation_error"] = error
            return update

        builder.add_node("post_moderation", post_moderation_node)

    async def finalization_node(state: TracerState) -> TracerStateUpdate:
        if phase_context is None or selected_artifact_repository is None:
            return await _finalize(state)
        events, response = await run_finalization(
            state,
            context=phase_context,
            artifacts=selected_artifact_repository,
        )
        done_event = TracerStreamEvent(
            event_key="lifecycle:completed:0",
            type="done",
            data=response.model_dump(by_alias=True),
            sequence=len(state["events"]) + len(events) + 1,
        )
        return {
            "events": [
                *state["events"],
                *[_event_state(event, event.sequence) for event in events],
                done_event.model_dump(exclude_none=True),
            ],
            "final_response": response,
        }

    builder.add_node("finalization", finalization_node)
    builder.add_edge(START, "query")
    builder.add_edge("query", "pre_moderation")
    builder.add_conditional_edges(
        "pre_moderation",
        _next_after_pre_moderation,
        {"question_refinement": "question_refinement", "end": END},
    )
    builder.add_conditional_edges(
        "question_refinement",
        lambda state: "end" if state.get("halted", False) else "retrieval",
        {"retrieval": "retrieval", "end": END},
    )
    builder.add_conditional_edges(
        "retrieval",
        lambda state: "end" if state.get("halted", False) else "reranking",
        {"reranking": "reranking", "end": END},
    )
    reranking_routes = {"finalization": "finalization", "end": END}
    if answer_actor is not None and selected_artifact_repository is not None:
        reranking_routes["answer"] = "answer"
    builder.add_conditional_edges(
        "reranking",
        lambda state: (
            "end"
            if state.get("halted", False)
            else "answer"
            if answer_actor is not None and selected_artifact_repository is not None
            else "finalization"
        ),
        reranking_routes,
    )
    if answer_actor is not None and selected_artifact_repository is not None:
        answer_routes = {"finalization": "finalization", "end": END}
        if groundedness_actor is not None:
            answer_routes["groundedness"] = "groundedness"
        else:
            answer_routes["post_moderation"] = "post_moderation"
        builder.add_conditional_edges(
            "answer",
            lambda state: (
                "end"
                if state.get("halted", False)
                else "groundedness"
                if groundedness_actor is not None
                else "post_moderation"
            ),
            answer_routes,
        )
        if groundedness_actor is not None:
            builder.add_conditional_edges(
                "groundedness",
                lambda state: (
                    "end" if state.get("halted", False) else "post_moderation"
                ),
                {"post_moderation": "post_moderation", "end": END},
            )
        builder.add_conditional_edges(
            "post_moderation",
            lambda state: "end" if state.get("halted", False) else "finalization",
            {"finalization": "finalization", "end": END},
        )
    builder.add_edge("finalization", END)
    return builder.compile(checkpointer=checkpointer)


tracer_graph = build_tracer_graph()
