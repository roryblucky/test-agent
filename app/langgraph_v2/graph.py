"""Minimal typed LangGraph used by the v2 tracer."""

from __future__ import annotations

import unicodedata
from typing import Any, NotRequired, TypedDict, cast
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.langgraph_v2.answer import AnswerActor, AnswerCancelled, run_answer
from app.langgraph_v2.artifacts import ArtifactRef
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.contracts import (
    EventPersistence,
    TracerGraphEvent,
    TracerQueryResponse,
    TracerStreamEvent,
)
from app.langgraph_v2.finalization import finalize_in_memory, run_finalization
from app.langgraph_v2.groundedness import GroundednessActor, run_groundedness
from app.langgraph_v2.history import ConversationTurn, select_sliding_window_history
from app.langgraph_v2.phase_results import PhaseExecutionContext
from app.langgraph_v2.post_moderation import run_post_moderation
from app.langgraph_v2.pre_moderation import (
    MockModerationProvider,
    ModerationProvider,
    run_pre_moderation,
)
from app.langgraph_v2.question_refinement import (
    MockQuestionRefinementActor,
    QuestionRefinementActor,
    run_question_refinement,
)
from app.langgraph_v2.reranking import MockRanker, Ranker, run_reranking
from app.langgraph_v2.retrieval import MockRetriever, Retriever, run_retrieval
from app.langgraph_v2.run_events import CancellationObserved, EventInput, EventRecord
from app.models.domain import GroundednessResult
from app.models.workflow import CitationReference


class TracerState(TypedDict):
    """Typed state for the ingress-to-finalization tracer graph."""

    query: str
    conversation_id: str
    turn_id: NotRequired[str]
    client_request_id: str | None
    events: list[dict[str, Any]]
    history: NotRequired[list[ConversationTurn]]
    halted: NotRequired[bool]
    moderation: NotRequired[dict[str, Any]]
    refined_query: NotRequired[str]
    refinement_usage: NotRequired[dict[str, Any]]
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
    refinement_usage: dict[str, Any]
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

    history: list[ConversationTurn] = []
    if phase_context is not None and phase_context.message_repository is not None:
        messages = await phase_context.message_repository.list_messages(
            context=cast(TrustedRequestContext, phase_context.request_context),
            conversation_id=state["conversation_id"],
        )
        history = select_sliding_window_history(
            messages,
            token_budget=phase_context.history_token_budget,
            current_turn_id=phase_context.current_turn_id
            or (UUID(state["turn_id"]) if state.get("turn_id") is not None else None),
        )
    events = (
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
    )
    return {
        "events": [
            _event_state(event, index, persistence="none")
            for index, event in enumerate(events, 1)
        ],
        "history": history,
    }


def _event_state(
    event: EventInput | EventRecord,
    sequence: int,
    *,
    persistence: EventPersistence = "transport",
) -> dict[str, Any]:
    """Convert journal or in-memory event data into graph state."""
    state_event = TracerGraphEvent(
        event_key=event.event_key,
        type=cast(Any, event.type),
        step=event.step,
        data=event.data,
        sequence=sequence,
        persistence=persistence,
    )
    return state_event.model_dump(
        exclude_none=True,
        exclude={"persistence"} if persistence == "transport" else None,
    )


def canonical_query(query: str) -> str:
    """Normalize query text without changing internal whitespace."""
    return unicodedata.normalize("NFC", query.replace("\r\n", "\n")).strip()


async def _finalize(state: TracerState) -> TracerStateUpdate:
    return finalize_in_memory(state)


async def _check_cancellation(context: PhaseExecutionContext | None) -> None:
    """Stop before entering the next persistent graph node."""
    if (
        context is not None
        and context.cancellation_check is not None
        and await context.cancellation_check()
    ):
        raise CancellationObserved("cancellation observed at graph boundary")


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
        await _check_cancellation(phase_context)
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
        await _check_cancellation(phase_context)
        events, halted, decision = await run_pre_moderation(
            state, provider=selected_moderation_provider
        )
        sequence_start = len(state["events"])
        return {
            "events": [
                *state["events"],
                *[
                    _event_state(event, sequence_start + index, persistence="none")
                    for index, event in enumerate(events, 1)
                ],
            ],
            "halted": halted,
            "moderation": decision.model_dump(exclude_none=True),
        }

    builder.add_node("pre_moderation", pre_moderation_node)

    async def question_refinement_node(state: TracerState) -> TracerStateUpdate:
        await _check_cancellation(phase_context)
        events, halted, result, error = await run_question_refinement(
            state, actor=selected_refinement_actor
        )
        update: TracerStateUpdate = {
            "events": [
                *state["events"],
                *[
                    _event_state(
                        event,
                        len(state["events"]) + index,
                        persistence="none",
                    )
                    for index, event in enumerate(events, 1)
                ],
            ],
            "halted": halted,
        }
        if result is not None:
            update["refined_query"] = result.resolved_query.standalone_query
            if result.usage:
                update["refinement_usage"] = result.usage
        if error is not None:
            update["refinement_error"] = error
        return update

    builder.add_node("question_refinement", question_refinement_node)

    async def retrieval_node(state: TracerState) -> TracerStateUpdate:
        await _check_cancellation(phase_context)
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
        await _check_cancellation(phase_context)
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
            await _check_cancellation(phase_context)
            events, result, halted, error = await run_answer(
                state,
                context=phase_context,
                artifacts=selected_artifact_repository,
                actor=answer_actor,
                stream_writer=get_stream_writer(),
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
                await _check_cancellation(phase_context)
                events, result, error = await run_groundedness(
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
                }
                if result is not None:
                    update["groundedness"] = result
                if error is not None:
                    update["groundedness_error"] = error
                return update

            builder.add_node("groundedness", groundedness_node)

        async def post_moderation_node(state: TracerState) -> TracerStateUpdate:
            await _check_cancellation(phase_context)
            events, decision, error = await run_post_moderation(
                state,
                context=phase_context,
                provider=selected_moderation_provider,
            )
            update: TracerStateUpdate = {
                "events": [
                    *state["events"],
                    *[_event_state(event, event.sequence) for event in events],
                ],
            }
            if decision is not None:
                update["post_moderation"] = decision.model_dump(exclude_none=True)
            if error is not None:
                update["post_moderation_error"] = error
            return update

        builder.add_node("post_moderation", post_moderation_node)

    async def finalization_node(state: TracerState) -> TracerStateUpdate:
        await _check_cancellation(phase_context)
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
            builder.add_edge("groundedness", "post_moderation")
        builder.add_edge("post_moderation", "finalization")
    builder.add_edge("finalization", END)
    return builder.compile(checkpointer=checkpointer)


tracer_graph = build_tracer_graph()
