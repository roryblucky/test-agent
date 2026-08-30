"""Minimal typed LangGraph used by the v2 tracer."""

from __future__ import annotations

import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable, Iterable
from typing import Any, NotRequired, Protocol, TypedDict, cast
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from langgraph.types import StateSnapshot

from app.langgraph_v2.answer import (
    AnswerActor,
    AnswerCancelled,
    CancellationObserved,
    run_answer,
)
from app.langgraph_v2.artifacts import ArtifactRef, ArtifactScope, ArtifactStore
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.contracts import LiveStreamEvent, TracerQueryResponse
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.finalization import finalize_in_memory, run_finalization
from app.langgraph_v2.groundedness import GroundednessActor, run_groundedness
from app.langgraph_v2.history import (
    DEFAULT_HISTORY_TOKEN_BUDGET,
    ConversationTurn,
    select_sliding_window_history,
)
from app.langgraph_v2.output_assessments import OutputAssessmentAudit
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
from app.models.domain import GroundednessResult
from app.models.workflow import CitationReference


class TracerState(TypedDict):
    """Typed state for the ingress-to-finalization tracer graph."""

    query: str
    conversation_id: str
    turn_id: NotRequired[str]
    client_request_id: str | None
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
    groundedness_usage: NotRequired[dict[str, Any]]
    groundedness_error: NotRequired[str]
    post_moderation: NotRequired[dict[str, Any]]
    post_moderation_error: NotRequired[str]
    final_response: NotRequired[TracerQueryResponse]


class TracerStateUpdate(TypedDict, total=False):
    """Partial state update returned by one tracer node."""

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
    groundedness_usage: dict[str, Any]
    groundedness_error: str
    post_moderation: dict[str, Any]
    post_moderation_error: str
    final_response: TracerQueryResponse


class TracerGraph(Protocol):
    """Typed application boundary around LangGraph's partially typed API."""

    async def ainvoke(
        self,
        graph_input: TracerState | None,
        /,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> TracerState:
        """Invoke a new graph turn or resume one from its checkpoint."""
        ...

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        """Read the current checkpoint state."""
        ...

    def astream(
        self,
        graph_input: Any | None,
        /,
        *,
        config: RunnableConfig | None = None,
        stream_mode: list[str] | str | None = None,
        durability: str | None = None,
    ) -> AsyncIterator[Any]:
        """Stream graph output using the requested LangGraph mode."""
        ...


async def _query(
    state: TracerState,
    *,
    message_repository: ConversationMessageRepository | None = None,
    request_context: TrustedRequestContext | None = None,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    current_turn_id: UUID | None = None,
) -> TracerStateUpdate:
    query = state["query"]
    canonical = canonical_query(query)

    history: list[ConversationTurn] = []
    if message_repository is not None and request_context is not None:
        messages = await message_repository.list_messages(
            context=request_context,
            conversation_id=state["conversation_id"],
        )
        history = select_sliding_window_history(
            messages,
            token_budget=history_token_budget,
            current_turn_id=current_turn_id
            or (
                UUID(turn_id) if (turn_id := state.get("turn_id")) is not None else None
            ),
        )
    _emit_events(
        (
            LiveStreamEvent(
                type="step_start",
                step="query",
            ),
            LiveStreamEvent(
                type="step_completed",
                step="query",
                data={"query": canonical},
            ),
        )
    )
    return {
        "history": history,
    }


def _emit_events(events: Iterable[LiveStreamEvent]) -> None:
    """Write public envelopes directly to LangGraph's live custom stream."""
    writer = get_stream_writer()
    for event in events:
        writer(event.to_stream_payload())


def canonical_query(query: str) -> str:
    """Normalize query text without changing internal whitespace."""
    return unicodedata.normalize("NFC", query.replace("\r\n", "\n")).strip()


async def _finalize(state: TracerState) -> TracerStateUpdate:
    events, response = finalize_in_memory(state)
    _emit_events(events)
    _emit_events(
        (
            LiveStreamEvent(
                type="done",
                data=response.model_dump(by_alias=True),
                checkpoint_terminal=True,
            ),
        )
    )
    return {"final_response": response}


async def _check_cancellation(
    cancellation_check: Callable[[], Awaitable[bool]] | None,
) -> None:
    """Stop before entering the next persistent graph node."""
    if cancellation_check is not None and await cancellation_check():
        raise CancellationObserved("cancellation observed at graph boundary")


def _next_after_pre_moderation(state: TracerState) -> str:
    """Stop the graph on a flagged query before any later phase."""
    return "end" if state.get("halted", False) else "question_refinement"


def _next_after_question_refinement(state: TracerState) -> str:
    return "end" if state.get("halted", False) else "retrieval"


def _next_after_retrieval(state: TracerState) -> str:
    return "end" if state.get("halted", False) else "reranking"


def build_tracer_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    tenant_id: str | None = None,
    current_turn_id: UUID | None = None,
    artifact_repository: ArtifactStore | None = None,
    message_repository: ConversationMessageRepository | None = None,
    request_context: TrustedRequestContext | None = None,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    cancellation_check: Callable[[], Awaitable[bool]] | None = None,
    output_assessment_audit: OutputAssessmentAudit | None = None,
    moderation_provider: ModerationProvider | None = None,
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    answer_actor: AnswerActor | None = None,
    groundedness_actor: GroundednessActor | None = None,
) -> TracerGraph:
    """Compile the deterministic ingress-to-finalization LangGraph."""
    if message_repository is not None and request_context is None:
        raise ValueError("request_context is required with message_repository")
    if (
        tenant_id is not None
        and request_context is not None
        and request_context.tenant_id != tenant_id
    ):
        raise ValueError("request_context tenant_id must match tenant_id")

    builder: StateGraph[TracerState, None, TracerState, TracerState] = StateGraph(
        TracerState
    )

    async def query_node(state: TracerState) -> TracerStateUpdate:
        await _check_cancellation(cancellation_check)
        return await _query(
            state,
            message_repository=message_repository,
            request_context=request_context,
            history_token_budget=history_token_budget,
            current_turn_id=current_turn_id,
        )

    builder.add_node("query", query_node)  # pyright: ignore[reportUnknownMemberType]
    selected_moderation_provider = moderation_provider or MockModerationProvider()
    selected_refinement_actor = refinement_actor or MockQuestionRefinementActor()
    selected_retriever = retriever or MockRetriever()
    selected_ranker = ranker or MockRanker()
    selected_artifact_repository = artifact_repository

    def artifact_scope(state: TracerState) -> ArtifactScope | None:
        if request_context is None:
            return None
        raw_turn_id = current_turn_id or state.get("turn_id")
        if raw_turn_id is None:
            return None
        try:
            turn_id = UUID(str(raw_turn_id))
        except ValueError:
            return None
        return ArtifactScope(
            context=request_context,
            conversation_id=state["conversation_id"],
            turn_id=turn_id,
        )

    async def pre_moderation_node(state: TracerState) -> TracerStateUpdate:
        await _check_cancellation(cancellation_check)
        events, halted, decision = await run_pre_moderation(
            state, provider=selected_moderation_provider
        )
        _emit_events(events)
        return {
            "halted": halted,
            "moderation": decision.model_dump(exclude_none=True),
        }

    builder.add_node(  # pyright: ignore[reportUnknownMemberType]
        "pre_moderation", pre_moderation_node
    )

    async def question_refinement_node(state: TracerState) -> TracerStateUpdate:
        await _check_cancellation(cancellation_check)
        events, halted, result, error = await run_question_refinement(
            state, actor=selected_refinement_actor
        )
        _emit_events(events)
        update: TracerStateUpdate = {
            "halted": halted,
        }
        if result is not None:
            update["refined_query"] = result.resolved_query.standalone_query
            if result.usage:
                update["refinement_usage"] = result.usage
        if error is not None:
            update["refinement_error"] = error
        return update

    builder.add_node(  # pyright: ignore[reportUnknownMemberType]
        "question_refinement", question_refinement_node
    )

    async def retrieval_node(state: TracerState) -> TracerStateUpdate:
        await _check_cancellation(cancellation_check)
        scope = artifact_scope(state)
        if scope is None or selected_artifact_repository is None:
            return {"artifact_refs": []}
        events, refs, _, halted, error = await run_retrieval(
            state,
            scope=scope,
            artifacts=selected_artifact_repository,
            retriever=selected_retriever,
        )
        _emit_events(events)
        update: TracerStateUpdate = {
            "halted": halted,
            "artifact_refs": refs,
        }
        if error is not None:
            update["retrieval_error"] = error
        return update

    builder.add_node(  # pyright: ignore[reportUnknownMemberType]
        "retrieval", retrieval_node
    )

    async def reranking_node(state: TracerState) -> TracerStateUpdate:
        await _check_cancellation(cancellation_check)
        scope = artifact_scope(state)
        if scope is None or selected_artifact_repository is None:
            return {"ranked_refs": state.get("artifact_refs", [])}
        events, refs, halted, error = await run_reranking(
            state,
            scope=scope,
            artifacts=selected_artifact_repository,
            ranker=selected_ranker,
        )
        _emit_events(events)
        update: TracerStateUpdate = {
            "halted": halted,
            "ranked_refs": refs,
        }
        if error is not None:
            update["reranking_error"] = error
        return update

    builder.add_node(  # pyright: ignore[reportUnknownMemberType]
        "reranking", reranking_node
    )

    answer_enabled = (
        answer_actor is not None
        and selected_artifact_repository is not None
        and request_context is not None
    )
    if answer_enabled:

        async def answer_node(state: TracerState) -> TracerStateUpdate:
            assert tenant_id is not None
            assert selected_artifact_repository is not None
            assert answer_actor is not None
            scope = artifact_scope(state)
            assert scope is not None
            await _check_cancellation(cancellation_check)
            _, result, halted, error = await run_answer(
                state,
                scope=scope,
                cancellation_check=cancellation_check,
                artifacts=selected_artifact_repository,
                actor=answer_actor,
                stream_writer=get_stream_writer(),
            )
            update: TracerStateUpdate = {
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
                and cancellation_check is not None
                and await cancellation_check()
            ):
                raise AnswerCancelled("answer publication cancelled at graph boundary")
            return update

        builder.add_node(  # pyright: ignore[reportUnknownMemberType]
            "answer", answer_node
        )

        if groundedness_actor is not None:

            async def groundedness_node(state: TracerState) -> TracerStateUpdate:
                assert tenant_id is not None
                assert selected_artifact_repository is not None
                scope = artifact_scope(state)
                assert scope is not None
                await _check_cancellation(cancellation_check)
                events, result, usage, error = await run_groundedness(
                    state,
                    scope=scope,
                    current_turn_id=current_turn_id,
                    output_assessment_audit=output_assessment_audit,
                    artifacts=selected_artifact_repository,
                    actor=groundedness_actor,
                )
                _emit_events(events)
                update: TracerStateUpdate = {}
                if result is not None:
                    update["groundedness"] = result
                if usage:
                    update["groundedness_usage"] = usage
                if error is not None:
                    update["groundedness_error"] = error
                return update

            builder.add_node(  # pyright: ignore[reportUnknownMemberType]
                "groundedness", groundedness_node
            )

        async def post_moderation_node(state: TracerState) -> TracerStateUpdate:
            assert tenant_id is not None
            await _check_cancellation(cancellation_check)
            events, decision, error = await run_post_moderation(
                state,
                tenant_id=tenant_id,
                current_turn_id=current_turn_id,
                output_assessment_audit=output_assessment_audit,
                provider=selected_moderation_provider,
            )
            _emit_events(events)
            update: TracerStateUpdate = {}
            if decision is not None:
                update["post_moderation"] = decision.model_dump(exclude_none=True)
            if error is not None:
                update["post_moderation_error"] = error
            return update

        builder.add_node(  # pyright: ignore[reportUnknownMemberType]
            "post_moderation", post_moderation_node
        )

    async def finalization_node(state: TracerState) -> TracerStateUpdate:
        await _check_cancellation(cancellation_check)
        scope = artifact_scope(state)
        if scope is None or selected_artifact_repository is None:
            return await _finalize(state)
        events, response = await run_finalization(
            state,
            scope=scope,
            artifacts=selected_artifact_repository,
        )
        _emit_events(events)
        _emit_events(
            (
                LiveStreamEvent(
                    type="done",
                    data=response.model_dump(by_alias=True),
                    checkpoint_terminal=True,
                ),
            )
        )
        return {
            "final_response": response,
        }

    builder.add_node(  # pyright: ignore[reportUnknownMemberType]
        "finalization", finalization_node
    )
    builder.add_edge(START, "query")
    builder.add_edge("query", "pre_moderation")
    builder.add_conditional_edges(
        "pre_moderation",
        _next_after_pre_moderation,
        {"question_refinement": "question_refinement", "end": END},
    )
    builder.add_conditional_edges(
        "question_refinement",
        _next_after_question_refinement,
        {"retrieval": "retrieval", "end": END},
    )
    builder.add_conditional_edges(
        "retrieval",
        _next_after_retrieval,
        {"reranking": "reranking", "end": END},
    )
    reranking_routes: dict[Hashable, str] = {
        "finalization": "finalization",
        "end": END,
    }
    if answer_enabled:
        reranking_routes["answer"] = "answer"

    def next_after_reranking(state: TracerState) -> str:
        if state.get("halted", False):
            return "end"
        if answer_enabled:
            return "answer"
        return "finalization"

    builder.add_conditional_edges(
        "reranking",
        next_after_reranking,
        reranking_routes,
    )
    if answer_enabled:
        answer_routes: dict[Hashable, str] = {
            "finalization": "finalization",
            "end": END,
        }
        if groundedness_actor is not None:
            answer_routes["groundedness"] = "groundedness"
        else:
            answer_routes["post_moderation"] = "post_moderation"

        def next_after_answer(state: TracerState) -> str:
            if state.get("halted", False):
                return "end"
            return (
                "groundedness" if groundedness_actor is not None else "post_moderation"
            )

        builder.add_conditional_edges(
            "answer",
            next_after_answer,
            answer_routes,
        )
        if groundedness_actor is not None:
            builder.add_edge("groundedness", "post_moderation")
        builder.add_edge("post_moderation", "finalization")
    builder.add_edge("finalization", END)
    return cast(
        TracerGraph,
        builder.compile(  # pyright: ignore[reportUnknownMemberType]
            checkpointer=checkpointer
        ),
    )


tracer_graph = build_tracer_graph()
