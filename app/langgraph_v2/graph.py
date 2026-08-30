"""Production LangGraph Linear Core for the v2 query API."""

from __future__ import annotations

import unicodedata
from collections.abc import AsyncIterator, Hashable, Iterable
from typing import Annotated, Any, NotRequired, Protocol, TypedDict, cast
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.channels import UntrackedValue  # pyright: ignore[reportMissingTypeStubs]
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from langgraph.types import StateSnapshot

from app.langgraph_v2.answer import AnswerActor, run_answer
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.contracts import LinearQueryResponse, LiveStreamEvent
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.evidence import Evidence
from app.langgraph_v2.finalization import run_finalization
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


class LinearGraphState(TypedDict):
    """Typed state for the ingress-to-finalization Linear Graph."""

    query: str
    conversation_id: str
    turn_id: NotRequired[str]
    client_request_id: str | None
    history: NotRequired[list[ConversationTurn]]
    halted: NotRequired[bool]
    moderation: NotRequired[dict[str, Any] | None]
    refined_query: NotRequired[str | None]
    refinement_usage: NotRequired[dict[str, Any]]
    refinement_error: NotRequired[str | None]
    retrieval_error: NotRequired[str | None]
    reranking_error: NotRequired[str | None]
    evidence: NotRequired[Annotated[list[Evidence], UntrackedValue]]
    ranked_evidence: NotRequired[Annotated[list[Evidence], UntrackedValue]]
    answer: NotRequired[str | None]
    answer_usage: NotRequired[dict[str, Any]]
    citations: NotRequired[list[CitationReference]]
    answer_error: NotRequired[str | None]
    groundedness: NotRequired[GroundednessResult | None]
    groundedness_usage: NotRequired[dict[str, Any]]
    groundedness_error: NotRequired[str | None]
    post_moderation: NotRequired[dict[str, Any] | None]
    post_moderation_error: NotRequired[str | None]
    final_response: NotRequired[LinearQueryResponse | None]


class LinearGraphStateUpdate(TypedDict, total=False):
    """Partial state update returned by one Linear Graph node."""

    history: list[ConversationTurn]
    halted: bool
    moderation: dict[str, Any] | None
    refined_query: str | None
    refinement_usage: dict[str, Any]
    refinement_error: str | None
    retrieval_error: str | None
    reranking_error: str | None
    evidence: list[Evidence]
    ranked_evidence: list[Evidence]
    answer: str | None
    answer_usage: dict[str, Any]
    citations: list[CitationReference]
    answer_error: str | None
    groundedness: GroundednessResult | None
    groundedness_usage: dict[str, Any]
    groundedness_error: str | None
    post_moderation: dict[str, Any] | None
    post_moderation_error: str | None
    final_response: LinearQueryResponse | None


class LinearGraph(Protocol):
    """Typed application boundary around LangGraph's partially typed API."""

    async def ainvoke(
        self,
        graph_input: LinearGraphState | None,
        /,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> LinearGraphState:
        """Invoke a new graph turn or resume one from its checkpoint."""
        ...

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        """Read the current checkpoint state."""
        ...

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any],
        as_node: str,
    ) -> RunnableConfig:
        """Create a state checkpoint attributed to one node."""
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
    state: LinearGraphState,
    *,
    message_repository: ConversationMessageRepository | None = None,
    request_context: TrustedRequestContext | None = None,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    current_turn_id: UUID | None = None,
) -> LinearGraphStateUpdate:
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
        "halted": False,
        "moderation": None,
        "refined_query": None,
        "refinement_usage": {},
        "refinement_error": None,
        "retrieval_error": None,
        "reranking_error": None,
        "evidence": [],
        "ranked_evidence": [],
        "answer": None,
        "answer_usage": {},
        "citations": [],
        "answer_error": None,
        "groundedness": None,
        "groundedness_usage": {},
        "groundedness_error": None,
        "post_moderation": None,
        "post_moderation_error": None,
        "final_response": None,
    }


def _emit_events(events: Iterable[LiveStreamEvent]) -> None:
    """Write public envelopes directly to LangGraph's live custom stream."""
    writer = get_stream_writer()
    for event in events:
        writer(event.to_stream_payload())


def canonical_query(query: str) -> str:
    """Normalize query text without changing internal whitespace."""
    return unicodedata.normalize("NFC", query.replace("\r\n", "\n")).strip()


def _next_after_pre_moderation(state: LinearGraphState) -> str:
    """Stop the graph on a flagged query before any later phase."""
    return "end" if state.get("halted", False) else "question_refinement"


def _next_after_question_refinement(state: LinearGraphState) -> str:
    return "end" if state.get("halted", False) else "retrieval"


def _next_after_retrieval(state: LinearGraphState) -> str:
    return "end" if state.get("halted", False) else "reranking"


def build_linear_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    tenant_id: str | None = None,
    current_turn_id: UUID | None = None,
    message_repository: ConversationMessageRepository | None = None,
    request_context: TrustedRequestContext | None = None,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    output_assessment_audit: OutputAssessmentAudit | None = None,
    moderation_provider: ModerationProvider | None = None,
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    answer_actor: AnswerActor | None = None,
    groundedness_actor: GroundednessActor | None = None,
) -> LinearGraph:
    """Compile the deterministic ingress-to-finalization Linear Graph."""
    if message_repository is not None and request_context is None:
        raise ValueError("request_context is required with message_repository")
    if answer_actor is not None and tenant_id is None:
        raise ValueError("tenant_id is required with answer_actor")
    if (
        tenant_id is not None
        and request_context is not None
        and request_context.tenant_id != tenant_id
    ):
        raise ValueError("request_context tenant_id must match tenant_id")

    builder: StateGraph[LinearGraphState, None, LinearGraphState, LinearGraphState] = (
        StateGraph(LinearGraphState)
    )

    async def query_node(state: LinearGraphState) -> LinearGraphStateUpdate:
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

    async def pre_moderation_node(state: LinearGraphState) -> LinearGraphStateUpdate:
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

    async def question_refinement_node(
        state: LinearGraphState,
    ) -> LinearGraphStateUpdate:
        events, halted, result, error = await run_question_refinement(
            state, actor=selected_refinement_actor
        )
        _emit_events(events)
        update: LinearGraphStateUpdate = {
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

    async def retrieval_node(state: LinearGraphState) -> LinearGraphStateUpdate:
        events, evidence, halted, error = await run_retrieval(
            state,
            retriever=selected_retriever,
        )
        _emit_events(events)
        update: LinearGraphStateUpdate = {
            "halted": halted,
            "evidence": evidence,
        }
        if error is not None:
            update["retrieval_error"] = error
        return update

    builder.add_node(  # pyright: ignore[reportUnknownMemberType]
        "retrieval", retrieval_node
    )

    async def reranking_node(state: LinearGraphState) -> LinearGraphStateUpdate:
        events, evidence, halted, error = await run_reranking(
            state,
            ranker=selected_ranker,
        )
        _emit_events(events)
        update: LinearGraphStateUpdate = {
            "halted": halted,
            "ranked_evidence": evidence,
        }
        if error is not None:
            update["reranking_error"] = error
        return update

    builder.add_node(  # pyright: ignore[reportUnknownMemberType]
        "reranking", reranking_node
    )

    answer_enabled = answer_actor is not None
    if answer_enabled:

        async def answer_node(state: LinearGraphState) -> LinearGraphStateUpdate:
            assert answer_actor is not None
            _, result, halted, error = await run_answer(
                state,
                actor=answer_actor,
                stream_writer=get_stream_writer(),
            )
            update: LinearGraphStateUpdate = {
                "halted": halted,
            }
            if result is not None:
                update["answer"] = result.answer
                update["answer_usage"] = result.usage
                update["citations"] = result.citations
            if error is not None:
                update["answer_error"] = error
            return update

        builder.add_node(  # pyright: ignore[reportUnknownMemberType]
            "answer", answer_node
        )

        if groundedness_actor is not None:

            async def groundedness_node(
                state: LinearGraphState,
            ) -> LinearGraphStateUpdate:
                assert tenant_id is not None
                events, result, usage, error = await run_groundedness(
                    state,
                    tenant_id=tenant_id,
                    current_turn_id=current_turn_id,
                    output_assessment_audit=output_assessment_audit,
                    actor=groundedness_actor,
                )
                _emit_events(events)
                return {
                    "groundedness": result,
                    "groundedness_usage": usage,
                    "groundedness_error": error,
                }

            builder.add_node(  # pyright: ignore[reportUnknownMemberType]
                "groundedness", groundedness_node
            )

        async def post_moderation_node(
            state: LinearGraphState,
        ) -> LinearGraphStateUpdate:
            assert tenant_id is not None
            events, decision, error = await run_post_moderation(
                state,
                tenant_id=tenant_id,
                current_turn_id=current_turn_id,
                output_assessment_audit=output_assessment_audit,
                provider=selected_moderation_provider,
            )
            _emit_events(events)
            update: LinearGraphStateUpdate = {}
            if decision is not None:
                update["post_moderation"] = decision.model_dump(exclude_none=True)
            if error is not None:
                update["post_moderation_error"] = error
            return update

        builder.add_node(  # pyright: ignore[reportUnknownMemberType]
            "post_moderation", post_moderation_node
        )

    async def finalization_node(state: LinearGraphState) -> LinearGraphStateUpdate:
        events, response = await run_finalization(state)
        if (
            message_repository is not None
            and request_context is not None
            and current_turn_id is not None
            and response.answer is not None
        ):
            await message_repository.persist_assistant_message(
                context=request_context,
                conversation_id=state["conversation_id"],
                turn_id=current_turn_id,
                content=response.answer,
                idempotency_key=f"turn:{current_turn_id}:assistant",
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

    def next_after_reranking(state: LinearGraphState) -> str:
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

        def next_after_answer(state: LinearGraphState) -> str:
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
        LinearGraph,
        builder.compile(  # pyright: ignore[reportUnknownMemberType]
            checkpointer=checkpointer
        ),
    )


linear_graph = build_linear_graph()
