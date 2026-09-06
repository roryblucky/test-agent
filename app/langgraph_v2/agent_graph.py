"""Clarification-first Agent Graph behind the shared v2 request lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
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


class AgentGraphState(TypedDict):
    """Persisted state currently needed by the clarification path only."""

    query: str
    conversation_id: str
    request_id: str
    conversation_messages: NotRequired[Annotated[list[BaseMessage], add_messages]]
    halted: NotRequired[bool]
    standalone_query: NotRequired[str | None]
    clarification: NotRequired[dict[str, Any] | None]
    answer: NotRequired[str | None]
    final_response: NotRequired[dict[str, Any] | None]


class AgentGraphStateUpdate(TypedDict, total=False):
    """Partial Agent Graph update returned by one node."""

    conversation_messages: list[BaseMessage]
    halted: bool
    standalone_query: str | None
    clarification: dict[str, Any] | None
    answer: str | None
    final_response: dict[str, Any] | None


def _emit(events: Sequence[LiveStreamEvent]) -> None:
    writer = get_stream_writer()
    for event in events:
        writer(event.to_stream_payload())


def _clarification_answer(clarification: QueryUnderstandingClarification) -> str:
    questions = clarification.questions
    if not 1 <= len(questions) <= 3:
        raise ValueError("Query Understanding clarification requires one to three questions")
    if any(not question.question.strip() for question in questions):
        raise ValueError("Query Understanding clarification questions must not be blank")
    if any(len(question.options) > 4 for question in questions):
        raise ValueError("Query Understanding clarification options exceed the limit")
    return questions[0].question


def build_agent_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    query_understanding_actor: QueryUnderstandingActor,
    moderation_provider: ModerationProvider,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    checkpoint_state_adapter: AgentCheckpointStateAdapter | None = None,
) -> RequestOwnedGraph:
    """Compile the first Agent path: clarification or a closed future seam."""
    state_adapter = checkpoint_state_adapter or AgentCheckpointStateAdapter()
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
            "clarification": None,
            "answer": None,
            "final_response": None,
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
        if clarification is None:
            raise RuntimeError("Agent research execution is not available")
        return {
            "standalone_query": result.resolved_query.standalone_query,
            "clarification": clarification.model_dump(mode="json"),
            "answer": _clarification_answer(clarification),
        }

    async def finalize_state(state: AgentGraphState) -> AgentGraphStateUpdate:
        clarification_value = state.get("clarification")
        answer = state.get("answer")
        if not isinstance(clarification_value, dict):
            raise TypeError("Agent clarification is invalid")
        if not isinstance(answer, str):
            raise TypeError("Agent clarification answer is invalid")
        clarification = QueryUnderstandingClarification.model_validate(
            clarification_value
        )
        response = V2QueryResponse(
            query=state["query"],
            answer=answer,
            clarification=clarification,
            conversation_id=state["conversation_id"],
            metadata={
                "steps_executed": [
                    "initializer",
                    "pre_moderation",
                    "query_understanding",
                    "finalize_state",
                ]
            },
            citations=[],
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

    builder.add_node("initializer", initializer)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("pre_moderation", pre_moderation)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("query_understanding", query_understanding)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("finalize_state", finalize_state)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("publish", publish)  # pyright: ignore[reportUnknownMemberType]
    builder.add_edge(START, "initializer")
    builder.add_edge("initializer", "pre_moderation")
    builder.add_conditional_edges(
        "pre_moderation",
        next_after_pre_moderation,
        {"query_understanding": "query_understanding", "end": END},
    )
    builder.add_edge("query_understanding", "finalize_state")
    builder.add_edge("finalize_state", "publish")
    builder.add_edge("publish", END)
    return cast(
        RequestOwnedGraph,
        builder.compile(  # pyright: ignore[reportUnknownMemberType]
            checkpointer=checkpointer
        ),
    )
