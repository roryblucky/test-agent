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
from pydantic import BaseModel, ConfigDict

from app.langgraph_v2.agent_completion import (
    IncompleteResearch,
    insufficient_evidence_answer,
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


class CoordinatorActor(Protocol):
    """Propose one bounded Coordinator decision."""

    async def decide(self, input: CoordinatorInput) -> Finish:
        """Return the typed Coordinator decision."""
        ...


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
    clarification: NotRequired[dict[str, Any] | None]
    answer: NotRequired[str | None]
    completion_status: NotRequired[str | None]
    termination_reason: NotRequired[str | None]
    final_response: NotRequired[dict[str, Any] | None]


class AgentGraphStateUpdate(TypedDict, total=False):
    """Partial Agent Graph update returned by one node."""

    conversation_messages: list[BaseMessage]
    halted: bool
    standalone_query: str | None
    intent: dict[str, Any] | None
    research_scope: dict[str, Any] | None
    clarification: dict[str, Any] | None
    answer: str | None
    completion_status: str | None
    termination_reason: str | None
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
    coordinator_actor: CoordinatorActor,
    intent_policies: Mapping[str, AgentIntentPolicy],
    moderation_provider: ModerationProvider,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    checkpoint_state_adapter: AgentCheckpointStateAdapter | None = None,
) -> RequestOwnedGraph:
    """Compile clarification plus first legal Coordinator Finish path."""
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
            "intent": None,
            "research_scope": None,
            "clarification": None,
            "answer": None,
            "completion_status": None,
            "termination_reason": None,
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
        decision = await coordinator_actor.decide(
            CoordinatorInput(
                standalone_query=standalone_query,
                intent=scope.intent,
                specialist_descriptors=scope.specialist_descriptors,
            )
        )
        Finish.model_validate(decision)
        _emit(
            (
                LiveStreamEvent(type="step_start", step="coordinator"),
                LiveStreamEvent(type="step_completed", step="coordinator"),
            )
        )
        return {}

    async def research_completion(state: AgentGraphState) -> AgentGraphStateUpdate:
        del state
        completion = IncompleteResearch(insufficient_evidence=True)
        return {
            "answer": insufficient_evidence_answer(completion),
            "completion_status": "incomplete",
            "termination_reason": "insufficient_evidence",
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
            if completion_status != "incomplete" or termination_reason != "insufficient_evidence":
                raise TypeError("Agent research completion is invalid")
            metadata["steps_executed"].extend(
                ["resolve_scope", "coordinator", "research_completion"]
            )
            metadata["completion_status"] = completion_status
            metadata["termination_reason"] = termination_reason
        metadata["steps_executed"].append("finalize_state")
        response = V2QueryResponse(
            query=state["query"],
            answer=answer,
            clarification=clarification,
            conversation_id=state["conversation_id"],
            metadata=metadata,
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

    def next_after_query_understanding(state: AgentGraphState) -> str:
        return "finalize_state" if state.get("clarification") is not None else "resolve_scope"

    builder.add_node("initializer", initializer)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("pre_moderation", pre_moderation)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("query_understanding", query_understanding)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("resolve_scope", resolve_scope)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("coordinator", coordinator)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("research_completion", research_completion)  # pyright: ignore[reportUnknownMemberType]
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
    builder.add_edge("coordinator", "research_completion")
    builder.add_edge("research_completion", "finalize_state")
    builder.add_edge("finalize_state", "publish")
    builder.add_edge("publish", END)
    return cast(
        RequestOwnedGraph,
        builder.compile(  # pyright: ignore[reportUnknownMemberType]
            checkpointer=checkpointer
        ),
    )
