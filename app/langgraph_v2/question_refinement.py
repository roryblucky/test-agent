"""Structured question-refinement actor used by the v2 linear graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.langgraph_v2.history import ConversationTurn, to_model_message_history
from app.langgraph_v2.run_events import EventInput
from app.models.workflow import ResolvedQuery

REFINEMENT_ERROR_MESSAGE = "Question refinement failed."


class V2ResolvedQuery(ResolvedQuery):
    """V2 refinement output requiring a non-empty standalone question."""

    standalone_query: str = Field(min_length=1)


class QuestionRefinementResult(BaseModel):
    """Validated refinement plus usage returned through the actor port."""

    resolved_query: V2ResolvedQuery
    usage: dict[str, Any] = Field(default_factory=dict)


class QuestionRefinementActor(Protocol):
    """PydanticAI-backed seam for producing structured standalone questions."""

    async def refine(
        self, query: str, history: Sequence[ConversationTurn]
    ) -> QuestionRefinementResult:
        """Return a validated standalone question and its model usage."""
        ...


class MockQuestionRefinementActor:
    """Deterministic actor used when no tenant model registry is configured."""

    async def refine(
        self, query: str, history: Sequence[ConversationTurn] = ()
    ) -> QuestionRefinementResult:
        """Keep the query unchanged while satisfying the structured contract."""
        del history
        return QuestionRefinementResult(
            resolved_query=V2ResolvedQuery(original_query=query, standalone_query=query)
        )


class PydanticAIQuestionRefinementActor:
    """Adapt a PydanticAI Agent to the v2 actor protocol."""

    def __init__(self, agent: Agent[Any, V2ResolvedQuery]) -> None:
        self._agent = agent

    async def refine(
        self, query: str, history: Sequence[ConversationTurn] = ()
    ) -> QuestionRefinementResult:
        """Run the agent and return its validated structured output."""
        if history:
            result = await self._agent.run(
                query,
                message_history=to_model_message_history(history),
            )
        else:
            result = await self._agent.run(query)
        usage_method = getattr(result, "usage", None)
        usage_payload: dict[str, Any] = {}
        if callable(usage_method):
            usage = usage_method()
            usage_payload = asdict(usage) if is_dataclass(usage) else dict(vars(usage))
        return QuestionRefinementResult(
            resolved_query=V2ResolvedQuery.model_validate(result.output.model_dump()),
            usage=usage_payload,
        )


def build_question_refinement_actor(
    registry: Any,
    *,
    model_name: str = "fast",
    instructions: str | None = None,
) -> PydanticAIQuestionRefinementActor:
    """Create a role-configured PydanticAI refinement actor."""
    agent = registry.create_agent(
        model_name,
        output_type=V2ResolvedQuery,
        instructions=instructions or "Return a standalone question as structured JSON.",
    )
    return PydanticAIQuestionRefinementActor(agent)


def refinement_events(
    result: ResolvedQuery | None = None,
    *,
    error: str | None = None,
) -> tuple[EventInput, ...]:
    """Build stable Events for a successful or failed refinement."""
    events = [
        EventInput(
            event_key="phase:question_refinement:step_start:1",
            type="step_start",
            step="llm:refine_question",
        )
    ]
    if error is not None:
        events.append(
            EventInput(
                event_key="phase:question_refinement:error:1",
                type="error",
                data=error,
            )
        )
    else:
        assert result is not None
        events.append(
            EventInput(
                event_key="phase:question_refinement:step_completed:1",
                type="step_completed",
                step="llm:refine_question",
                data={"refined_query": result.standalone_query},
            )
        )
    return tuple(events)


async def run_question_refinement(
    state: Mapping[str, Any],
    *,
    actor: QuestionRefinementActor,
) -> tuple[tuple[EventInput, ...], bool, QuestionRefinementResult | None, str | None]:
    """Return refinement State without reading or writing an application journal."""
    try:
        history = [
            ConversationTurn.model_validate(turn) for turn in state.get("history", [])
        ]
        result = await actor.refine(state["query"], history)
        result = QuestionRefinementResult.model_validate(result)
    except Exception as exc:
        message = str(exc) or REFINEMENT_ERROR_MESSAGE
        return refinement_events(error=message), True, None, message
    return refinement_events(result.resolved_query), False, result, None
