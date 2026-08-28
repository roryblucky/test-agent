"""Structured question-refinement actor used by the v2 linear graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import Field
from pydantic_ai import Agent

from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultInput
from app.langgraph_v2.run_events import EventInput, EventRecord
from app.models.workflow import ResolvedQuery

REFINEMENT_ERROR_MESSAGE = "Question refinement failed."


class V2ResolvedQuery(ResolvedQuery):
    """V2 refinement output requiring a non-empty standalone question."""

    standalone_query: str = Field(min_length=1)


class QuestionRefinementActor(Protocol):
    """PydanticAI-backed seam for producing structured standalone questions."""

    async def refine(self, query: str) -> ResolvedQuery:
        """Return a validated standalone question."""
        ...


class MockQuestionRefinementActor:
    """Deterministic actor used when no tenant model registry is configured."""

    async def refine(self, query: str) -> ResolvedQuery:
        """Keep the query unchanged while satisfying the structured contract."""
        return ResolvedQuery(original_query=query, standalone_query=query)


class PydanticAIQuestionRefinementActor:
    """Adapt a PydanticAI Agent to the v2 actor protocol."""

    def __init__(self, agent: Agent[Any, V2ResolvedQuery]) -> None:
        self._agent = agent

    async def refine(self, query: str) -> ResolvedQuery:
        """Run the agent and return its validated structured output."""
        result = await self._agent.run(query)
        return result.output


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
    context: PhaseExecutionContext,
    actor: QuestionRefinementActor,
) -> tuple[list[EventRecord], bool, ResolvedQuery | None, str | None]:
    """Journal, replay, and return one structured refinement result."""

    async def invoke() -> PhaseResultInput:
        try:
            result = V2ResolvedQuery.model_validate(
                (await actor.refine(state["query"])).model_dump()
            )
        except Exception as exc:
            message = str(exc) or REFINEMENT_ERROR_MESSAGE
            return PhaseResultInput(
                phase_name="question_refinement",
                normalized_result={"failed": True, "error": message},
                events=refinement_events(error=message),
                terminal_status="failed",
            )
        return PhaseResultInput(
            phase_name="question_refinement",
            normalized_result=result.model_dump(exclude_none=True),
            events=refinement_events(result),
        )

    result = await context.repository.get_or_invoke(
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        owner_instance_id=context.owner_instance_id,
        execution_epoch=context.execution_epoch,
        phase_name="question_refinement",
        invoke=invoke,
    )
    if result.normalized_result.get("failed") is True:
        return list(result.events), True, None, str(result.normalized_result["error"])
    return (
        list(result.events),
        False,
        V2ResolvedQuery.model_validate(result.normalized_result),
        None,
    )
