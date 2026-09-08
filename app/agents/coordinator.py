"""PydanticAI Coordinator actor with no executable business Tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior

from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import DispatchBatch
from app.langgraph_v2.agent_coordination import (
    CoordinationCandidateRejected,
    CoordinatorActorResult,
    CoordinatorDecision,
    CoordinatorDecisionExhausted,
    CoordinatorInput,
    validate_coordinator_decision,
)
from app.langgraph_v2.agent_termination import COORDINATION_LIMIT, TASK_LIMIT

COORDINATOR_TIMEOUT_SECONDS = 60
COORDINATOR_MAX_TOKENS = 1500
COORDINATOR_OUTPUT_RETRIES = 1

COORDINATOR_INSTRUCTIONS = """\
You are the Coordinator Agent for an enterprise research platform.
Return only one structured decision: either Finish, or Dispatch for one eligible
registered Specialist batch with non-empty objectives. Select earlier successful
Task Results only through context_task_ids supplied in the current projection.
Do not Dispatch when dispatch_allowed is false. Never propose more Tasks than
remaining_task_slots; return Finish when no valid Dispatch remains.
Do not answer the user, call Tools, select Skills, or grant authority.
"""


class _CoordinatorModelRetry(ModelRetry):
    """Keep the code-owned stop reason while sending actionable model feedback."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _retry_feedback(
    input: CoordinatorInput,
    decision: CoordinatorDecision,
    reason: str,
) -> str:
    """Return safe, actionable feedback for one rejected typed decision."""
    if reason == COORDINATION_LIMIT:
        return "Dispatch is not allowed in this round; return Finish."
    if reason == TASK_LIMIT:
        task_count = len(decision.tasks) if isinstance(decision, DispatchBatch) else 0
        return (
            f"Dispatch proposed {task_count} Tasks but remaining_task_slots is "
            f"{input.remaining_task_slots}; reduce the Task count or return Finish."
        )
    return reason


def _validate_output(
    ctx: RunContext[CoordinatorInput],
    decision: CoordinatorDecision,
) -> CoordinatorDecision:
    """Ask PydanticAI to repair one typed but policy-invalid decision."""
    try:
        return validate_coordinator_decision(ctx.deps, decision)
    except CoordinationCandidateRejected as error:
        raise _CoordinatorModelRetry(
            error.reason,
            _retry_feedback(ctx.deps, decision, error.reason),
        ) from error


@dataclass(frozen=True)
class PydanticAICoordinatorActor:
    """Run one policy-bounded, already-configured Coordinator Agent."""

    agent: Agent[CoordinatorInput, CoordinatorDecision]

    async def decide(self, input: CoordinatorInput) -> CoordinatorActorResult:
        """Run one Agent invocation whose retry preserves validation history."""
        prompt = json.dumps(
            {"input": input.model_dump(mode="json")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            result = await self.agent.run(
                prompt,
                deps=input,
                model_settings={
                    "max_tokens": COORDINATOR_MAX_TOKENS,
                    "timeout": COORDINATOR_TIMEOUT_SECONDS,
                },
            )
        except UnexpectedModelBehavior as error:
            cause = error.__cause__
            if isinstance(cause, _CoordinatorModelRetry):
                reason = cause.reason
            elif isinstance(cause, ValidationError):
                reason = "Coordinator decision is invalid"
            else:
                raise
            return CoordinatorDecisionExhausted(reason=reason)
        return result.output


def create_coordinator_agent(
    registry: ModelRegistry,
    *,
    model_name: str,
) -> Agent[CoordinatorInput, CoordinatorDecision]:
    """Create Coordinator with one structured-output retry and no Tools."""
    agent = cast(
        Agent[CoordinatorInput, CoordinatorDecision],
        registry.create_agent(
            model_name,
            deps_type=CoordinatorInput,
            output_type=cast(type[Any], CoordinatorDecision),
            instructions=COORDINATOR_INSTRUCTIONS,
            tools=(),
            tool_retries=0,
            output_retries=COORDINATOR_OUTPUT_RETRIES,
            end_strategy="early",
        ),
    )
    agent.output_validator(_validate_output)
    return agent
