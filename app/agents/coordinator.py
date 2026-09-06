"""PydanticAI Coordinator actor with no executable business Tools."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_coordination import (
    CoordinatorDecision,
    CoordinatorInput,
    CoordinatorOutputInvalid,
)

COORDINATOR_TIMEOUT_SECONDS = 60
COORDINATOR_MAX_TOKENS = 1500

COORDINATOR_INSTRUCTIONS = """\
You are the Coordinator Agent for an enterprise research platform.
Return only one structured decision: either Finish, or Dispatch for one eligible
registered Specialist batch with non-empty objectives. Select earlier successful
Task Results only through context_task_ids supplied in the current projection.
Do not answer the user, call Tools, select Skills, or grant authority.
"""


@dataclass(frozen=True)
class PydanticAICoordinatorActor:
    """Run one policy-bounded Coordinator invocation without tools or retries."""

    agent: Agent[None, CoordinatorDecision]

    async def decide(self, input: CoordinatorInput) -> CoordinatorDecision:
        """Decide using only deterministic current-Run Coordinator input."""
        return await self._run(input)

    async def repair(
        self,
        input: CoordinatorInput,
        *,
        rejection: str,
    ) -> CoordinatorDecision:
        """Make the sole same-round repair without rebuilding Coordinator input."""
        return await self._run(input, validation_feedback=rejection)

    async def _run(
        self,
        input: CoordinatorInput,
        *,
        validation_feedback: str | None = None,
    ) -> CoordinatorDecision:
        """Run one request with optional deterministic validation feedback."""
        prompt = json.dumps(
            {
                "input": input.model_dump(mode="json"),
                "validation_feedback": validation_feedback,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        async with asyncio.timeout(COORDINATOR_TIMEOUT_SECONDS):
            try:
                result = await self.agent.run(
                    prompt,
                    model_settings={"max_tokens": COORDINATOR_MAX_TOKENS},
                )
            except UnexpectedModelBehavior as error:
                raise CoordinatorOutputInvalid("Coordinator decision is invalid") from error
        return result.output


def create_coordinator_agent(
    registry: ModelRegistry,
    *,
    model_name: str,
) -> Agent[None, CoordinatorDecision]:
    """Create first-round Coordinator with explicit one-request limits."""
    return cast(
        Agent[None, CoordinatorDecision],
        registry.create_agent(
            model_name,
            output_type=cast(type[Any], CoordinatorDecision),
            instructions=COORDINATOR_INSTRUCTIONS,
            tools=(),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        ),
    )
