"""PydanticAI Coordinator actor with no executable business Tools."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from pydantic_ai import Agent

from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_graph import CoordinatorInput, Finish

COORDINATOR_TIMEOUT_SECONDS = 60
COORDINATOR_MAX_TOKENS = 1500

COORDINATOR_INSTRUCTIONS = """\
You are the Coordinator Agent for an enterprise research platform.
Return only the structured Finish decision. Do not answer the user, call Tools,
select Skills, or grant authority.
"""


@dataclass(frozen=True)
class PydanticAICoordinatorActor:
    """Run one policy-bounded Coordinator invocation without tools or retries."""

    agent: Agent[None, Finish]

    async def decide(self, input: CoordinatorInput) -> Finish:
        """Decide using only deterministic current-Run Coordinator input."""
        prompt = json.dumps(
            input.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        async with asyncio.timeout(COORDINATOR_TIMEOUT_SECONDS):
            result = await self.agent.run(
                prompt,
                model_settings={"max_tokens": COORDINATOR_MAX_TOKENS},
            )
        return result.output


def create_coordinator_agent(
    registry: ModelRegistry,
    *,
    model_name: str,
) -> Agent[None, Finish]:
    """Create first-round Coordinator with explicit one-request limits."""
    return registry.create_agent(
        model_name,
        output_type=Finish,
        instructions=COORDINATOR_INSTRUCTIONS,
        tools=(),
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )
