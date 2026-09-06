"""PydanticAI no-Tool Specialist actor for first accepted Task."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from pydantic_ai import Agent

from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import SpecialistFindingDraft, SpecialistTaskInput

SPECIALIST_TIMEOUT_SECONDS = 60
SPECIALIST_MAX_TOKENS = 2000

SPECIALIST_INSTRUCTIONS = """\
You are a Specialist Agent. Complete only the assigned objective.
Return one structured finding. Do not call Tools, activate Skills, grant authority,
or include execution diagnostics.
"""


@dataclass(frozen=True)
class PydanticAISpecialistActor:
    """Run one no-Tool Specialist invocation with fixed actor-local limits."""

    agent: Agent[None, SpecialistFindingDraft]

    async def run(self, input: SpecialistTaskInput) -> SpecialistFindingDraft:
        """Return one structured finding from the assigned Task only."""
        prompt = json.dumps(
            {"task_id": input.task_id, "objective": input.objective},
            sort_keys=True,
            separators=(",", ":"),
        )
        async with asyncio.timeout(SPECIALIST_TIMEOUT_SECONDS):
            result = await self.agent.run(
                prompt,
                model_settings={"max_tokens": SPECIALIST_MAX_TOKENS},
            )
        return result.output


def create_specialist_agent(
    registry: ModelRegistry,
    *,
    model_name: str,
) -> Agent[None, SpecialistFindingDraft]:
    """Create one first-round Specialist with no executable Tools or retries."""
    return registry.create_agent(
        model_name,
        output_type=SpecialistFindingDraft,
        instructions=SPECIALIST_INSTRUCTIONS,
        tools=(),
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )
