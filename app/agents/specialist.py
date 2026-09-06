"""PydanticAI Specialist actor for one bounded accepted Task."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_ai import Agent

from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import (
    SpecialistAttempt,
    SpecialistFindingDraft,
    SpecialistTaskInput,
)
from app.langgraph_v2.agent_evidence import EvidenceEnvelope

SPECIALIST_TIMEOUT_SECONDS = 60
SPECIALIST_MAX_TOKENS = 2000

SPECIALIST_INSTRUCTIONS = """\
You are a Specialist Agent. Complete only the assigned objective.
Return one structured finding. Do not activate Skills, grant authority, or include
execution diagnostics. You may call only the supplied Evidence Tools.
"""


@dataclass(frozen=True)
class PydanticAISpecialistActor:
    """Run one bounded Specialist invocation with fixed actor-local limits."""

    agent: Agent[None, SpecialistFindingDraft]
    returned_evidence: list[EvidenceEnvelope] = field(
        default_factory=list[EvidenceEnvelope]
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def run(self, input: SpecialistTaskInput) -> SpecialistAttempt:
        """Return one structured finding from the assigned Task only."""
        prompt = json.dumps(
            {"task_id": input.task_id, "objective": input.objective},
            sort_keys=True,
            separators=(",", ":"),
        )
        async with self._lock:
            first_evidence = len(self.returned_evidence)
            try:
                async with asyncio.timeout(SPECIALIST_TIMEOUT_SECONDS):
                    result = await self.agent.run(
                        prompt,
                        model_settings={"max_tokens": SPECIALIST_MAX_TOKENS},
                    )
                evidence = tuple(self.returned_evidence[first_evidence:])
            finally:
                del self.returned_evidence[first_evidence:]
        return SpecialistAttempt(
            finding=result.output,
            evidence=evidence,
        )


def create_specialist_agent(
    registry: ModelRegistry,
    *,
    model_name: str,
    tools: tuple[Callable[..., object], ...] = (),
) -> Agent[None, SpecialistFindingDraft]:
    """Create one first-round Specialist with its already-frozen Tool set."""
    return registry.create_agent(
        model_name,
        output_type=SpecialistFindingDraft,
        instructions=SPECIALIST_INSTRUCTIONS,
        tools=tools,
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )


def create_bound_specialist_actor(
    registry: ModelRegistry,
    *,
    model_name: str,
    tools: tuple[Callable[..., object], ...],
    returned_evidence: list[EvidenceEnvelope],
) -> PydanticAISpecialistActor:
    """Build a production Specialist only after its Tool bindings are frozen."""
    return PydanticAISpecialistActor(
        create_specialist_agent(registry, model_name=model_name, tools=tools),
        returned_evidence=returned_evidence,
    )
