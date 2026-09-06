"""PydanticAI Specialist actor for one bounded accepted Task."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.tool_manager import ToolManager

from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import (
    SpecialistAttempt,
    SpecialistFindingDraft,
    SpecialistTaskInput,
)
from app.langgraph_v2.agent_evidence import SpecialistToolCapture
from app.langgraph_v2.agent_skills import SkillInvocation

SPECIALIST_TIMEOUT_SECONDS = 60
SPECIALIST_MAX_TOKENS = 2000

SPECIALIST_INSTRUCTIONS = """\
You are a Specialist Agent. Complete only the assigned objective.
Return one structured finding. You may activate one eligible Skill named in the
supplied summaries when it helps the objective. Skill activation never grants
authority. You may call only the supplied Evidence Tools and must not include
execution diagnostics.
"""


@dataclass(frozen=True)
class PydanticAISpecialistActor:
    """Run one bounded Specialist invocation with fixed actor-local limits."""

    agent: Agent[None, SpecialistFindingDraft]
    tool_capture: SpecialistToolCapture = field(default_factory=SpecialistToolCapture)
    skill_invocation: SkillInvocation | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def run(self, input: SpecialistTaskInput) -> SpecialistAttempt:
        """Return one structured finding from the assigned Task only."""
        skill_summaries = (
            self.skill_invocation.summaries
            if self.skill_invocation is not None
            else ()
        )
        prompt = json.dumps(
            {
                "task_id": input.task_id,
                "objective": input.objective,
                "skill_summaries": [
                    summary.model_dump(mode="json")
                    for summary in skill_summaries
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        async with self._lock:
            first_evidence = len(self.tool_capture.evidence)
            first_unavailability = len(self.tool_capture.unavailability)
            try:
                with ToolManager.parallel_execution_mode("parallel_ordered_events"):
                    result: AgentRunResult[
                        SpecialistFindingDraft
                    ] = await self.agent.run(
                        prompt,
                        model_settings={
                            "max_tokens": SPECIALIST_MAX_TOKENS,
                            "timeout": SPECIALIST_TIMEOUT_SECONDS,
                            "parallel_tool_calls": True,
                        },
                    )
                evidence = tuple(self.tool_capture.evidence[first_evidence:])
                unavailability = tuple(
                    self.tool_capture.unavailability[first_unavailability:]
                )
            finally:
                del self.tool_capture.evidence[first_evidence:]
                del self.tool_capture.unavailability[first_unavailability:]
        return SpecialistAttempt(
            finding=result.output,
            evidence=evidence,
            unavailability=unavailability,
            skill_pins=(
                self.skill_invocation.pins
                if self.skill_invocation is not None
                else ()
            ),
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
    tool_capture: SpecialistToolCapture,
    skill_invocation: SkillInvocation | None = None,
) -> PydanticAISpecialistActor:
    """Build a production Specialist only after its Tool bindings are frozen."""
    return PydanticAISpecialistActor(
        create_specialist_agent(registry, model_name=model_name, tools=tools),
        tool_capture=tool_capture,
        skill_invocation=skill_invocation,
    )
