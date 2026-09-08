"""PydanticAI Synthesis actor for a bounded Evidence projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import Agent, ModelRetry, RunContext

from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_evidence import (
    FinancialResearchReport,
    PreparedSynthesis,
    SynthesisCandidateRejected,
    validate_synthesis_candidate,
    validate_synthesis_projection,
)

SYNTHESIS_TIMEOUT_SECONDS = 120
SYNTHESIS_MAX_TOKENS = 4000
SYNTHESIS_OUTPUT_RETRIES = 1
SYNTHESIS_INSTRUCTIONS = """\
Write one Markdown report grounded exclusively in the supplied Evidence excerpts.
Every factual claim must cite its positional Evidence with exactly [[E:n]], where n
is the one-based item position in the prepared Evidence list. For an eligible
Calculation, use exactly its supplied [[C:n]] alias; never retype or infer a
numerical value. Do not cite absent Evidence or use Tools, retries, or external
knowledge.
"""


def _validate_output(
    ctx: RunContext[PreparedSynthesis],
    candidate: FinancialResearchReport,
) -> FinancialResearchReport:
    """Return marker failures to the same native Agent run for correction."""
    try:
        return validate_synthesis_candidate(candidate, ctx.deps)
    except SynthesisCandidateRejected as error:
        raise ModelRetry("; ".join(error.validation_errors)) from error


@dataclass(frozen=True)
class PydanticAISynthesisActor:
    """Run one already-configured Agent from a frozen Evidence projection."""

    agent: Agent[PreparedSynthesis, FinancialResearchReport]

    async def synthesize(self, prepared: PreparedSynthesis) -> FinancialResearchReport:
        """Return a report after at most two requests in one native Agent run."""
        validate_synthesis_projection(prepared)
        prompt = json.dumps(
            prepared.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        result = await self.agent.run(
            prompt,
            deps=prepared,
            model_settings={
                "max_tokens": SYNTHESIS_MAX_TOKENS,
                "timeout": SYNTHESIS_TIMEOUT_SECONDS,
            },
        )
        return result.output


def create_synthesis_agent(
    registry: ModelRegistry, *, model_name: str
) -> Agent[PreparedSynthesis, FinancialResearchReport]:
    """Create Synthesis with one marker-correction retry and no Tools."""
    agent = cast(
        Agent[PreparedSynthesis, FinancialResearchReport],
        registry.create_agent(
            model_name,
            deps_type=PreparedSynthesis,
            output_type=cast(type[Any], FinancialResearchReport),
            instructions=SYNTHESIS_INSTRUCTIONS,
            tools=(),
            retries=0,
            tool_retries=0,
            output_retries=SYNTHESIS_OUTPUT_RETRIES,
            end_strategy="early",
        ),
    )
    agent.output_validator(_validate_output)
    return agent
