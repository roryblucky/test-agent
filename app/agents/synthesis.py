"""PydanticAI Synthesis actor for a bounded Evidence projection."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_evidence import (
    FinancialResearchReport,
    PreparedSynthesis,
    SynthesisOutputInvalid,
)

SYNTHESIS_TIMEOUT_SECONDS = 120
SYNTHESIS_MAX_TOKENS = 4000
SYNTHESIS_INSTRUCTIONS = """\
Write one Markdown report grounded exclusively in the supplied Evidence excerpts.
Every factual claim must cite its positional Evidence with exactly [[E:n]], where n
is the one-based item position in the prepared Evidence list. For an eligible
Calculation, use exactly its supplied [[C:n]] alias; never retype or infer a
numerical value. Do not cite absent Evidence or use Tools, retries, or external
knowledge.
"""


@dataclass(frozen=True)
class PydanticAISynthesisActor:
    """Run one no-Tool report candidate from a frozen Evidence projection."""

    agent: Agent[None, FinancialResearchReport]

    async def synthesize(self, prepared: PreparedSynthesis) -> FinancialResearchReport:
        """Return Markdown with only prepared Evidence markers."""
        return await self._run(prepared)

    async def repair(
        self,
        prepared: PreparedSynthesis,
        *,
        validation_errors: tuple[str, ...],
    ) -> FinancialResearchReport:
        """Return the sole repair from the exact frozen input and safe gate errors."""
        return await self._run(prepared, validation_errors=validation_errors)

    async def _run(
        self,
        prepared: PreparedSynthesis,
        *,
        validation_errors: tuple[str, ...] = (),
    ) -> FinancialResearchReport:
        """Make exactly one no-Tool request for one frozen Synthesis input."""
        prompt_body: object = prepared.model_dump(mode="json")
        if validation_errors:
            prompt_body = {
                "prepared": prompt_body,
                "validation_errors": validation_errors,
            }
        prompt = json.dumps(prompt_body, sort_keys=True, separators=(",", ":"))
        async with asyncio.timeout(SYNTHESIS_TIMEOUT_SECONDS):
            try:
                result = await self.agent.run(
                    prompt, model_settings={"max_tokens": SYNTHESIS_MAX_TOKENS}
                )
            except UnexpectedModelBehavior as error:
                raise SynthesisOutputInvalid() from error
        return result.output


def create_synthesis_agent(
    registry: ModelRegistry, *, model_name: str
) -> Agent[None, FinancialResearchReport]:
    """Create the fixed no-Tool first Synthesis invocation."""
    return registry.create_agent(
        model_name,
        output_type=FinancialResearchReport,
        instructions=SYNTHESIS_INSTRUCTIONS,
        tools=(),
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )
