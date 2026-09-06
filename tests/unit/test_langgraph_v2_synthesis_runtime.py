"""Bounded Synthesis actor coverage."""

from typing import cast

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from app.agents.synthesis import (
    SYNTHESIS_INSTRUCTIONS,
    SYNTHESIS_MAX_TOKENS,
    SYNTHESIS_TIMEOUT_SECONDS,
    PydanticAISynthesisActor,
    create_synthesis_agent,
)
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_evidence import (
    FinancialResearchReport,
    PreparedEvidence,
    PreparedSynthesis,
)


class _Registry:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create_agent(self, model_name: str, **kwargs: object) -> object:
        assert model_name == "synthesis"
        self.kwargs = kwargs
        return object()


def test_synthesis_factory_instructs_evidence_markers() -> None:
    registry = _Registry()

    create_synthesis_agent(cast(ModelRegistry, registry), model_name="synthesis")

    assert registry.kwargs is not None
    assert registry.kwargs["instructions"] == SYNTHESIS_INSTRUCTIONS
    assert registry.kwargs["tools"] == ()


@pytest.mark.asyncio
async def test_synthesis_returns_marked_report_with_fixed_request_limits() -> None:
    actor = PydanticAISynthesisActor(
        Agent(
            TestModel(
                call_tools=[],
                custom_output_args={"markdown_report": "Apple grew. [[E:1]]"},
            ),
            output_type=FinancialResearchReport,
            tools=(),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )
    result = await actor.synthesize(
        PreparedSynthesis(
            standalone_query="Apple",
            intent="market",
            evidence=(
                PreparedEvidence(
                    id="evidence-1",
                    source="filing",
                    source_url="https://example.test",
                    title="Filing",
                    excerpt="Apple grew.",
                ),
            ),
        )
    )
    assert result.markdown_report == "Apple grew. [[E:1]]"
    assert SYNTHESIS_TIMEOUT_SECONDS == 120
    assert SYNTHESIS_MAX_TOKENS == 4000
