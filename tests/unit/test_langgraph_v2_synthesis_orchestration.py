"""Synthesis publication orchestration coverage."""

import pytest
from pydantic import ValidationError

from app.langgraph_v2.agent_evidence import (
    FinancialResearchReport,
    PreparedEvidence,
    PreparedSynthesis,
    SynthesisCandidateRejected,
    synthesize_report,
)


def _prepared() -> PreparedSynthesis:
    return PreparedSynthesis(
        standalone_query="Apple outlook",
        intent="market_outlook",
        evidence=(
            PreparedEvidence(
                id="evidence-1",
                source="filing",
                source_url="https://example.test/filing",
                title="Annual filing",
                excerpt="Apple revenue grew.",
            ),
        ),
    )


def test_prepared_synthesis_rejects_noncanonical_input_fields() -> None:
    with pytest.raises(ValidationError, match="conversation_history"):
        PreparedSynthesis.model_validate(
            {
                **_prepared().model_dump(mode="json"),
                "conversation_history": ["must not reach synthesis"],
            }
        )


@pytest.mark.asyncio
async def test_synthesize_report_publishes_one_actor_candidate() -> None:
    class _Synthesis:
        def __init__(self) -> None:
            self.inputs: list[PreparedSynthesis] = []

        async def synthesize(
            self, prepared: PreparedSynthesis
        ) -> FinancialResearchReport:
            self.inputs.append(prepared)
            return FinancialResearchReport(markdown_report="Apple grew. [[E:1]]")

    prepared = _prepared()
    actor = _Synthesis()

    published = await synthesize_report(actor, prepared)

    assert published.answer == "Apple grew. [[E:1]]"
    assert actor.inputs == [prepared]
    assert actor.inputs[0] is prepared


@pytest.mark.asyncio
async def test_rejected_candidate_is_not_repaired_by_orchestration() -> None:
    class _InvalidSynthesis:
        def __init__(self) -> None:
            self.calls = 0

        async def synthesize(
            self, prepared: PreparedSynthesis
        ) -> FinancialResearchReport:
            del prepared
            self.calls += 1
            return FinancialResearchReport(markdown_report="Apple grew.")

    actor = _InvalidSynthesis()

    with pytest.raises(SynthesisCandidateRejected, match="Evidence marker is missing"):
        await synthesize_report(actor, _prepared())

    assert actor.calls == 1
