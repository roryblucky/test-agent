"""Synthesis repair orchestration coverage."""

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
async def test_gate_rejection_repairs_the_same_frozen_prepared_value_once() -> None:
    class _RepairingSynthesis:
        def __init__(self) -> None:
            self.first_inputs: list[PreparedSynthesis] = []
            self.repair_inputs: list[PreparedSynthesis] = []
            self.validation_errors: tuple[str, ...] | None = None

        async def synthesize(
            self, prepared: PreparedSynthesis
        ) -> FinancialResearchReport:
            self.first_inputs.append(prepared)
            return FinancialResearchReport(
                markdown_report="Apple grew. [[E:1]] [[E:1]]"
            )

        async def repair(
            self,
            prepared: PreparedSynthesis,
            *,
            validation_errors: tuple[str, ...],
        ) -> FinancialResearchReport:
            self.repair_inputs.append(prepared)
            self.validation_errors = validation_errors
            return FinancialResearchReport(markdown_report="Apple grew. [[E:1]]")

    prepared = _prepared()
    actor = _RepairingSynthesis()

    published = await synthesize_report(actor, prepared)

    assert published.answer == "Apple grew. [[E:1]]"
    assert actor.first_inputs == [prepared]
    assert actor.repair_inputs == [prepared]
    assert actor.first_inputs[0] is prepared
    assert actor.repair_inputs[0] is prepared
    assert actor.validation_errors == ("Evidence marker is duplicated",)


@pytest.mark.asyncio
async def test_second_rejected_candidate_is_fatal_without_a_third_invocation() -> None:
    class _StillInvalidSynthesis:
        def __init__(self) -> None:
            self.calls = 0

        async def synthesize(
            self, prepared: PreparedSynthesis
        ) -> FinancialResearchReport:
            del prepared
            self.calls += 1
            return FinancialResearchReport(markdown_report="Apple grew.")

        async def repair(
            self,
            prepared: PreparedSynthesis,
            *,
            validation_errors: tuple[str, ...],
        ) -> FinancialResearchReport:
            del prepared, validation_errors
            self.calls += 1
            return FinancialResearchReport(markdown_report="Apple grew.")

    actor = _StillInvalidSynthesis()

    with pytest.raises(SynthesisCandidateRejected, match="Evidence marker is missing"):
        await synthesize_report(actor, _prepared())

    assert actor.calls == 2
