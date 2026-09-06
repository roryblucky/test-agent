"""Conservative, code-owned incomplete research rendering coverage."""

from datetime import UTC, datetime

import pytest

from app.langgraph_v2.agent_completion import (
    IncompleteResearch,
    completion_termination_reason,
    render_incomplete_research,
)
from app.langgraph_v2.agent_evidence import DataGapView, ToolUnavailableReason


def test_incomplete_research_prepends_escaped_data_gap_disclosure() -> None:
    answer = render_incomplete_research(
        "Published report.",
        IncompleteResearch(
            insufficient_evidence=False,
            data_gaps=(
                DataGapView(
                    requested_coverage="[click](https://evil.test)",
                    reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
                    observed_at=datetime(2026, 9, 6, 12, tzinfo=UTC),
                ),
            ),
        ),
    )

    assert answer.startswith(
        "Incomplete research: requested data was unavailable:\n"
        "- \\[click\\]\\(https://evil\\.test\\)\n\n"
    )
    assert answer.endswith("Published report.")
    assert "[click](https://evil.test)" not in answer


def test_incomplete_research_collapses_newlines_and_escapes_blockquotes() -> None:
    answer = render_incomplete_research(
        "Published report.",
        IncompleteResearch(
            insufficient_evidence=False,
            data_gaps=(
                DataGapView(
                    requested_coverage="> unavailable\nsource",
                    reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
                    observed_at=datetime(2026, 9, 6, 12, tzinfo=UTC),
                ),
            ),
        ),
    )

    assert "- \\> unavailable source" in answer
    assert "\nsource" not in answer


def test_incomplete_research_discloses_expected_task_failure() -> None:
    answer = render_incomplete_research(
        "Published report.",
        IncompleteResearch(
            insufficient_evidence=False,
            task_failures=1,
        ),
    )

    assert answer.startswith(
        "Incomplete research: one requested task could not complete.\n\n"
    )


def test_task_failure_disclosure_reuses_the_canonical_accepted_objective() -> None:
    answer = render_incomplete_research(
        "Published report.",
        IncompleteResearch(
            insufficient_evidence=False,
            task_failures=1,
            failed_task_objectives=("Café analysis",),
        ),
    )

    assert "- Café analysis" in answer
    with pytest.raises(ValueError, match="not canonical"):
        IncompleteResearch(
            insufficient_evidence=False,
            task_failures=1,
            failed_task_objectives=("Café\nanalysis",),
        )


def test_structural_reasons_are_ordered_and_preserved_in_completion() -> None:
    completion = IncompleteResearch(
        insufficient_evidence=False,
        structural_reasons=(
            "calculation_state_limit",
            "prepared_synthesis_limit",
        ),
    )

    assert completion_termination_reason(completion) == "execution_limit"
    assert render_incomplete_research("", completion) == (
        "Incomplete research: the Calculation state limit ended further work.\n"
        "Incomplete research: the prepared Synthesis limit ended further work."
    )
    with pytest.raises(ValueError, match="canonically ordered"):
        IncompleteResearch(
            insufficient_evidence=False,
            structural_reasons=(
                "prepared_synthesis_limit",
                "calculation_state_limit",
            ),
        )
