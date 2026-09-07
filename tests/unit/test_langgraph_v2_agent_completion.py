"""Conservative, code-owned incomplete research rendering coverage."""

from datetime import UTC, datetime

import pytest

from app.langgraph_v2.agent_completion import (
    FailedTaskDisclosure,
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
            failed_task_ids=("task-1",),
        ),
        failed_tasks=(
            FailedTaskDisclosure(task_id="task-1", objective="Market analysis"),
        ),
    )

    assert answer.startswith(
        "Incomplete research: one requested task could not complete.\n"
        "- Market analysis\n\n"
    )


def test_incomplete_research_preserves_failed_task_ids_across_round_trip() -> None:
    completion = IncompleteResearch(
        insufficient_evidence=False,
        failed_task_ids=("task-1",),
    )

    assert completion.failed_task_ids == ("task-1",)
    assert IncompleteResearch.model_validate_json(completion.model_dump_json()) == completion
    assert render_incomplete_research(
        "Published report.",
        completion,
        failed_tasks=(
            FailedTaskDisclosure(task_id="task-1", objective="Café analysis"),
        ),
    ) == (
        "Incomplete research: one requested task could not complete.\n"
        "- Café analysis\n\nPublished report."
    )


def test_incomplete_research_rejects_a_disclosure_for_the_wrong_failed_task() -> None:
    completion = IncompleteResearch(
        insufficient_evidence=False,
        failed_task_ids=("task-1",),
    )

    with pytest.raises(ValueError, match="do not match accepted failures"):
        render_incomplete_research(
            "Published report.",
            completion,
            failed_tasks=(
                FailedTaskDisclosure(task_id="task-2", objective="Market analysis"),
            ),
        )


def test_task_failure_disclosure_reuses_the_canonical_accepted_objective() -> None:
    answer = render_incomplete_research(
        "Published report.",
        IncompleteResearch(
            insufficient_evidence=False,
            failed_task_ids=("task-1",),
        ),
        failed_tasks=(
            FailedTaskDisclosure(task_id="task-1", objective="Café analysis"),
        ),
    )

    assert "- Café analysis" in answer
    with pytest.raises(ValueError, match="not canonical"):
        render_incomplete_research(
            "Published report.",
            IncompleteResearch(
                insufficient_evidence=False,
                failed_task_ids=("task-1",),
            ),
            failed_tasks=(
                FailedTaskDisclosure(task_id="task-1", objective="Café\nanalysis"),
            ),
        )


def test_structural_reasons_are_ordered_and_preserved_in_completion() -> None:
    completion = IncompleteResearch(
        insufficient_evidence=False,
        structural_reasons=(
            "task_limit",
            "coordination_limit",
        ),
    )

    assert completion_termination_reason(completion) == "execution_limit"
    assert render_incomplete_research("", completion) == (
        "Incomplete research: the Task limit ended further work.\n"
        "Incomplete research: the Coordination limit ended further work."
    )
    with pytest.raises(ValueError, match="canonically ordered"):
        IncompleteResearch(
            insufficient_evidence=False,
            structural_reasons=(
                "coordination_limit",
                "task_limit",
            ),
        )


@pytest.mark.parametrize(
    ("completion", "expected_reason"),
    [
        (
            IncompleteResearch(
                insufficient_evidence=False,
                failed_task_ids=("task-1",),
            ),
            "partial_results",
        ),
        (
            IncompleteResearch(
                insufficient_evidence=False,
                structural_reasons=("task_limit",),
            ),
            "execution_limit",
        ),
        (
            IncompleteResearch(
                insufficient_evidence=True,
                failed_task_ids=("task-1",),
                structural_reasons=("task_limit",),
            ),
            "partial_results_and_execution_limit",
        ),
        (IncompleteResearch(insufficient_evidence=True), "insufficient_evidence"),
    ],
)
def test_completion_termination_reason_preserves_partial_and_limit_priority(
    completion: IncompleteResearch,
    expected_reason: str,
) -> None:
    assert completion_termination_reason(completion) == expected_reason
