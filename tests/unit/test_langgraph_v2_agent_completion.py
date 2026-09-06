"""Conservative, code-owned incomplete research rendering coverage."""

from datetime import UTC, datetime

from app.langgraph_v2.agent_completion import (
    IncompleteResearch,
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
