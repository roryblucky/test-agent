"""Deterministic conservative research-completion values."""

from pydantic import BaseModel, ConfigDict

from app.langgraph_v2.agent_evidence import DataGapView

INSUFFICIENT_EVIDENCE_DISCLOSURE = (
    "Incomplete research: no eligible Evidence was available."
)
DATA_GAP_DISCLOSURE = "Incomplete research: requested data was unavailable:"
_MARKDOWN_ESCAPED_CHARACTERS = frozenset("\\`*_{}[]<>()#+-.!|")


class IncompleteResearch(BaseModel):
    """Application-only projection for a bounded incomplete terminal path."""

    model_config = ConfigDict(frozen=True)

    insufficient_evidence: bool
    data_gaps: tuple[DataGapView, ...] = ()

    @property
    def has_data_gaps(self) -> bool:
        """Expose the sole partial-result signal used by terminal routing."""
        return bool(self.data_gaps)


def insufficient_evidence_answer(completion: IncompleteResearch) -> str:
    """Render the sole fixed disclosure for a zero-Evidence completion."""
    if not completion.insufficient_evidence:
        raise ValueError("Incomplete Research requires a completion signal")
    return render_incomplete_research("", completion)


def _escape_markdown(value: str) -> str:
    return "".join(
        f"\\{character}" if character in _MARKDOWN_ESCAPED_CHARACTERS else character
        for character in value
    )


def render_incomplete_research(answer: str, completion: IncompleteResearch) -> str:
    """Render code-owned incompleteness and its bounded missing coverage labels."""
    lines: list[str] = []
    if completion.has_data_gaps:
        lines.extend(
            (DATA_GAP_DISCLOSURE,)
            + tuple(
                f"- {_escape_markdown(gap.requested_coverage)}"
                for gap in completion.data_gaps
            )
        )
    if completion.insufficient_evidence:
        lines.append(INSUFFICIENT_EVIDENCE_DISCLOSURE)
    block = "\n".join(lines)
    if not block:
        raise ValueError("Incomplete Research requires a completion signal")
    return f"{block}\n\n{answer}" if answer else block
