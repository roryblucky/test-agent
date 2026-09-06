"""Deterministic conservative research-completion values."""

from pydantic import BaseModel, ConfigDict

from app.langgraph_v2.agent_evidence import DataGapView

INSUFFICIENT_EVIDENCE_DISCLOSURE = (
    "Incomplete research: no eligible Evidence was available."
)
DATA_GAP_DISCLOSURE = "Incomplete research: requested data was unavailable:"


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
    return render_incomplete_research(INSUFFICIENT_EVIDENCE_DISCLOSURE, completion)


def render_incomplete_research(answer: str, completion: IncompleteResearch) -> str:
    """Render code-owned incompleteness and its bounded missing coverage labels."""
    if not completion.has_data_gaps:
        return answer
    coverage = "\n".join(
        f"- {gap.requested_coverage}" for gap in completion.data_gaps
    )
    return f"{answer}\n\n{DATA_GAP_DISCLOSURE}\n{coverage}"
