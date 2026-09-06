"""Deterministic conservative research-completion values."""

from pydantic import BaseModel, ConfigDict

INSUFFICIENT_EVIDENCE_DISCLOSURE = (
    "Incomplete research: no eligible Evidence was available."
)
DATA_GAP_DISCLOSURE = (
    "Incomplete research: one or more requested data sources were unavailable."
)


class IncompleteResearch(BaseModel):
    """Application-only projection for a bounded incomplete terminal path."""

    model_config = ConfigDict(frozen=True)

    insufficient_evidence: bool
    has_data_gaps: bool = False


def insufficient_evidence_answer(completion: IncompleteResearch) -> str:
    """Render the sole fixed disclosure for a zero-Evidence completion."""
    if not completion.insufficient_evidence:
        raise ValueError("Incomplete Research requires a completion signal")
    return append_data_gap_disclosure(INSUFFICIENT_EVIDENCE_DISCLOSURE, completion)


def append_data_gap_disclosure(answer: str, completion: IncompleteResearch) -> str:
    """Append the conservative code-owned gap disclosure when required."""
    if not completion.has_data_gaps:
        return answer
    return f"{answer}\n\n{DATA_GAP_DISCLOSURE}"
