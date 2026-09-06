"""Deterministic conservative research-completion values."""

from pydantic import BaseModel, ConfigDict

INSUFFICIENT_EVIDENCE_DISCLOSURE = (
    "Incomplete research: no eligible Evidence was available."
)


class IncompleteResearch(BaseModel):
    """Application-only projection for a bounded incomplete terminal path."""

    model_config = ConfigDict(frozen=True)

    insufficient_evidence: bool


def insufficient_evidence_answer(completion: IncompleteResearch) -> str:
    """Render the sole fixed disclosure for a zero-Evidence completion."""
    if not completion.insufficient_evidence:
        raise ValueError("Incomplete Research requires a completion signal")
    return INSUFFICIENT_EVIDENCE_DISCLOSURE
