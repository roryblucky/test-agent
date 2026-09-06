"""Deterministic conservative research-completion values."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.langgraph_v2.agent_batch import normalize_task_objective
from app.langgraph_v2.agent_evidence import DataGapView

INSUFFICIENT_EVIDENCE_DISCLOSURE = (
    "Incomplete research: no eligible Evidence was available."
)
DATA_GAP_DISCLOSURE = "Incomplete research: requested data was unavailable:"
TASK_FAILURE_DISCLOSURE = "Incomplete research: one requested task could not complete."
STRUCTURAL_LIMIT_DISCLOSURES = {
    "task_limit": "Incomplete research: the Task limit ended further work.",
    "coordination_limit": "Incomplete research: the Coordination limit ended further work.",
    "coordinator_context_limit": "Incomplete research: the Coordinator context limit ended further work.",
    "coordination_invalid": "Incomplete research: the Coordinator could not produce a valid next decision.",
}
_MARKDOWN_ESCAPED_CHARACTERS = frozenset("\\`*_{}[]<>()#+-.!|~")


class IncompleteResearch(BaseModel):
    """Application-only projection for a bounded incomplete terminal path."""

    model_config = ConfigDict(frozen=True)

    insufficient_evidence: bool
    data_gaps: tuple[DataGapView, ...] = ()
    task_failures: int = Field(default=0, ge=0)
    failed_task_objectives: tuple[str, ...] = ()
    structural_reason: Literal[
        "task_limit",
        "coordination_limit",
        "coordinator_context_limit",
        "coordination_invalid",
    ] | None = None

    @field_validator("failed_task_objectives")
    @classmethod
    def _validate_failed_task_objectives(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require disclosure to reuse the accepted canonical Task objective."""
        if any(normalize_task_objective(objective) != objective for objective in value):
            raise ValueError("Failed Task objective is not canonical")
        return value

    @property
    def has_data_gaps(self) -> bool:
        """Expose the sole partial-result signal used by terminal routing."""
        return bool(self.data_gaps)

    @property
    def has_task_failures(self) -> bool:
        """Expose expected Task inability as a separate partial-result signal."""
        return self.task_failures > 0

    @property
    def has_structural_limit(self) -> bool:
        """Expose a deterministic bound that ended further coordination."""
        return self.structural_reason is not None


def insufficient_evidence_answer(completion: IncompleteResearch) -> str:
    """Render the sole fixed disclosure for a zero-Evidence completion."""
    if not completion.insufficient_evidence:
        raise ValueError("Incomplete Research requires a completion signal")
    return render_incomplete_research("", completion)


def _escape_markdown(value: str) -> str:
    return "".join(
        f"\\{character}" if character in _MARKDOWN_ESCAPED_CHARACTERS else character
        for character in " ".join(value.split())
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
    if completion.has_task_failures:
        lines.extend(
            (TASK_FAILURE_DISCLOSURE,)
            + tuple(
                f"- {_escape_markdown(objective)}"
                for objective in completion.failed_task_objectives
            )
        )
    if completion.structural_reason is not None:
        lines.append(STRUCTURAL_LIMIT_DISCLOSURES[completion.structural_reason])
    if completion.insufficient_evidence:
        lines.append(INSUFFICIENT_EVIDENCE_DISCLOSURE)
    block = "\n".join(lines)
    if not block:
        raise ValueError("Incomplete Research requires a completion signal")
    return f"{block}\n\n{answer}" if answer else block
