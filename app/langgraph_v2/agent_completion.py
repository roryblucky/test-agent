"""Deterministic conservative research-completion values."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.langgraph_v2.agent_batch import normalize_task_objective
from app.langgraph_v2.agent_evidence import DataGapView
from app.langgraph_v2.agent_termination import (
    COORDINATION_INVALID,
    COORDINATION_LIMIT,
    STRUCTURAL_REASON_ORDER,
    TASK_LIMIT,
    StructuralReason,
)

INSUFFICIENT_EVIDENCE_DISCLOSURE = (
    "Incomplete research: no eligible Evidence was available."
)
DATA_GAP_DISCLOSURE = "Incomplete research: requested data was unavailable:"
TASK_FAILURE_DISCLOSURE = "Incomplete research: one requested task could not complete."
STRUCTURAL_LIMIT_DISCLOSURES = {
    TASK_LIMIT: "Incomplete research: the Task limit ended further work.",
    COORDINATION_LIMIT: "Incomplete research: the Coordination limit ended further work.",
    COORDINATION_INVALID: "Incomplete research: the Coordinator could not produce a valid next decision.",
}
_MARKDOWN_ESCAPED_CHARACTERS = frozenset("\\`*_{}[]<>()#+-.!|~")

CompletionTerminationReason = Literal[
    "partial_results",
    "execution_limit",
    "partial_results_and_execution_limit",
    "insufficient_evidence",
]


class FailedTaskDisclosure(BaseModel):
    """Canonical accepted Task text paired with its stable identity at render time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)

    @field_validator("objective")
    @classmethod
    def _validate_objective(cls, value: str) -> str:
        if normalize_task_objective(value) != value:
            raise ValueError("Failed Task objective is not canonical")
        return value


class IncompleteResearch(BaseModel):
    """Application-only projection for a bounded incomplete terminal path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    insufficient_evidence: bool
    data_gaps: tuple[DataGapView, ...] = ()
    failed_task_ids: tuple[str, ...] = ()
    structural_reasons: tuple[StructuralReason, ...] = ()

    @field_validator("failed_task_ids")
    @classmethod
    def _validate_failed_task_ids(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not task_id for task_id in value) or len(set(value)) != len(value):
            raise ValueError("Failed Task identifiers are invalid")
        return value

    @field_validator("structural_reasons")
    @classmethod
    def _validate_structural_reasons(
        cls, value: tuple[StructuralReason, ...]
    ) -> tuple[StructuralReason, ...]:
        if (
            len(set(value)) != len(value)
            or tuple(sorted(value, key=STRUCTURAL_REASON_ORDER.__getitem__)) != value
        ):
            raise ValueError("Structural reasons are not canonically ordered")
        return value

    @property
    def has_data_gaps(self) -> bool:
        """Expose the sole partial-result signal used by terminal routing."""
        return bool(self.data_gaps)

    @property
    def has_task_failures(self) -> bool:
        """Expose expected Task inability as a separate partial-result signal."""
        return bool(self.failed_task_ids)

    @property
    def has_structural_limit(self) -> bool:
        """Expose a deterministic bound that ended further coordination."""
        return bool(self.structural_reasons)


def completion_termination_reason(
    completion: IncompleteResearch,
) -> CompletionTerminationReason:
    """Classify incomplete output without discarding any accepted causes."""
    if completion.has_structural_limit:
        return (
            "partial_results_and_execution_limit"
            if completion.has_data_gaps or completion.has_task_failures
            else "execution_limit"
        )
    if completion.has_data_gaps or completion.has_task_failures:
        return "partial_results"
    return "insufficient_evidence"


def insufficient_evidence_answer(
    completion: IncompleteResearch,
    *,
    failed_tasks: tuple[FailedTaskDisclosure, ...] = (),
) -> str:
    """Render the sole fixed disclosure for a zero-Evidence completion."""
    if not completion.insufficient_evidence:
        raise ValueError("Incomplete Research requires a completion signal")
    return render_incomplete_research(
        "", completion, failed_tasks=failed_tasks
    )


def _escape_markdown(value: str) -> str:
    return "".join(
        f"\\{character}" if character in _MARKDOWN_ESCAPED_CHARACTERS else character
        for character in " ".join(value.split())
    )


def render_incomplete_research(
    answer: str,
    completion: IncompleteResearch,
    *,
    failed_tasks: tuple[FailedTaskDisclosure, ...] = (),
) -> str:
    """Render code-owned incompleteness and its bounded missing coverage labels."""
    if tuple(task.task_id for task in failed_tasks) != completion.failed_task_ids:
        raise ValueError("Failed Task disclosures do not match accepted failures")
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
                for objective in (task.objective for task in failed_tasks)
            )
        )
    lines.extend(
        STRUCTURAL_LIMIT_DISCLOSURES[reason] for reason in completion.structural_reasons
    )
    if completion.insufficient_evidence:
        lines.append(INSUFFICIENT_EVIDENCE_DISCLOSURE)
    block = "\n".join(lines)
    if not block:
        raise ValueError("Incomplete Research requires a completion signal")
    return f"{block}\n\n{answer}" if answer else block
