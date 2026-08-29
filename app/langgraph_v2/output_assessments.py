"""Audit port and POC adapters for completed-output assessments."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

OutputAssessmentType = Literal["groundedness", "post_moderation"]


class OutputAssessmentAuditRecord(BaseModel):
    """Tenant- and Turn-scoped assessment payload for an audit sink."""

    tenant_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn_id: UUID
    assessment_id: str = Field(min_length=1)
    assessment_type: OutputAssessmentType
    result: dict[str, Any]

    @property
    def assessment_identity(self) -> str:
        """Expose the stable identity using the domain vocabulary."""
        return self.assessment_id


class OutputAssessmentAudit(Protocol):
    """Port for recording one completed-output assessment."""

    async def record(self, assessment: OutputAssessmentAuditRecord) -> None:
        """Record an assessment without changing its canonical Answer."""
        ...


def output_assessment_id(
    turn_id: UUID | str,
    assessment_type: OutputAssessmentType,
) -> str:
    """Build a deterministic identity suitable for downstream deduplication."""
    return f"turn:{turn_id}:assessment:{assessment_type}"


class LoggingOutputAssessmentAudit:
    """Logging adapter used by the v2 POC in place of BigQuery infrastructure."""

    async def record(self, assessment: OutputAssessmentAuditRecord) -> None:
        """Emit the structured assessment to the application logger."""
        logger.info(
            "langgraph_v2 output assessment",
            extra={"output_assessment": assessment.model_dump(mode="json")},
        )


class MockOutputAssessmentAudit:
    """In-memory adapter for direct tests and local POC use."""

    def __init__(self) -> None:
        self.records: list[OutputAssessmentAuditRecord] = []

    async def record(self, assessment: OutputAssessmentAuditRecord) -> None:
        """Retain assessment records in invocation order."""
        self.records.append(assessment)


def _turn_id_from_state(
    state: Mapping[str, Any],
    context: Any,
) -> UUID | None:
    turn_id = getattr(context, "current_turn_id", None)
    if turn_id is not None:
        return turn_id
    raw_turn_id = state.get("turn_id")
    if raw_turn_id is None:
        return None
    try:
        return UUID(str(raw_turn_id))
    except (TypeError, ValueError):
        return None


async def record_output_assessment(
    audit: OutputAssessmentAudit | None,
    *,
    state: Mapping[str, Any],
    context: Any,
    assessment_type: OutputAssessmentType,
    result: Mapping[str, Any],
) -> None:
    """Best-effort record one stable assessment inside its phase invocation."""
    if audit is None:
        return
    turn_id = _turn_id_from_state(state, context)
    conversation_id = state.get("conversation_id")
    if turn_id is None or not isinstance(conversation_id, str):
        logger.warning(
            "Skipping output assessment audit without Conversation and Turn identity"
        )
        return
    record = OutputAssessmentAuditRecord(
        tenant_id=context.tenant_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        assessment_id=output_assessment_id(turn_id, assessment_type),
        assessment_type=assessment_type,
        result=dict(result),
    )
    try:
        await audit.record(record)
    except Exception:
        logger.exception(
            "Output assessment audit adapter failed for assessment_id=%s",
            record.assessment_id,
        )
