"""Audit port and POC adapters for completed-output assessments."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class OutputAssessmentScope:
    """Trusted identity scope supplied by a phase to its audit port."""

    tenant_id: str
    conversation_id: str
    turn_id: UUID


def build_output_assessment_scope(
    *,
    tenant_id: str,
    conversation_id: str | None,
    turn_id: UUID | str | None,
) -> OutputAssessmentScope | None:
    """Build a scope only when the phase has complete Turn identity."""
    if not conversation_id or turn_id is None:
        logger.warning(
            "Skipping output assessment audit without Conversation and Turn identity"
        )
        return None
    try:
        normalized_turn_id = UUID(str(turn_id))
    except (TypeError, ValueError):
        logger.warning(
            "Skipping output assessment audit without a valid Turn identity"
        )
        return None
    return OutputAssessmentScope(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        turn_id=normalized_turn_id,
    )


async def record_output_assessment(
    audit: OutputAssessmentAudit | None,
    *,
    scope: OutputAssessmentScope | None,
    assessment_type: OutputAssessmentType,
    result: Mapping[str, Any],
) -> None:
    """Best-effort record one stable assessment inside its phase invocation."""
    if audit is None or scope is None:
        return
    record = OutputAssessmentAuditRecord(
        tenant_id=scope.tenant_id,
        conversation_id=scope.conversation_id,
        turn_id=scope.turn_id,
        assessment_id=output_assessment_id(scope.turn_id, assessment_type),
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
