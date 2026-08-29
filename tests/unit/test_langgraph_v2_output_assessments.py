from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.langgraph_v2.output_assessments import (
    MockOutputAssessmentAudit,
    OutputAssessmentAuditRecord,
    output_assessment_id,
)


def test_output_assessment_identity_is_stable_and_type_specific() -> None:
    turn_id = uuid4()

    groundedness_id = output_assessment_id(turn_id, "groundedness")
    repeated_id = output_assessment_id(turn_id, "groundedness")
    moderation_id = output_assessment_id(turn_id, "post_moderation")

    assert groundedness_id == repeated_id
    assert groundedness_id != moderation_id


@pytest.mark.asyncio
async def test_mock_audit_records_tenant_turn_and_assessment_identity() -> None:
    turn_id = uuid4()
    audit = MockOutputAssessmentAudit()
    record = OutputAssessmentAuditRecord(
        tenant_id="tenant-a",
        conversation_id="conversation-a",
        turn_id=turn_id,
        assessment_id=output_assessment_id(turn_id, "groundedness"),
        assessment_type="groundedness",
        result={"is_grounded": False, "score": 0.2},
    )

    await audit.record(record)

    assert audit.records == [record]


@pytest.mark.asyncio
async def test_audit_adapter_failure_is_best_effort() -> None:
    class FailingAudit:
        async def record(self, assessment: object) -> None:
            del assessment
            raise RuntimeError("audit unavailable")

    from app.langgraph_v2.output_assessments import record_output_assessment

    await record_output_assessment(
        FailingAudit(),
        state={"conversation_id": "conversation-a", "turn_id": str(uuid4())},
        context=SimpleNamespace(tenant_id="tenant-a", current_turn_id=None),
        assessment_type="groundedness",
        result={"failed": True, "error": "evaluator unavailable"},
    )
