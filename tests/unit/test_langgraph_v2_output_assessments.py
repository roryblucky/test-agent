from __future__ import annotations

from uuid import uuid4

import pytest

from app.langgraph_v2.output_assessments import (
    BigQueryOutputAssessmentAudit,
    MockOutputAssessmentAudit,
    OutputAssessmentAuditRecord,
    OutputAssessmentScope,
    output_assessment_id,
)


class _BigQueryClient:
    def __init__(self) -> None:
        self.rows: list[tuple[str, list[dict[str, object]], list[str]]] = []

    def insert_rows_json(
        self,
        table_ref: str,
        rows: list[dict[str, object]],
        *,
        row_ids: list[str],
    ) -> list[object]:
        self.rows.append((table_ref, rows, row_ids))
        return []

    def close(self) -> None:
        pass


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
async def test_bigquery_audit_writes_assessment_schema_with_stable_row_id() -> None:
    turn_id = uuid4()
    assessment_id = output_assessment_id(turn_id, "groundedness")
    client = _BigQueryClient()
    audit = BigQueryOutputAssessmentAudit(project_id="project-a", client=client)

    await audit.record(
        OutputAssessmentAuditRecord(
            tenant_id="tenant-a",
            conversation_id="conversation-a",
            turn_id=turn_id,
            assessment_id=assessment_id,
            assessment_type="groundedness",
            result={"is_grounded": True, "score": 1.0},
        )
    )

    table_ref, rows, row_ids = client.rows[0]
    assert table_ref == "project-a.audit_logs.kms_output_assessments"
    assert row_ids == [assessment_id]
    assert rows[0]["tenant_id"] == "tenant-a"
    assert rows[0]["turn_id"] == str(turn_id)
    assert rows[0]["result"] == {"is_grounded": True, "score": 1.0}


@pytest.mark.asyncio
async def test_audit_adapter_failure_is_best_effort() -> None:
    class FailingAudit:
        async def record(self, assessment: object) -> None:
            del assessment
            raise RuntimeError("audit unavailable")

    from app.langgraph_v2.output_assessments import record_output_assessment

    await record_output_assessment(
        FailingAudit(),
        scope=OutputAssessmentScope(
            tenant_id="tenant-a",
            conversation_id="conversation-a",
            turn_id=uuid4(),
        ),
        assessment_type="groundedness",
        result={"failed": True, "error": "evaluator unavailable"},
    )
