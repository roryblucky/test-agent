from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
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
        self.call_thread_ids: list[int] = []
        self.closed = False

    def insert_rows_json(
        self,
        table_ref: str,
        rows: list[dict[str, object]],
        *,
        row_ids: list[str],
    ) -> list[object]:
        self.call_thread_ids.append(threading.get_ident())
        self.rows.append((table_ref, rows, row_ids))
        return []

    def close(self) -> None:
        self.call_thread_ids.append(threading.get_ident())
        self.closed = True


def _scope(
    request_id: object,
    *,
    tenant_id: str = "tenant-a",
    conversation_id: str = "conversation-a",
) -> OutputAssessmentScope:
    return OutputAssessmentScope(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        request_id=str(request_id),
    )


def test_output_assessment_identity_is_stable_and_type_specific() -> None:
    request_id = uuid4()
    scope = _scope(request_id)

    groundedness_id = output_assessment_id(scope, "groundedness")
    repeated_id = output_assessment_id(scope, "groundedness")
    moderation_id = output_assessment_id(scope, "post_moderation")

    assert groundedness_id == repeated_id
    assert groundedness_id != moderation_id
    assert groundedness_id != output_assessment_id(
        _scope(request_id, conversation_id="conversation-b"),
        "groundedness",
    )
    assert groundedness_id != output_assessment_id(
        _scope(request_id, tenant_id="tenant-b"),
        "groundedness",
    )


@pytest.mark.asyncio
async def test_mock_audit_records_request_and_scoped_assessment_identity() -> None:
    request_id = uuid4()
    scope = _scope(request_id)
    audit = MockOutputAssessmentAudit()
    record = OutputAssessmentAuditRecord(
        tenant_id="tenant-a",
        conversation_id="conversation-a",
        request_id=str(request_id),
        assessment_id=output_assessment_id(scope, "groundedness"),
        assessment_type="groundedness",
        result={"is_grounded": False, "score": 0.2},
    )

    await audit.record(record)

    assert audit.records == [record]


@pytest.mark.asyncio
async def test_bigquery_audit_writes_assessment_schema_with_stable_row_id() -> None:
    event_loop_thread_id = threading.get_ident()
    request_id = uuid4()
    assessment_id = output_assessment_id(_scope(request_id), "groundedness")
    client = _BigQueryClient()
    audit = BigQueryOutputAssessmentAudit(project_id="project-a", client=client)

    await audit.record(
        OutputAssessmentAuditRecord(
            tenant_id="tenant-a",
            conversation_id="conversation-a",
            request_id=str(request_id),
            assessment_id=assessment_id,
            assessment_type="groundedness",
            result={"is_grounded": True, "score": 1.0},
        )
    )

    table_ref, rows, row_ids = client.rows[0]
    assert table_ref == "project-a.audit_logs.kms_output_assessments"
    assert row_ids == [assessment_id]
    assert rows[0]["tenant_id"] == "tenant-a"
    assert rows[0]["request_id"] == str(request_id)
    assert rows[0]["result"] == {"is_grounded": True, "score": 1.0}
    assert len(client.call_thread_ids) == 1
    assert client.call_thread_ids[0] != event_loop_thread_id

    await audit.close()

    assert client.closed
    assert client.call_thread_ids[1] != event_loop_thread_id


@pytest.mark.asyncio
async def test_bigquery_audit_closes_client_when_table_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _BigQueryClient()

    class FailingSetupAudit(BigQueryOutputAssessmentAudit):
        def _ensure_table(self, bigquery: Any, client: Any) -> None:
            del bigquery, client
            raise RuntimeError("table setup failed")

    def client_factory(*, project: str) -> _BigQueryClient:
        assert project == "project-a"
        return client

    def import_bigquery(_module_name: str) -> SimpleNamespace:
        return SimpleNamespace(Client=client_factory)

    monkeypatch.setattr(
        "app.langgraph_v2.output_assessments.importlib.import_module",
        import_bigquery,
    )
    request_id = uuid4()
    scope = _scope(request_id)
    audit = FailingSetupAudit(project_id="project-a")

    with pytest.raises(RuntimeError, match="table setup failed"):
        await audit.record(
            OutputAssessmentAuditRecord(
                tenant_id="tenant-a",
                conversation_id="conversation-a",
                request_id=str(request_id),
                assessment_id=output_assessment_id(scope, "groundedness"),
                assessment_type="groundedness",
                result={"is_grounded": True},
            )
        )

    assert client.closed


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
            request_id=str(uuid4()),
        ),
        assessment_type="groundedness",
        result={"failed": True, "error": "evaluator unavailable"},
    )
