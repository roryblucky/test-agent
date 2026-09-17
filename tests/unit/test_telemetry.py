"""Bounded application telemetry helpers."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import Tracer

import app.core.telemetry as telemetry_module


def test_specialist_definition_pin_is_recorded_without_definition_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def get_tracer(_name: str) -> Tracer:
        return provider.get_tracer("test")

    monkeypatch.setattr(
        telemetry_module.trace,
        "get_tracer",
        get_tracer,
    )

    telemetry_module.record_specialist_definition_pin(
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
        pin="a" * 64,
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "specialist.task.accepted"
    assert dict(spans[0].attributes or {}) == {
        "tenant.id": "tenant-a",
        "request.id": "request-1",
        "task.id": "task-1",
        "specialist.definition.pin": "a" * 64,
    }
