"""Bounded application telemetry helpers."""

from __future__ import annotations

from typing import Any

import pytest

import app.core.telemetry as telemetry_module


def test_specialist_definition_pin_is_recorded_without_definition_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class Span:
        def add_event(self, name: str, attributes: dict[str, Any]) -> None:
            events.append((name, attributes))

    monkeypatch.setattr(telemetry_module.trace, "get_current_span", Span)

    telemetry_module.record_specialist_definition_pin(
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
        pin="a" * 64,
    )

    assert events == [
        (
            "specialist.task.accepted",
            {
                "tenant.id": "tenant-a",
                "request.id": "request-1",
                "task.id": "task-1",
                "specialist.definition.pin": "a" * 64,
            },
        )
    ]
