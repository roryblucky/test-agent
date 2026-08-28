"""Redacted OpenTelemetry primitives for the v2 orchestration kernel."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from time import perf_counter
from typing import Any
from uuid import UUID

from opentelemetry import metrics, trace
from opentelemetry.metrics import Meter
from opentelemetry.trace import Span, Status, StatusCode, Tracer

_SAFE_ATTRIBUTE_KEYS = frozenset(
    {
        "actor.role",
        "cancellation.operation",
        "checkpoint.exact",
        "checkpoint.exact_parent",
        "conversation.id",
        "error.type",
        "execution.epoch",
        "graph.phase",
        "http.request.method",
        "http.route",
        "persistence.kind",
        "provider.role",
        "recovery.operation",
        "replay.mode",
        "run.id",
        "run.status",
    }
)


class V2Telemetry:
    """One tracer and low-cardinality meter with a constrained span surface."""

    def __init__(self, *, tracer: Tracer, meter: Meter) -> None:
        self.tracer = tracer
        self.operations = meter.create_counter(
            "langgraph_v2.operations",
            description="Completed v2 operations by bounded operation and outcome.",
        )
        self.duration = meter.create_histogram(
            "langgraph_v2.operation.duration",
            unit="s",
            description="V2 operation duration by bounded operation and outcome.",
        )


_TELEMETRY = V2Telemetry(
    tracer=trace.get_tracer("app.langgraph_v2"),
    meter=metrics.get_meter("app.langgraph_v2"),
)


@contextmanager
def observe(
    operation: str,
    *,
    run_id: UUID | str | None = None,
    conversation_id: str | None = None,
    execution_epoch: int | None = None,
    attributes: Mapping[str, str | bool | int | float] | None = None,
) -> Iterator[Span]:
    """Emit a redacted span and bounded metrics for one fixed operation.

    Exception values are deliberately not recorded.  They can contain queries,
    credentials, model output, Evidence, or provider payloads.
    """
    span_attributes: dict[str, str | bool | int | float] = {}
    if run_id is not None:
        span_attributes["run.id"] = str(run_id)
    if conversation_id is not None:
        span_attributes["conversation.id"] = conversation_id
    if execution_epoch is not None:
        span_attributes["execution.epoch"] = execution_epoch
    if attributes:
        unknown = set(attributes) - _SAFE_ATTRIBUTE_KEYS
        if unknown:
            raise ValueError(f"unsafe telemetry attribute keys: {sorted(unknown)}")
        span_attributes.update(attributes)

    started_at = perf_counter()
    outcome = "ok"
    with _TELEMETRY.tracer.start_as_current_span(
        f"langgraph_v2.{operation}",
        attributes=span_attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException as error:
            outcome = "error"
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            metric_attributes = {"operation": operation, "outcome": outcome}
            _TELEMETRY.operations.add(1, metric_attributes)
            _TELEMETRY.duration.record(
                perf_counter() - started_at,
                metric_attributes,
            )


def safe_span_attribute(span: Span, key: str, value: Any) -> None:
    """Set one late-bound attribute only when it belongs to the safe schema."""
    if key not in _SAFE_ATTRIBUTE_KEYS:
        raise ValueError(f"unsafe telemetry attribute key: {key}")
    if isinstance(value, UUID):
        value = str(value)
    if not isinstance(value, (str, bool, int, float)):
        raise TypeError(f"unsupported telemetry attribute type: {type(value).__name__}")
    span.set_attribute(key, value)
