"""Redacted OpenTelemetry primitives for the v2 orchestration kernel."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from hmac import new as hmac_new
from secrets import token_bytes
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import UUID

from opentelemetry import context as otel_context
from opentelemetry import metrics, trace
from opentelemetry.context import Context
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
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
        "operation.outcome",
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
_METER_PROVIDER_LOCK = Lock()
_FALLBACK_METRIC_READER: InMemoryMetricReader | None = None
_IDENTIFIER_KEY = token_bytes(32)
_BOUNDED_OUTCOMES = frozenset(
    {"ok", "error", "completed", "failed", "cancelled", "fenced"}
)


@dataclass
class OperationObservation:
    """Per-invocation state independent of provider span object reuse."""

    span: Span
    outcome: str = "ok"


def ensure_meter_provider() -> InMemoryMetricReader | None:
    """Install a local SDK reader only when the process has no meter provider.

    An application-configured provider always wins.  The fallback makes v2
    metrics real and inspectable without requiring an external collector.
    """
    global _FALLBACK_METRIC_READER
    with _METER_PROVIDER_LOCK:
        if _FALLBACK_METRIC_READER is not None:
            return _FALLBACK_METRIC_READER
        provider = metrics.get_meter_provider()
        if type(provider).__name__ != "_ProxyMeterProvider":
            return None
        reader = InMemoryMetricReader()
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        _FALLBACK_METRIC_READER = reader
        return reader


@contextmanager
def observe(
    operation: str,
    *,
    run_id: UUID | str | None = None,
    conversation_id: str | None = None,
    execution_epoch: int | None = None,
    attributes: Mapping[str, str | bool | int | float] | None = None,
    parent_context: Context | None = None,
) -> Iterator[OperationObservation]:
    """Emit a redacted span and bounded metrics for one fixed operation.

    Exception values are deliberately not recorded.  They can contain queries,
    credentials, model output, Evidence, or provider payloads.
    """
    span_attributes: dict[str, str | bool | int | float] = {}
    if run_id is not None:
        span_attributes["run.id"] = redacted_identifier(run_id)
    if conversation_id is not None:
        span_attributes["conversation.id"] = redacted_identifier(conversation_id)
    if execution_epoch is not None:
        span_attributes["execution.epoch"] = execution_epoch
    if attributes:
        unknown = set(attributes) - _SAFE_ATTRIBUTE_KEYS
        if unknown:
            raise ValueError(f"unsafe telemetry attribute keys: {sorted(unknown)}")
        if {"run.id", "conversation.id"} & set(attributes):
            raise ValueError("identifier attributes must use the redacting arguments")
        span_attributes.update(attributes)

    started_at = perf_counter()
    with _TELEMETRY.tracer.start_as_current_span(
        f"langgraph_v2.{operation}",
        context=parent_context,
        attributes=span_attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        observation = OperationObservation(span=span)
        try:
            yield observation
        except BaseException as error:
            observation.outcome = "error"
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            metric_attributes = {
                "operation": operation,
                "outcome": observation.outcome,
            }
            _TELEMETRY.operations.add(1, metric_attributes)
            _TELEMETRY.duration.record(
                perf_counter() - started_at,
                metric_attributes,
            )


def set_operation_outcome(
    observation: OperationObservation,
    outcome: str,
) -> None:
    """Record a bounded outcome when a control boundary consumes an error."""
    if outcome not in _BOUNDED_OUTCOMES:
        raise ValueError(f"unsupported telemetry outcome: {outcome}")
    observation.outcome = outcome
    observation.span.set_attribute("operation.outcome", outcome)
    if outcome in {"failed", "fenced"}:
        observation.span.set_status(Status(StatusCode.ERROR))


def redacted_identifier(value: UUID | str) -> str:
    """Return a process-stable opaque identifier without exporting caller input."""
    digest = hmac_new(
        _IDENTIFIER_KEY,
        str(value).encode("utf-8"),
        sha256,
    ).hexdigest()
    return f"id:{digest}"


def context_for_span(observation: OperationObservation) -> Context:
    """Capture one safe propagation handle without serializing span content."""
    return trace.set_span_in_context(observation.span)


@contextmanager
def activate_context(context: Context) -> Iterator[None]:
    """Explicitly propagate a captured trace context into a new async task."""
    token = otel_context.attach(context)
    try:
        yield
    finally:
        otel_context.detach(token)


def safe_span_attribute(
    observation: OperationObservation,
    key: str,
    value: Any,
) -> None:
    """Set one late-bound attribute only when it belongs to the safe schema."""
    if key not in _SAFE_ATTRIBUTE_KEYS:
        raise ValueError(f"unsafe telemetry attribute key: {key}")
    if isinstance(value, UUID):
        value = str(value)
    if key in {"run.id", "conversation.id"}:
        value = redacted_identifier(str(value))
    if not isinstance(value, (str, bool, int, float)):
        raise TypeError(f"unsupported telemetry attribute type: {type(value).__name__}")
    observation.span.set_attribute(key, value)
