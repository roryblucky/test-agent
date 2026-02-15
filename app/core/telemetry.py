"""Telemetry service — OpenTelemetry configuration and instrumentation helpers.

Provides a unified way to configure tracing and a decorator to instrument
functions with spans.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.semconv.resource import ResourceAttributes

P = ParamSpec("P")
R = TypeVar("R")


class TelemetryService:
    """Configures OpenTelemetry tracing."""

    def __init__(self, service_name: str, version: str = "0.1.0") -> None:
        self.resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: service_name,
                ResourceAttributes.SERVICE_VERSION: version,
            }
        )
        self.provider = TracerProvider(resource=self.resource)

        # Configure exporter based on environment
        # In a real app, check env vars (e.g., K_SERVICE) to detect GCP
        try:
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Use GCP Trace Exporter for production/GCP envs
            exporter = CloudTraceSpanExporter()
            processor = BatchSpanProcessor(exporter)
        except ImportError:
            # Fallback to Console for local dev if GCP libs missing
            processor = SimpleSpanProcessor(ConsoleSpanExporter())

        self.provider.add_span_processor(processor)

        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer(service_name, version)

        # Determine if running in Google Cloud to set up logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure Google Cloud Logging with trace correlation."""
        try:
            import google.cloud.logging
            from google.cloud.logging.handlers import CloudLoggingHandler

            client = google.cloud.logging.Client()
            handler = CloudLoggingHandler(client)

            # Add GCP handler to root logger
            # effectively replaces console/stream handler in k8s
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)

        except ImportError:
            pass  # Fallback to standard logging (console)

    def instrument(self) -> None:
        """Apply auto-instrumentation (future expansion)."""
        # Could add FastAPIInstrumentor here


def trace_span(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to wrap a function execution in an OpenTelemetry span.

    Args:
        name: Optional span name. If not provided, uses the function name.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as span:
                try:
                    return await func(*args, **kwargs)  # type: ignore
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as span:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator
