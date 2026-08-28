"""End-to-end OpenTelemetry coverage and redaction for the v2 kernel."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel
from pydantic_ai import Agent

import app.langgraph_v2.observability as observability
from app.langgraph_v2.answer import AnswerOutput, PydanticAIAnswerActor
from app.langgraph_v2.api import register_tracer_routes
from app.langgraph_v2.groundedness import (
    GroundednessOutput,
    PydanticAIGroundednessActor,
)
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.question_refinement import (
    PydanticAIQuestionRefinementActor,
    V2ResolvedQuery,
)


def test_failed_operation_exports_only_safe_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors remain diagnosable without exporting their sensitive messages."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    telemetry = observability.V2Telemetry(
        tracer=tracer_provider.get_tracer("test"),
        meter=meter_provider.get_meter("test"),
    )
    monkeypatch.setattr(observability, "_TELEMETRY", telemetry)
    run_id = uuid4()
    conversation_id = "conversation-safe-id"
    secret = "query=private-question credential=top-secret"

    with pytest.raises(RuntimeError, match="private-question"):
        with observability.observe(
            "provider.invoke",
            run_id=run_id,
            conversation_id=conversation_id,
            attributes={"provider.role": "retrieval"},
        ):
            raise RuntimeError(secret)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "langgraph_v2.provider.invoke"
    assert spans[0].attributes == {
        "run.id": str(run_id),
        "conversation.id": conversation_id,
        "provider.role": "retrieval",
        "error.type": "RuntimeError",
    }
    assert spans[0].events == ()
    assert secret not in repr(spans[0])

    metrics = metric_reader.get_metrics_data()
    points = metrics.resource_metrics[0].scope_metrics[0].metrics[0].data.data_points
    assert [point.attributes for point in points] == [
        {"operation": "provider.invoke", "outcome": "error"}
    ]
    assert all("run.id" not in point.attributes for point in points)
    assert all("conversation.id" not in point.attributes for point in points)


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 2
    output_tokens: int = 3


class _AgentResult:
    def __init__(self, output: BaseModel) -> None:
        self.output = output

    def usage(self) -> _Usage:
        return _Usage()


class _Agent:
    def __init__(self, output: BaseModel) -> None:
        self._output = output

    async def run(self, prompt: str, **kwargs: Any) -> _AgentResult:
        del prompt, kwargs
        return _AgentResult(self._output)


def _actor(agent: _Agent) -> Agent[Any, Any]:
    return cast(Agent[Any, Any], agent)


def _observed_app(database_url: str, *, secret_query: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=database_url),
        ):
            yield

    app = FastAPI(lifespan=lifespan)
    register_tracer_routes(
        app,
        enabled=True,
        replay_enabled=True,
        resume_enabled=True,
        cancellation_enabled=True,
        refinement_actor=PydanticAIQuestionRefinementActor(
            _actor(
                _Agent(
                    V2ResolvedQuery(
                        original_query=secret_query,
                        standalone_query=secret_query,
                    )
                )
            )
        ),
        answer_actor=PydanticAIAnswerActor(
            _actor(_Agent(AnswerOutput(answer="private generated answer [1]")))
        ),
        groundedness_actor=PydanticAIGroundednessActor(
            _actor(_Agent(GroundednessOutput(is_grounded=True, score=1.0)))
        ),
    )
    return app


def test_complete_v2_path_emits_correlated_redacted_telemetry(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public path is observable without exporting business payloads."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    monkeypatch.setattr(
        observability,
        "_TELEMETRY",
        observability.V2Telemetry(
            tracer=tracer_provider.get_tracer("test"),
            meter=meter_provider.get_meter("test"),
        ),
    )
    secret_query = "private query credential=do-not-export"
    app = _observed_app(
        langgraph_v2_migrated_database_url,
        secret_query=secret_query,
    )

    with TestClient(app) as client:
        query = client.post(
            "/v2/query/stream",
            headers={"X-Application-Id": "tenant-sensitive"},
            json={"query": secret_query, "conversationId": "conversation-telemetry"},
        )
        run_id = query.headers["x-run-id"]
        replay = client.get(
            f"/v2/runs/{run_id}/stream",
            headers={"X-Application-Id": "tenant-sensitive"},
        )
        recovery = client.post(
            f"/v2/runs/{run_id}/resume/stream",
            headers={"X-Application-Id": "tenant-sensitive"},
        )
        cancellation = client.post(
            f"/v2/runs/{run_id}/cancel",
            headers={"X-Application-Id": "tenant-sensitive"},
        )

    assert query.status_code == 200
    assert replay.status_code == 200
    assert recovery.status_code == 409
    assert cancellation.status_code == 200

    spans = span_exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert {
        "langgraph_v2.http.request",
        "langgraph_v2.execution.run",
        "langgraph_v2.graph.phase",
        "langgraph_v2.pydantic_ai.invoke",
        "langgraph_v2.provider.invoke",
        "langgraph_v2.checkpoint.read",
        "langgraph_v2.checkpoint.write",
        "langgraph_v2.persistence.phase_result",
        "langgraph_v2.persistence.event_batch",
        "langgraph_v2.replay.follow",
        "langgraph_v2.recovery.resume",
        "langgraph_v2.cancellation.request",
    } <= names
    assert {
        span.attributes["graph.phase"]
        for span in spans
        if span.name == "langgraph_v2.graph.phase"
    } == {
        "query",
        "pre_moderation",
        "question_refinement",
        "retrieval",
        "reranking",
        "answer",
        "groundedness",
        "post_moderation",
        "finalization",
    }
    assert {
        span.attributes["actor.role"]
        for span in spans
        if span.name == "langgraph_v2.pydantic_ai.invoke"
    } == {"question_refinement", "answer", "groundedness"}
    query_http_span = next(
        span
        for span in spans
        if span.name == "langgraph_v2.http.request"
        and span.attributes["http.route"] == "/v2/query/stream"
    )
    query_trace_id = query_http_span.context.trace_id
    assert all(
        span.context.trace_id == query_trace_id
        for span in spans
        if span.name
        in {
            "langgraph_v2.execution.run",
            "langgraph_v2.graph.phase",
            "langgraph_v2.pydantic_ai.invoke",
            "langgraph_v2.provider.invoke",
            "langgraph_v2.checkpoint.read",
            "langgraph_v2.checkpoint.write",
            "langgraph_v2.persistence.phase_result",
            "langgraph_v2.persistence.event_batch",
        }
    )

    exported = repr(spans)
    assert secret_query not in exported
    assert "private generated answer" not in exported
    assert "tenant-sensitive" not in exported
    assert all(not span.events for span in spans)

    metrics_data = metric_reader.get_metrics_data()
    points = [
        point
        for resource in metrics_data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        for point in metric.data.data_points
    ]
    assert points
    assert all(set(point.attributes) == {"operation", "outcome"} for point in points)
    assert all("run.id" not in point.attributes for point in points)
    assert all("conversation.id" not in point.attributes for point in points)
