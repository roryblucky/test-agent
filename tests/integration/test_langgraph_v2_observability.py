"""End-to-end OpenTelemetry coverage and redaction for the v2 kernel."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from pydantic_ai import Agent

import app.langgraph_v2.observability as observability
import app.langgraph_v2.postgres as postgres_module
from app.langgraph_v2.answer import AnswerOutput, PydanticAIAnswerActor
from app.langgraph_v2.api import register_tracer_routes
from app.langgraph_v2.cancellation import CancellationRepository
from app.langgraph_v2.groundedness import (
    GroundednessOutput,
    PydanticAIGroundednessActor,
)
from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.question_refinement import (
    PydanticAIQuestionRefinementActor,
    V2ResolvedQuery,
)
from app.langgraph_v2.replay import PersistedEventFollower
from app.langgraph_v2.run_events import EventInput, RunEventRepository
from app.services.exceptions import TenantNotFoundError
from tests.integration.test_langgraph_v2_tracer import (
    parse_sse,
    persistent_tracer_app,
)


@dataclass(frozen=True)
class _TelemetryCapture:
    spans: InMemorySpanExporter
    metrics: InMemoryMetricReader


def _install_capture(monkeypatch: pytest.MonkeyPatch) -> _TelemetryCapture:
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
    return _TelemetryCapture(spans=span_exporter, metrics=metric_reader)


def _metric_labels(capture: _TelemetryCapture) -> list[dict[str, Any]]:
    data = capture.metrics.get_metrics_data()
    return [
        dict(point.attributes)
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        for point in metric.data.data_points
    ]


def test_failed_operation_exports_only_safe_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors remain diagnosable without exporting their sensitive messages."""
    capture = _install_capture(monkeypatch)
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

    with observability.observe(
        "provider.invoke",
        run_id=run_id,
        conversation_id=conversation_id,
        attributes={"provider.role": "retrieval"},
    ):
        pass

    spans = capture.spans.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].name == "langgraph_v2.provider.invoke"
    assert spans[0].attributes == {
        "run.id": spans[1].attributes["run.id"],
        "conversation.id": spans[1].attributes["conversation.id"],
        "provider.role": "retrieval",
        "error.type": "RuntimeError",
    }
    assert str(run_id) not in spans[0].attributes["run.id"]
    assert conversation_id not in spans[0].attributes["conversation.id"]
    assert spans[0].attributes["run.id"].startswith("id:")
    assert spans[0].attributes["conversation.id"].startswith("id:")
    assert spans[0].events == ()
    assert secret not in repr(spans[0])

    metrics = capture.metrics.get_metrics_data()
    points = metrics.resource_metrics[0].scope_metrics[0].metrics[0].data.data_points
    assert [point.attributes for point in points] == [
        {"operation": "provider.invoke", "outcome": "error"},
        {"operation": "provider.invoke", "outcome": "ok"},
    ]
    assert all("run.id" not in point.attributes for point in points)
    assert all("conversation.id" not in point.attributes for point in points)


def test_startup_installs_an_exporting_metric_reader_without_a_host_provider(
    langgraph_v2_migrated_database_url: str,
) -> None:
    """The production lifespan records metrics without app configuration."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=langgraph_v2_migrated_database_url),
        ):
            yield

    app = FastAPI(lifespan=lifespan)
    with TestClient(app):
        reader = app.state.langgraph_v2_metric_reader
        assert isinstance(reader, PeriodicExportingMetricReader)
        with observability.observe("execution.run"):
            pass
        assert reader.force_flush()


def test_identifier_redaction_is_stable_for_deployment_key() -> None:
    """Separate process configuration produces the same opaque correlation ID."""
    environment = {"LANGGRAPH_V2_TELEMETRY_KEY": "k" * 32}
    first = observability.IdentifierRedactor.from_environment(environment)
    second = observability.IdentifierRedactor.from_environment(dict(environment))

    assert first.redact("conversation-secret") == second.redact("conversation-secret")
    assert "conversation-secret" not in first.redact("conversation-secret")
    assert "kkkk" not in repr(first)


def test_otlp_metric_exporter_uses_the_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback pipeline can export to an operator-managed OTLP collector."""
    exporter_calls: list[str | None] = []
    exporter = object()
    reader = object()
    monkeypatch.setattr(
        observability,
        "OTLPMetricExporter",
        lambda *, endpoint=None: exporter_calls.append(endpoint) or exporter,
    )
    monkeypatch.setattr(
        observability,
        "PeriodicExportingMetricReader",
        lambda selected: reader if selected is exporter else None,
    )

    configured = observability.build_metric_reader(
        {
            "LANGGRAPH_V2_METRICS_EXPORTER": "otlp",
            "LANGGRAPH_V2_OTLP_METRICS_ENDPOINT": "https://collector.example/v1/metrics",
        }
    )

    assert configured is reader
    assert exporter_calls == ["https://collector.example/v1/metrics"]


def test_existing_application_meter_provider_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup fallback never replaces telemetry configured by the host app."""
    configured_provider = object()
    replacements: list[Any] = []
    monkeypatch.setattr(observability, "_FALLBACK_METRIC_READER", None)
    monkeypatch.setattr(
        observability.metrics,
        "get_meter_provider",
        lambda: configured_provider,
    )
    monkeypatch.setattr(
        observability.metrics,
        "set_meter_provider",
        replacements.append,
    )

    assert observability.ensure_meter_provider() is None
    assert replacements == []


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
    capture = _install_capture(monkeypatch)
    secret_query = "private query credential=do-not-export"
    secret_session_id = "private session credential=do-not-export"
    app = _observed_app(
        langgraph_v2_migrated_database_url,
        secret_query=secret_query,
    )

    with TestClient(app) as client:
        query = client.post(
            "/v2/query/stream",
            headers={"X-Application-Id": "tenant-sensitive"},
            json={"query": secret_query, "sessionId": secret_session_id},
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
        missing_recovery = client.post(
            f"/v2/runs/{uuid4()}/resume/stream",
            headers={"X-Application-Id": "tenant-sensitive"},
        )
        cancellation = client.post(
            f"/v2/runs/{run_id}/cancel",
            headers={"X-Application-Id": "tenant-sensitive"},
        )
        artifact = client.get(
            f"/v2/artifacts/{uuid4()}",
            headers={"X-Application-Id": "tenant-sensitive"},
        )

    assert isinstance(
        app.state.langgraph_v2_metric_reader,
        PeriodicExportingMetricReader,
    )

    assert query.status_code == 200
    assert replay.status_code == 200
    assert recovery.status_code == 409
    assert missing_recovery.status_code == 404
    assert cancellation.status_code == 200
    assert artifact.status_code == 404

    spans = capture.spans.get_finished_spans()
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
    assert secret_session_id not in exported
    assert "private generated answer" not in exported
    assert "tenant-sensitive" not in exported
    assert all(not span.events for span in spans)
    assert {
        span.attributes["http.route"]
        for span in spans
        if span.name == "langgraph_v2.http.request"
    } >= {
        "/v2/query/stream",
        "/v2/runs/{run_id}/resume/stream",
        "/v2/artifacts/{artifact_id}",
    }

    metrics_data = capture.metrics.get_metrics_data()
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


class _UnknownTenantManager:
    def get_providers(self, tenant_id: str) -> None:
        raise TenantNotFoundError(tenant_id)


def test_preflight_rejections_emit_redacted_http_spans(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected requests remain observable before an SSE iterator exists."""
    capture = _install_capture(monkeypatch)
    app = _observed_app(
        langgraph_v2_migrated_database_url,
        secret_query="unused",
    )
    app.state.tenant_manager = _UnknownTenantManager()

    with TestClient(app) as client:
        invalid_query = client.post(
            "/v2/query/stream",
            json={"query": "private invalid query"},
        )
        unknown_tenant = client.post(
            "/v2/query/stream",
            headers={"X-Application-Id": "private-missing-tenant"},
            json={"query": "private rejected query"},
        )
        del app.state.tenant_manager
        missing_replay = client.get(
            f"/v2/runs/{uuid4()}/stream",
            headers={"X-Application-Id": "tenant-a"},
        )
        missing_artifact = client.get(
            f"/v2/artifacts/{uuid4()}",
            headers={"X-Application-Id": "tenant-a"},
        )
        missing_resume = client.post(
            f"/v2/runs/{uuid4()}/resume/stream",
            headers={"X-Application-Id": "tenant-a"},
        )
        missing_cancel = client.post(
            f"/v2/runs/{uuid4()}/cancel",
            headers={"X-Application-Id": "tenant-a"},
        )
        asyncio.run(
            app.state.langgraph_v2_runtime.stop_and_wait_for_checkpoint_boundary()
        )
        shutdown = client.post(
            "/v2/query/stream",
            headers={"X-Application-Id": "tenant-a"},
            json={"query": "private shutdown query"},
        )

    assert invalid_query.status_code == 422
    assert unknown_tenant.status_code == 404
    assert missing_replay.status_code == 404
    assert missing_artifact.status_code == 404
    assert missing_resume.status_code == 404
    assert missing_cancel.status_code == 404
    assert shutdown.status_code == 503
    spans = [
        span
        for span in capture.spans.get_finished_spans()
        if span.name == "langgraph_v2.http.request"
    ]
    routes = [span.attributes["http.route"] for span in spans]
    assert routes.count("/v2/query/stream") == 3
    assert "/v2/runs/{run_id}/stream" in routes
    assert "/v2/artifacts/{artifact_id}" in routes
    assert "/v2/runs/{run_id}/resume/stream" in routes
    assert "/v2/runs/{run_id}/cancel" in routes
    assert all(span.status.status_code.name == "ERROR" for span in spans)
    exported = repr(spans)
    assert "private-missing-tenant" not in exported
    assert "private rejected query" not in exported
    assert "private shutdown query" not in exported


def test_groundedness_setup_failure_is_redacted_and_diagnosable(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor construction failure emits a bounded setup operation."""
    capture = _install_capture(monkeypatch)
    app = _observed_app(
        langgraph_v2_migrated_database_url,
        secret_query="unused",
    )
    monkeypatch.setattr(
        "app.langgraph_v2.api._resolve_groundedness_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("private groundedness credential=never-export")
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            headers={"X-Application-Id": "tenant-a"},
            json={"query": "private setup query"},
        )

    assert response.status_code == 200
    span = next(
        span
        for span in capture.spans.get_finished_spans()
        if span.name == "langgraph_v2.pydantic_ai.setup"
    )
    assert span.attributes["actor.role"] == "groundedness"
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.status.status_code.name == "ERROR"
    exported = repr(span)
    assert "private groundedness" not in exported
    assert "private setup query" not in exported


@pytest.mark.asyncio
async def test_lifespan_shutdown_interruption_emits_outcome(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown recovery reports the durable interruption it applies."""
    capture = _install_capture(monkeypatch)
    monkeypatch.setattr(postgres_module, "_INSTANCE_ID", "test-shutdown-instance")
    app = _observed_app(
        langgraph_v2_migrated_database_url,
        secret_query="unused",
    )

    async with app.router.lifespan_context(app):
        await RunEventRepository(app.state.langgraph_v2_postgres_pool).create_run(
            tenant_id="private-shutdown-tenant",
            run_id=uuid4(),
            conversation_id="private-shutdown-conversation",
            owner_instance_id="test-shutdown-instance",
        )

    span = next(
        span
        for span in capture.spans.get_finished_spans()
        if span.name == "langgraph_v2.recovery.interrupt_shutdown"
    )
    assert span.attributes["recovery.operation"] == "interrupt_shutdown"
    assert span.attributes["operation.outcome"] == "completed"
    assert span.attributes["run.status"] == "interrupted"
    assert span.attributes["recovery.run_count"] == 1
    exported = repr(span)
    assert "private-shutdown-tenant" not in exported
    assert "private-shutdown-conversation" not in exported


class _ResumeGraph:
    async def ainvoke(
        self,
        state: Any,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        del state, config
        return {
            "events": [
                {
                    "event_key": "recovery:completed:2",
                    "type": "done",
                    "data": {"source": "recovered"},
                    "sequence": 1,
                }
            ]
        }


async def _seed_stale_run(database_url: str) -> UUID:
    async with AsyncConnectionPool(database_url, min_size=1, max_size=2) as pool:
        repository = RunEventRepository(pool)
        run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-recovery",
            owner_instance_id="expired-owner",
        )
        await repository.update_checkpoint_pointer(
            tenant_id=run.tenant_id,
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            checkpoint_id="checkpoint-safe-id",
            checkpoint_ns="checkpoint-safe-namespace",
        )
        async with pool.connection() as connection:
            await connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = %s AND run_id = %s
                """,
                (run.tenant_id, run.run_id),
            )
        return run.run_id


def test_successful_recovery_propagates_one_trace_to_execution_and_http(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed task and its SSE follower remain children of recovery."""
    capture = _install_capture(monkeypatch)
    run_id = asyncio.run(_seed_stale_run(langgraph_v2_migrated_database_url))
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        _ResumeGraph(),
        resume_enabled=True,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v2/runs/{run_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a"},
        )

    assert response.status_code == 200
    assert [event["type"] for event in parse_sse(response.text)] == ["error", "done"]
    spans = capture.spans.get_finished_spans()
    recovery = next(
        span for span in spans if span.name == "langgraph_v2.recovery.resume"
    )
    execution = next(
        span for span in spans if span.name == "langgraph_v2.execution.run"
    )
    http = next(
        span
        for span in spans
        if span.name == "langgraph_v2.http.request"
        and span.attributes["http.route"] == "/v2/runs/{run_id}/resume/stream"
    )
    stream = next(
        span
        for span in spans
        if span.name == "langgraph_v2.http.stream"
        and span.attributes["http.route"] == "/v2/runs/{run_id}/resume/stream"
    )
    replay = next(span for span in spans if span.name == "langgraph_v2.replay.follow")
    assert recovery.attributes["execution.epoch"] == 2
    assert recovery.attributes["run.status"] == "running"
    assert (
        recovery.context.trace_id
        == execution.context.trace_id
        == http.context.trace_id
        == stream.context.trace_id
        == replay.context.trace_id
    )
    assert recovery.parent.span_id == http.context.span_id
    assert execution.parent.span_id == recovery.context.span_id
    assert stream.parent.span_id == recovery.context.span_id
    assert replay.parent.span_id == stream.context.span_id
    assert execution.attributes["operation.outcome"] == "completed"
    assert execution.attributes["run.status"] == "completed"
    assert {"operation": "execution.run", "outcome": "completed"} in _metric_labels(
        capture
    )


class _CancelDuringRefinement:
    def __init__(self) -> None:
        self.app: FastAPI | None = None

    async def refine(self, query: str, history: Any) -> V2ResolvedQuery:
        del history
        assert self.app is not None
        pool = self.app.state.langgraph_v2_postgres_pool
        async with pool.connection() as connection:
            result = await connection.execute(
                """
                SELECT run_id FROM langgraph_v2.runs
                WHERE tenant_id = 'tenant-a' AND status = 'running'
                ORDER BY created_at DESC LIMIT 1
                """
            )
            run_id = (await result.fetchone())[0]
        await CancellationRepository(pool).request(
            tenant_id="tenant-a",
            run_id=run_id,
        )
        return V2ResolvedQuery(original_query=query, standalone_query=query)


def test_cancellation_application_is_correlated_and_redacted(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cooperative cancellation emits its durable application span."""
    capture = _install_capture(monkeypatch)
    actor = _CancelDuringRefinement()
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=actor,
    )
    actor.app = app
    secret_query = "cancel private query credential=never-export"

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            headers={"X-Application-Id": "tenant-a"},
            json={"query": secret_query},
        )

    assert response.status_code == 200
    assert parse_sse(response.text)[-1]["type"] == "stopped"
    spans = capture.spans.get_finished_spans()
    applied = next(
        span for span in spans if span.name == "langgraph_v2.cancellation.apply"
    )
    execution = next(
        span for span in spans if span.name == "langgraph_v2.execution.run"
    )
    assert applied.context.trace_id == execution.context.trace_id
    assert applied.parent.span_id == execution.context.span_id
    assert applied.attributes["cancellation.operation"] == "apply"
    assert applied.attributes["run.status"] == "cancelled"
    assert execution.attributes["operation.outcome"] == "cancelled"
    assert execution.attributes["run.status"] == "cancelled"
    assert {"operation": "execution.run", "outcome": "cancelled"} in _metric_labels(
        capture
    )
    assert secret_query not in repr(spans)


class _FailingGraph:
    async def ainvoke(
        self,
        state: Any,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        del state, config
        raise RuntimeError("private provider response credential=never-export")


def test_consumed_execution_error_has_failed_span_and_metric_outcomes(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error converted to SSE is still an error in traces and metrics."""
    capture = _install_capture(monkeypatch)
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        _FailingGraph(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            headers={"X-Application-Id": "tenant-a"},
            json={"query": "private failure query"},
        )

    assert response.status_code == 200
    assert parse_sse(response.text)[-1]["type"] == "error"
    spans = capture.spans.get_finished_spans()
    execution = next(
        span for span in spans if span.name == "langgraph_v2.execution.run"
    )
    assert execution.attributes["operation.outcome"] == "failed"
    assert execution.attributes["run.status"] == "failed"
    assert execution.attributes["error.type"] == "RuntimeError"
    assert execution.status.status_code.name == "ERROR"
    assert {"operation": "execution.run", "outcome": "failed"} in _metric_labels(
        capture
    )
    exported = repr(spans)
    assert "private provider response" not in exported
    assert "private failure query" not in exported


@pytest.mark.asyncio
async def test_expired_claim_interruption_is_correlated_and_redacted(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follower-driven expiry remains visible without tenant or Event data."""
    capture = _install_capture(monkeypatch)
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=2,
    ) as pool:
        repository = RunEventRepository(pool)
        run = await repository.create_run(
            tenant_id="tenant-expired-secret",
            run_id=uuid4(),
            conversation_id="conversation-expired-secret",
            owner_instance_id="expired-owner",
        )
        await repository.append_event(
            tenant_id=run.tenant_id,
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            event=EventInput(
                event_key="phase:query:step_completed:1",
                type="step_completed",
                data={"raw_artifact": "never-export-this-artifact"},
            ),
        )
        async with pool.connection() as connection:
            await connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = %s AND run_id = %s
                """,
                (run.tenant_id, run.run_id),
            )
        follower = PersistedEventFollower(
            repository,
            LiveEventWakeups(),
            poll_interval_seconds=0.001,
        ).follow(
            tenant_id=run.tenant_id,
            run_id=run.run_id,
            after_sequence=1,
        )
        interrupted = await anext(follower)
        with pytest.raises(StopAsyncIteration):
            await anext(follower)

    assert interrupted.event_key == "lifecycle:interrupted:1"
    spans = capture.spans.get_finished_spans()
    recovery = next(
        span for span in spans if span.name == "langgraph_v2.recovery.interrupt_expired"
    )
    replay = next(span for span in spans if span.name == "langgraph_v2.replay.follow")
    assert recovery.context.trace_id == replay.context.trace_id
    assert recovery.parent.span_id == replay.context.span_id
    assert recovery.attributes["run.id"].startswith("id:")
    assert recovery.attributes["execution.epoch"] == 1
    assert recovery.attributes["recovery.operation"] == "interrupt_expired"
    assert recovery.attributes["run.status"] == "interrupted"
    exported = repr(spans)
    assert "tenant-expired-secret" not in exported
    assert "conversation-expired-secret" not in exported
    assert "never-export-this-artifact" not in exported
