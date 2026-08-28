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
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from pydantic_ai import Agent

import app.langgraph_v2.observability as observability
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

    spans = capture.spans.get_finished_spans()
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

    metrics = capture.metrics.get_metrics_data()
    points = metrics.resource_metrics[0].scope_metrics[0].metrics[0].data.data_points
    assert [point.attributes for point in points] == [
        {"operation": "provider.invoke", "outcome": "error"}
    ]
    assert all("run.id" not in point.attributes for point in points)
    assert all("conversation.id" not in point.attributes for point in points)


def test_startup_installs_a_real_metric_reader_without_an_external_collector(
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
        assert isinstance(reader, InMemoryMetricReader)
        with observability.observe("execution.run"):
            pass
        metrics_data = reader.get_metrics_data()

    assert any(
        point.attributes == {"operation": "execution.run", "outcome": "ok"}
        for resource in metrics_data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        for point in metric.data.data_points
    )


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

    assert isinstance(
        app.state.langgraph_v2_metric_reader,
        InMemoryMetricReader,
    )

    assert query.status_code == 200
    assert replay.status_code == 200
    assert recovery.status_code == 409
    assert cancellation.status_code == 200

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
    assert "private generated answer" not in exported
    assert "tenant-sensitive" not in exported
    assert all(not span.events for span in spans)

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
    replay = next(span for span in spans if span.name == "langgraph_v2.replay.follow")
    assert recovery.attributes["execution.epoch"] == 2
    assert recovery.attributes["run.status"] == "running"
    assert (
        recovery.context.trace_id
        == execution.context.trace_id
        == http.context.trace_id
        == replay.context.trace_id
    )
    assert execution.parent.span_id == recovery.context.span_id
    assert http.parent.span_id == recovery.context.span_id
    assert replay.parent.span_id == http.context.span_id


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
    assert secret_query not in repr(spans)


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
    assert recovery.attributes == {
        "run.id": str(run.run_id),
        "execution.epoch": 1,
        "recovery.operation": "interrupt_expired",
        "run.status": "interrupted",
    }
    exported = repr(spans)
    assert "tenant-expired-secret" not in exported
    assert "conversation-expired-secret" not in exported
    assert "never-export-this-artifact" not in exported
