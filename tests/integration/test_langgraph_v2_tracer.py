import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import reload
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

from app.api.schemas import QueryResponse
from app.langgraph_v2.api import TracerGraph, register_tracer_routes
from app.langgraph_v2.graph import TracerState
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.run_events import RunEventRepository
from app.services.events import EventEmitter

FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "langgraph_v2" / "v1_minimal_wire.json"
)


def parse_sse(response_text: str) -> list[dict]:
    return [
        json.loads(frame.removeprefix("data: "))
        for frame in response_text.strip().split("\n\n")
    ]


def persistent_tracer_app(
    database_url: str,
    graph: TracerGraph | None = None,
) -> FastAPI:
    """Create the test-only tracer with its real application database pool."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=database_url),
        ):
            yield

    app = FastAPI(lifespan=lifespan)
    register_tracer_routes(app, enabled=True, graph=graph)
    return app


@pytest.mark.asyncio
async def test_captured_fixture_matches_the_legacy_wire_implementation() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    request = fixture["request"]
    emitter = EventEmitter()
    await emitter.emit_step_start("query")
    await emitter.emit_step_completed("query", {"query": request["query"]})
    await emitter.emit_step_start("finalization")
    await emitter.emit_step_completed("finalization", {"status": "completed"})
    response = QueryResponse.model_validate(
        {
            "query": request["query"],
            "sessionId": request["sessionId"],
            "metadata": {"steps_executed": ["query", "finalization"]},
        }
    )
    await emitter.emit_done(response.model_dump())

    legacy_frames = [line async for line in emitter]

    assert [json.loads(line.removeprefix("data: ")) for line in legacy_frames] == fixture[
        "events"
    ]


def test_enabled_tracer_preserves_the_minimal_stream_contract(
    langgraph_v2_migrated_database_url: str,
) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json=fixture["request"],
            headers={
                "X-Application-Id": "tenant-a",
                "X-User-Groups": "research,wealth",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    UUID(response.headers["x-run-id"])
    assert response.headers["x-conversation-id"] == "conversation-123"
    assert response.text.endswith("\n\n")
    actual_events = parse_sse(response.text)
    assert [event.pop("sequence") for event in actual_events] == [1, 2, 3, 4, 5]
    assert actual_events == fixture["events"]


def test_request_header_and_generated_conversation_variants(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        missing_tenant = client.post("/v2/query/stream", json={"query": "hello"})
        generated = client.post(
            "/v2/query/stream",
            json={"query": "hello", "clientRequestId": "request-1"},
            headers={"X-Application-Id": "tenant-a"},
        )
        invalid_client_id = client.post(
            "/v2/query/stream",
            json={"query": "hello", "clientRequestId": "not allowed"},
            headers={"X-Application-Id": "tenant-a"},
        )

    assert missing_tenant.status_code == 422
    assert generated.status_code == 200
    conversation_id = generated.headers["x-conversation-id"]
    UUID(conversation_id)
    assert parse_sse(generated.text)[-1]["data"]["session_id"] == conversation_id
    assert invalid_client_id.status_code == 422


@pytest.mark.asyncio
async def test_http_adapter_accepts_a_deterministic_graph_fake(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class DeterministicGraphFake:
        def __init__(self) -> None:
            self.received_state: TracerState | None = None

        async def ainvoke(self, state: TracerState) -> dict:
            self.received_state = state
            return {
                "events": [
                    {
                        "event_key": "phase:fake:step_start:1",
                        "type": "step_start",
                        "step": "fake",
                        "sequence": 1,
                    },
                    {
                        "event_key": "lifecycle:completed:0",
                        "type": "done",
                        "data": {"source": "fake"},
                        "sequence": 2,
                    },
                ]
            }

    graph = DeterministicGraphFake()
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, graph)

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a"},
        )

    assert [event["type"] for event in parse_sse(response.text)] == [
        "step_start",
        "done",
    ]
    assert graph.received_state == {
        "query": "hello",
        "conversation_id": "conversation-1",
        "client_request_id": None,
        "events": [],
    }


def test_main_registers_the_tracer_only_when_the_feature_flag_is_enabled(
    monkeypatch,
) -> None:
    import app.main as main_module

    monkeypatch.setenv("LANGGRAPH_V2_TRACER_ENABLED", "1")
    enabled_app = reload(main_module).app
    assert "/v2/query/stream" in {route.path for route in enabled_app.routes}

    monkeypatch.delenv("LANGGRAPH_V2_TRACER_ENABLED")
    disabled_app = reload(main_module).app
    assert "/v2/query/stream" not in {route.path for route in disabled_app.routes}


def test_completed_tracer_persists_its_run_and_every_delivered_event(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a"},
        )

    delivered = parse_sse(response.text)
    run_id = UUID(response.headers["x-run-id"])

    async def load_persisted_result() -> tuple:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url,
            min_size=1,
            max_size=2,
        ) as pool:
            repository = RunEventRepository(pool)
            return (
                await repository.get_run("tenant-a", run_id),
                await repository.list_events("tenant-a", run_id),
            )

    run, persisted = asyncio.run(load_persisted_result())

    assert run.status == "completed"
    assert run.terminal_outcome == delivered[-1]["data"]
    assert [event.sequence for event in persisted] == [1, 2, 3, 4, 5]
    assert [event.event_key for event in persisted] == [
        "phase:query:step_start:1",
        "phase:query:step_completed:1",
        "phase:finalization:step_start:1",
        "phase:finalization:step_completed:1",
        "lifecycle:completed:0",
    ]
    assert [event.type for event in persisted] == [
        event["type"] for event in delivered
    ]
