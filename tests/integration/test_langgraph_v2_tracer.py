import asyncio
import json
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib import reload
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

import app.langgraph_v2.api as api_module
from app.api.schemas import QueryResponse
from app.langgraph_v2.api import TracerGraph, register_tracer_routes
from app.langgraph_v2.checkpointing import (
    FencedAsyncPostgresSaver,
    checkpoint_namespace_for,
    initial_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.graph import TracerState, build_tracer_graph
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.run_events import EventInput, RunEventRepository
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
    resume_enabled: bool = False,
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
    register_tracer_routes(
        app, enabled=True, graph=graph, resume_enabled=resume_enabled
    )
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

    assert [
        json.loads(line.removeprefix("data: ")) for line in legacy_frames
    ] == fixture["events"]


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

        async def ainvoke(
            self, state: TracerState | None, config: RunnableConfig | None = None
        ) -> dict:
            del config
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
    assert "/v2/query/stream" in {
        getattr(route, "path", None) for route in enabled_app.routes
    }

    monkeypatch.delenv("LANGGRAPH_V2_TRACER_ENABLED")
    disabled_app = reload(main_module).app
    assert "/v2/query/stream" not in {
        getattr(route, "path", None) for route in disabled_app.routes
    }


def test_resume_route_is_default_off() -> None:
    app = FastAPI()
    register_tracer_routes(app, enabled=True)
    assert "/v2/runs/{run_id}/resume/stream" not in {
        getattr(route, "path", None) for route in app.routes
    }


def test_resume_route_returns_404_for_missing_or_cross_tenant_run(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async def seed_other_tenant() -> UUID:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            run_id = uuid4()
            run = await repository.create_run(
                tenant_id="tenant-a",
                run_id=run_id,
                conversation_id="conversation-1",
                owner_instance_id="instance-a",
            )
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                checkpoint_id="checkpoint-1",
                checkpoint_ns="namespace-1",
            )
            return run_id

    app = persistent_tracer_app(langgraph_v2_migrated_database_url, resume_enabled=True)
    run_id = asyncio.run(seed_other_tenant())
    with TestClient(app) as client:
        cross_tenant = client.post(
            f"/v2/runs/{run_id}/resume/stream",
            headers={"X-Application-Id": "tenant-b"},
        )
        missing = client.post(
            f"/v2/runs/{uuid4()}/resume/stream",
            headers={"X-Application-Id": "tenant-a"},
        )
    assert cross_tenant.status_code == 404
    assert missing.status_code == 404


def test_resume_route_returns_409_for_an_active_owner(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async def seed_active_run() -> UUID:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            run_id = uuid4()
            run = await repository.create_run(
                tenant_id="tenant-a",
                run_id=run_id,
                conversation_id="conversation-1",
                owner_instance_id="instance-a",
            )
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                checkpoint_id="checkpoint-1",
                checkpoint_ns="namespace-1",
            )
            return run_id

    run_id = asyncio.run(seed_active_run())
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, resume_enabled=True)
    with TestClient(app) as client:
        response = client.post(
            f"/v2/runs/{run_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a"},
        )
    assert response.status_code == 409


def test_resume_replays_existing_event_without_duplicate_publication(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class ReplayGraph:
        async def ainvoke(
            self, state: TracerState | None, config: RunnableConfig | None = None
        ) -> dict:
            del state, config
            return {
                "events": [
                    {
                        "event_key": "phase:query:step_start:1",
                        "type": "step_start",
                        "step": "query",
                        "sequence": 1,
                    },
                    {
                        "event_key": "recovery:completed:2",
                        "type": "done",
                        "data": {"source": "recovered"},
                        "sequence": 2,
                    },
                ]
            }

    async def seed_run() -> UUID:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            run_id = uuid4()
            run = await repository.create_run(
                tenant_id="tenant-a",
                run_id=run_id,
                conversation_id="conversation-1",
                owner_instance_id="instance-a",
            )
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                checkpoint_id="checkpoint-1",
                checkpoint_ns="namespace-1",
            )
            await repository.append_event(
                tenant_id="tenant-a",
                run_id=run_id,
                event=EventInput(
                    event_key="phase:query:step_start:1",
                    type="step_start",
                    step="query",
                ),
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            )
            with psycopg.connect(
                langgraph_v2_migrated_database_url, autocommit=True
            ) as connection:
                connection.execute(
                    "UPDATE langgraph_v2.runs SET expires_at = clock_timestamp() - interval '1 second' WHERE tenant_id = %s AND run_id = %s",
                    ("tenant-a", run_id),
                )
            return run_id

    run_id = asyncio.run(seed_run())
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        ReplayGraph(),
        resume_enabled=True,
    )
    with TestClient(app) as client:
        response = client.post(
            f"/v2/runs/{run_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a"},
        )
    delivered = parse_sse(response.text)
    assert [event["type"] for event in delivered] == ["done"]
    assert [event["sequence"] for event in delivered] == [2]

    async def read_events() -> list:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            return await RunEventRepository(pool).list_events("tenant-a", run_id)

    persisted = asyncio.run(read_events())
    assert [event.event_key for event in persisted] == [
        "phase:query:step_start:1",
        "recovery:completed:2",
    ]


@pytest.mark.parametrize("initial_status", ["stale_running", "interrupted"])
def test_resume_route_runs_injected_graph_for_resumable_run(
    langgraph_v2_migrated_database_url: str,
    initial_status: str,
) -> None:
    class ResumeGraph:
        async def ainvoke(
            self, state: TracerState | None, config: RunnableConfig | None = None
        ) -> dict:
            del config
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

    async def seed_run() -> UUID:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            run_id = uuid4()
            run = await repository.create_run(
                tenant_id="tenant-a",
                run_id=run_id,
                conversation_id="conversation-1",
                owner_instance_id="instance-a",
            )
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                checkpoint_id="checkpoint-1",
                checkpoint_ns="namespace-1",
            )
            with psycopg.connect(
                langgraph_v2_migrated_database_url, autocommit=True
            ) as connection:
                if initial_status == "stale_running":
                    connection.execute(
                        "UPDATE langgraph_v2.runs SET expires_at = clock_timestamp() - interval '1 second' WHERE tenant_id = %s AND run_id = %s",
                        ("tenant-a", run_id),
                    )
                else:
                    connection.execute(
                        "UPDATE langgraph_v2.runs SET status = 'interrupted', owner_instance_id = '', expires_at = clock_timestamp() + interval '30 seconds' WHERE tenant_id = %s AND run_id = %s",
                        ("tenant-a", run_id),
                    )
            return run_id

    run_id = asyncio.run(seed_run())
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        ResumeGraph(),
        resume_enabled=True,
    )
    with TestClient(app) as client:
        response = client.post(
            f"/v2/runs/{run_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a"},
        )
    assert response.status_code == 200
    assert parse_sse(response.text)[0]["type"] == "done"


def test_resume_route_uses_real_checkpoint_recovery_path(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async def seed_run() -> UUID:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url,
            min_size=1,
            max_size=3,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        ) as pool:
            repository = RunEventRepository(pool)
            run_id = uuid4()
            run = await repository.create_run(
                tenant_id="tenant-a",
                run_id=run_id,
                conversation_id="conversation-1",
                owner_instance_id="instance-a",
            )
            old_namespace = checkpoint_namespace_for(
                "tenant-a", str(run_id), run.execution_epoch
            )
            old_ids: list[str] = []

            async def old_pointer(checkpoint_id: str, _: str) -> None:
                old_ids.append(checkpoint_id)

            await AsyncPostgresSaver(pool).setup()
            old_graph = build_tracer_graph(
                FencedAsyncPostgresSaver(
                    pool,
                    checkpoint_namespace=old_namespace,
                    pointer_writer=old_pointer,
                )
            )
            await old_graph.ainvoke(
                {
                    "query": "authoritative",
                    "conversation_id": "conversation-1",
                    "client_request_id": None,
                    "events": [],
                },
                config=initial_checkpoint_config(
                    thread_id=thread_id_for("tenant-a", "conversation-1"),
                    checkpoint_ns=old_namespace,
                ),
            )
            assert old_ids
            assert old_ids[-1] != old_ids[0]
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                checkpoint_id=old_ids[0],
                checkpoint_ns=old_namespace,
            )
            with psycopg.connect(
                langgraph_v2_migrated_database_url, autocommit=True
            ) as connection:
                connection.execute(
                    "UPDATE langgraph_v2.runs SET expires_at = clock_timestamp() - interval '1 second' WHERE tenant_id = %s AND run_id = %s",
                    ("tenant-a", run_id),
                )
            return run_id

    run_id = asyncio.run(seed_run())
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, resume_enabled=True)
    with TestClient(app) as client:
        response = client.post(
            f"/v2/runs/{run_id}/resume/stream",
            headers={"X-Application-Id": "tenant-a"},
        )

    delivered = parse_sse(response.text)
    assert response.status_code == 200
    assert [event["sequence"] for event in delivered] == [1, 2, 3, 4, 5]
    assert delivered[1]["data"]["query"] == "authoritative"

    async def read_run_and_events() -> tuple[str, int, list]:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            recovered = await repository.get_run("tenant-a", run_id)
            events = await repository.list_events("tenant-a", run_id)
            return recovered.status, recovered.execution_epoch, events

    status, epoch, events = asyncio.run(read_run_and_events())
    assert (status, epoch) == ("completed", 2)
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]


def test_concurrent_resume_requests_have_one_http_winner(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class SlowResumeGraph:
        async def ainvoke(
            self, state: TracerState | None, config: RunnableConfig | None = None
        ) -> dict:
            del state, config
            await asyncio.sleep(0.05)
            return {
                "events": [
                    {
                        "event_key": "recovery:completed:1",
                        "type": "done",
                        "data": {"source": "concurrent-recovery"},
                        "sequence": 1,
                    }
                ]
            }

    async def seed_run() -> UUID:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            run_id = uuid4()
            run = await repository.create_run(
                tenant_id="tenant-a",
                run_id=run_id,
                conversation_id="conversation-1",
                owner_instance_id="instance-a",
            )
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
                checkpoint_id="checkpoint-1",
                checkpoint_ns="namespace-1",
            )
            with psycopg.connect(
                langgraph_v2_migrated_database_url, autocommit=True
            ) as connection:
                connection.execute(
                    "UPDATE langgraph_v2.runs SET expires_at = clock_timestamp() - interval '1 second' WHERE tenant_id = %s AND run_id = %s",
                    ("tenant-a", run_id),
                )
            return run_id

    run_id = asyncio.run(seed_run())
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        SlowResumeGraph(),
        resume_enabled=True,
    )
    with TestClient(app) as client:

        def resume() -> int:
            return client.post(
                f"/v2/runs/{run_id}/resume/stream",
                headers={"X-Application-Id": "tenant-a"},
            ).status_code

        with ThreadPoolExecutor(max_workers=2) as executor:
            statuses = list(executor.map(lambda _: resume(), range(2)))

    assert sorted(statuses) == [200, 409]

    async def read_run_and_events() -> tuple[str, int, list]:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            recovered = await repository.get_run("tenant-a", run_id)
            events = await repository.list_events("tenant-a", run_id)
            return recovered.status, recovered.execution_epoch, events

    status, epoch, events = asyncio.run(read_run_and_events())
    assert (status, epoch) == ("completed", 2)
    assert [event.event_key for event in events] == ["recovery:completed:1"]


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
    assert [event.type for event in persisted] == [event["type"] for event in delivered]


def test_long_running_request_refreshes_its_claim(
    langgraph_v2_migrated_database_url: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr(api_module, "CLAIM_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    class SlowGraph:
        async def ainvoke(
            self, state: TracerState | None, config: RunnableConfig | None = None
        ) -> dict:
            del config
            await asyncio.sleep(0.05)
            return {
                "events": [
                    {
                        "event_key": "lifecycle:completed:0",
                        "type": "done",
                        "data": {"source": "slow"},
                        "sequence": 1,
                    }
                ]
            }

    app = persistent_tracer_app(langgraph_v2_migrated_database_url, SlowGraph())
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a"},
        )

    run_id = UUID(response.headers["x-run-id"])

    async def load_run():
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url,
            min_size=1,
            max_size=2,
        ) as pool:
            return await RunEventRepository(pool).get_run("tenant-a", run_id)

    run = asyncio.run(load_run())
    assert run.heartbeat_at > run.created_at


def test_persistence_failure_emits_no_unpersisted_event(
    langgraph_v2_migrated_database_url: str,
) -> None:
    with psycopg.connect(langgraph_v2_migrated_database_url) as connection:
        connection.execute(
            """
            CREATE FUNCTION langgraph_v2.reject_event_insert()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'forced event persistence failure';
            END;
            $$ LANGUAGE plpgsql
            """
        )
        connection.execute(
            """
            CREATE TRIGGER reject_event_insert
            BEFORE INSERT ON langgraph_v2.events
            FOR EACH ROW EXECUTE FUNCTION langgraph_v2.reject_event_insert()
            """
        )

    app = persistent_tracer_app(langgraph_v2_migrated_database_url)
    try:
        with TestClient(app) as client:
            with pytest.raises(
                psycopg.errors.RaiseException,
                match="forced event persistence failure",
            ):
                client.post(
                    "/v2/query/stream",
                    json={"query": "hello"},
                    headers={"X-Application-Id": "tenant-a"},
                )

        with psycopg.connect(langgraph_v2_migrated_database_url) as connection:
            event_count = connection.execute(
                "SELECT count(*) FROM langgraph_v2.events"
            ).fetchone()
        assert event_count == (0,)
    finally:
        with psycopg.connect(langgraph_v2_migrated_database_url) as connection:
            connection.execute(
                "DROP TRIGGER IF EXISTS reject_event_insert ON langgraph_v2.events"
            )
            connection.execute(
                "DROP FUNCTION IF EXISTS langgraph_v2.reject_event_insert()"
            )
