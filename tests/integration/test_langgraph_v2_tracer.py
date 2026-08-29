import asyncio
import json
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from importlib import reload
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

import app.langgraph_v2.api as api_module
from app.api.schemas import QueryResponse
from app.langgraph_v2.answer import ANSWER_CHUNK_INTERVAL_MS, AnswerActor
from app.langgraph_v2.api import TracerGraph, register_v2_routes
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import (
    FencedAsyncPostgresSaver,
    checkpoint_namespace_for,
    initial_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
)
from app.langgraph_v2.graph import TracerState, build_tracer_graph
from app.langgraph_v2.phase_results import PhaseResultRepository
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.pre_moderation import ModerationProvider
from app.langgraph_v2.question_refinement import QuestionRefinementActor
from app.langgraph_v2.reranking import Ranker
from app.langgraph_v2.retrieval import Retriever
from app.langgraph_v2.run_events import EventInput, RunEventRepository
from app.langgraph_v2.runtime import LocalRunRuntime, RuntimeStopping
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
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    moderation_provider: ModerationProvider | None = None,
    answer_actor: AnswerActor | None = None,
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
    register_v2_routes(
        app,
        enabled=True,
        graph=graph,
        refinement_actor=refinement_actor,
        retriever=retriever,
        ranker=ranker,
        moderation_provider=moderation_provider,
        answer_actor=answer_actor,
        answer_chunk_interval_ms=ANSWER_CHUNK_INTERVAL_MS,
        resume_enabled=resume_enabled,
    )
    return app


def v2_stream_endpoint(app: FastAPI) -> Any:
    """Return the public stream endpoint for direct subscriber lifecycle tests."""
    return next(
        route.endpoint
        for route in app.router.routes
        if getattr(route, "path", None) == "/v2/query/stream"
    )


def v2_resume_endpoint(app: FastAPI) -> Any:
    """Return the public resume endpoint for subscriber lifecycle tests."""
    return next(
        route.endpoint
        for route in app.router.routes
        if getattr(route, "path", None) == "/v2/runs/{run_id}/resume/stream"
    )


def stream_request(app: FastAPI) -> Request:
    """Build only the ASGI Request boundary needed by the stream endpoint."""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v2/query/stream",
            "headers": [],
            "app": app,
        }
    )


async def seed_subject_conversation(
    pool_or_database_url: AsyncConnectionPool[Any] | str,
    conversation_id: str = "conversation-1",
) -> None:
    """Seed the Conversation authorization required by v2 Run tests."""
    if isinstance(pool_or_database_url, str):
        async with AsyncConnectionPool(
            pool_or_database_url, min_size=1, max_size=2
        ) as pool:
            await seed_subject_conversation(pool, conversation_id)
        return
    await ConversationMessageRepository(pool_or_database_url).resolve_conversation(
        context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
        conversation_id=conversation_id,
    )


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


@pytest.mark.asyncio
async def test_captured_legacy_moderation_error_fixture() -> None:
    fixture = json.loads(
        (FIXTURE_PATH.parent / "v1_moderation_error_wire.json").read_text()
    )
    emitter = EventEmitter()
    await emitter.emit_step_start("moderation:pre")
    await emitter.emit_error(
        "Content flagged by moderation: query contains blocked term: blocked"
    )

    legacy_frames = [line async for line in emitter]

    assert [json.loads(line.removeprefix("data: ")) for line in legacy_frames] == (
        fixture["events"]
    )


@pytest.mark.asyncio
async def test_in_memory_graph_sequences_events_additively() -> None:
    result = await build_tracer_graph().ainvoke(
        {
            "query": "hello",
            "conversation_id": "conversation-1",
            "client_request_id": None,
            "events": [],
        }
    )

    assert [event["sequence"] for event in result["events"]] == list(range(1, 10))


def test_enabled_tracer_preserves_the_minimal_stream_contract(
    langgraph_v2_migrated_database_url: str,
) -> None:
    asyncio.run(
        seed_subject_conversation(
            langgraph_v2_migrated_database_url, "conversation-123"
        )
    )
    fixture = json.loads(FIXTURE_PATH.read_text())
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json=fixture["request"],
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
                "X-User-Groups": "research,wealth",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    UUID(response.headers["x-run-id"])
    assert response.headers["x-conversation-id"] == "conversation-123"
    assert response.text.endswith("\n\n")
    actual_events = parse_sse(response.text)
    assert [event.pop("sequence") for event in actual_events] == list(range(1, 14))
    assert actual_events == [
        {
            "type": "step_start",
            "step": "query",
        },
        {
            "type": "step_completed",
            "step": "query",
            "data": {"query": "What is LangGraph?"},
        },
        {
            "type": "step_start",
            "step": "moderation:pre",
        },
        {
            "type": "step_completed",
            "step": "moderation:pre",
            "data": {"is_flagged": False, "mode": "pre"},
        },
        {
            "type": "step_start",
            "step": "llm:refine_question",
        },
        {
            "type": "step_completed",
            "step": "llm:refine_question",
            "data": {"refined_query": "What is LangGraph?"},
        },
        {"type": "step_start", "step": "retriever"},
        {
            "type": "step_completed",
            "step": "retriever",
            "data": {
                "document_count": 1,
                "documents": [{"id": "mock-doc-1", "score": 1.0}],
                "artifact_ids": actual_events[7]["data"]["artifact_ids"],
            },
        },
        {"type": "step_start", "step": "reranker"},
        {
            "type": "step_completed",
            "step": "reranker",
            "data": {
                "document_count": 1,
                "selected_ids": ["mock-doc-1"],
            },
        },
        {
            "type": "step_start",
            "step": "finalization",
        },
        {
            "type": "step_completed",
            "step": "finalization",
            "data": {"status": "completed"},
        },
        {
            "type": "done",
            "data": {
                **fixture["events"][-1]["data"],
                "refined_query": "What is LangGraph?",
                "documents": [],
                "metadata": {
                    "steps_executed": [
                        "query",
                        "pre_moderation",
                        "question_refinement",
                        "retrieval",
                        "reranking",
                        "finalization",
                    ]
                },
            },
        },
    ]


def test_request_header_and_generated_conversation_variants(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        missing_tenant = client.post("/v2/query/stream", json={"query": "hello"})
        generated = client.post(
            "/v2/query/stream",
            json={"query": "hello", "clientRequestId": "request-1"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        invalid_client_id = client.post(
            "/v2/query/stream",
            json={"query": "hello", "clientRequestId": "not allowed"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert missing_tenant.status_code == 422
    assert generated.status_code == 200
    conversation_id = generated.headers["x-conversation-id"]
    UUID(conversation_id)
    assert parse_sse(generated.text)[-1]["data"]["session_id"] == conversation_id
    assert invalid_client_id.status_code == 422


def test_query_authorizes_existing_conversation_before_streaming(
    langgraph_v2_migrated_database_url: str,
) -> None:
    asyncio.run(seed_subject_conversation(langgraph_v2_migrated_database_url))
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)
    with TestClient(app) as client:
        owner = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "conversation-1"},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )
        missing = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "missing-conversation"},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )
        cross_subject = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "conversation-1"},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-b",
            },
        )
        missing_subject = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a"},
        )
        empty_subject = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": ""},
        )

    assert owner.status_code == 200
    assert missing.status_code == 404
    assert cross_subject.status_code == 404
    assert missing.json() == cross_subject.json()
    assert missing_subject.status_code == empty_subject.status_code == 422


def test_flagged_query_emits_error_before_finalization(
    langgraph_v2_migrated_database_url: str,
) -> None:
    asyncio.run(
        seed_subject_conversation(langgraph_v2_migrated_database_url, "conversation-1")
    )
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "please blocked this", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    delivered = parse_sse(response.text)
    assert response.status_code == 200
    assert [event["step"] for event in delivered[:-1]] == [
        "query",
        "query",
        "moderation:pre",
    ]
    assert delivered[-1]["type"] == "error"
    assert delivered[-1]["data"] == (
        "Content flagged by moderation: query contains blocked term: blocked"
    )
    assert all(event.get("step") != "finalization" for event in delivered)

    run_id = UUID(response.headers["x-run-id"])

    async def read_status() -> str:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            return (await RunEventRepository(pool).get_run("tenant-a", run_id)).status

    assert asyncio.run(read_status()) == "failed"


def test_failed_moderation_error_retry_is_idempotent(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async def exercise() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            run = await repository.create_run(
                tenant_id="tenant-a",
                run_id=uuid4(),
                conversation_id="conversation-1",
                owner_instance_id="instance-a",
            )
            event = EventInput(
                event_key="phase:pre_moderation:error:1",
                type="error",
                data="Content flagged by moderation: query contains blocked term: blocked",
            )
            first = await repository.fail_run(
                tenant_id="tenant-a",
                run_id=run.run_id,
                event=event,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            )
            repeated = await repository.fail_run(
                tenant_id="tenant-a",
                run_id=run.run_id,
                event=event,
                owner_instance_id=run.owner_instance_id,
                execution_epoch=run.execution_epoch,
            )
            assert repeated == first
            assert (await repository.get_run("tenant-a", run.run_id)).status == "failed"
            assert await repository.list_events("tenant-a", run.run_id) == [first]

    asyncio.run(exercise())


@pytest.mark.asyncio
async def test_http_adapter_accepts_a_deterministic_graph_fake(
    langgraph_v2_migrated_database_url: str,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )

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
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert [event["type"] for event in parse_sse(response.text)] == [
        "step_start",
        "done",
    ]
    turn_id = response.headers["X-Turn-Id"]
    UUID(turn_id)
    assert graph.received_state == {
        "query": "hello",
        "conversation_id": "conversation-1",
        "turn_id": turn_id,
        "client_request_id": None,
        "events": [],
    }


@pytest.mark.asyncio
async def test_closing_the_public_sse_subscription_keeps_its_run_executing(
    langgraph_v2_migrated_database_url: str,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )

    class BlockingGraph:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.completed = asyncio.Event()

        async def ainvoke(
            self, state: TracerState | None, config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            del state, config
            self.started.set()
            await self.release.wait()
            self.completed.set()
            return {
                "events": [
                    {
                        "event_key": "lifecycle:completed:0",
                        "type": "done",
                        "data": {"source": "detached-fake"},
                        "sequence": 1,
                    }
                ]
            }

    graph = BlockingGraph()
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, graph)
    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        subscriber = response.body_iterator
        pending_read = asyncio.create_task(anext(subscriber))
        await graph.started.wait()
        pending_read.cancel()
        with suppress(asyncio.CancelledError):
            await pending_read
        await subscriber.aclose()

        graph.release.set()
        for _ in range(100):
            if (
                graph.completed.is_set()
                and app.state.langgraph_v2_runtime.active_task_count == 0
            ):
                break
            await asyncio.sleep(0.01)

        assert graph.completed.is_set()
        assert app.state.langgraph_v2_runtime.active_task_count == 0
        run = await RunEventRepository(app.state.langgraph_v2_postgres_pool).get_run(
            "tenant-a", UUID(response.headers["x-run-id"])
        )
        assert run.status == "completed"
        assert [
            event.type
            for event in await RunEventRepository(
                app.state.langgraph_v2_postgres_pool
            ).list_events("tenant-a", run.run_id)
        ] == ["done"]


@pytest.mark.asyncio
async def test_lifespan_shutdown_interrupts_unfinished_locally_owned_runs(
    langgraph_v2_migrated_database_url: str,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )

    class BlockingGraph:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def ainvoke(
            self, state: TracerState | None, config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            del state, config
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    graph = BlockingGraph()
    app = persistent_tracer_app(langgraph_v2_migrated_database_url, graph)
    app.state.langgraph_v2_runtime = LocalRunRuntime(shutdown_grace_seconds=0)
    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        subscriber = response.body_iterator
        pending_read = asyncio.create_task(anext(subscriber))
        await graph.started.wait()
        pending_read.cancel()
        with suppress(asyncio.CancelledError):
            await pending_read
        await subscriber.aclose()
        run_id = UUID(response.headers["x-run-id"])

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        run = await RunEventRepository(pool).get_run("tenant-a", run_id)
    assert run.status == "interrupted"
    assert run.owner_instance_id == ""
    assert app.state.langgraph_v2_runtime.accepting is False
    assert app.state.langgraph_v2_runtime.active_task_count == 0
    assert graph.cancelled.is_set()


@pytest.mark.asyncio
async def test_start_rejection_releases_the_newly_claimed_run(
    langgraph_v2_migrated_database_url: str,
    monkeypatch,
) -> None:
    await seed_subject_conversation(
        langgraph_v2_migrated_database_url, "conversation-1"
    )
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)
    runtime = app.state.langgraph_v2_runtime

    def reject_start(execution) -> None:
        execution.close()
        raise RuntimeStopping("test shutdown race")

    monkeypatch.setattr(runtime, "start", reject_start)
    async with app.router.lifespan_context(app):
        response = await v2_stream_endpoint(app)(
            V2QueryRequest(query="hello", conversation_id="conversation-1"),
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        with pytest.raises(StopAsyncIteration):
            await anext(response.body_iterator)
        run = await RunEventRepository(app.state.langgraph_v2_postgres_pool).get_run(
            "tenant-a", UUID(response.headers["x-run-id"])
        )

    assert run.status == "interrupted"
    assert run.owner_instance_id == ""


@pytest.mark.asyncio
async def test_resume_registration_starts_before_its_sse_body_is_consumed(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class BlockingGraph:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def ainvoke(
            self, state: TracerState | None, config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            del state, config
            self.started.set()
            await self.release.wait()
            return {
                "events": [
                    {
                        "event_key": "lifecycle:completed:0",
                        "type": "done",
                        "data": {"source": "resume-detached"},
                        "sequence": 1,
                    }
                ]
            }

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunEventRepository(pool)
        await seed_subject_conversation(pool)
        run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await repository.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        async with pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    UPDATE langgraph_v2.runs
                    SET status = 'interrupted', owner_instance_id = ''
                    WHERE tenant_id = %s AND run_id = %s
                    """,
                    ("tenant-a", run.run_id),
                )

    graph = BlockingGraph()
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        graph,
        resume_enabled=True,
    )
    async with app.router.lifespan_context(app):
        response = await v2_resume_endpoint(app)(
            run.run_id,
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        await graph.started.wait()
        await response.body_iterator.aclose()
        graph.release.set()
        for _ in range(100):
            if app.state.langgraph_v2_runtime.active_task_count == 0:
                break
            await asyncio.sleep(0.01)
        resumed = await RunEventRepository(
            app.state.langgraph_v2_postgres_pool
        ).get_run("tenant-a", run.run_id)

    assert resumed.status == "completed"


@pytest.mark.asyncio
async def test_resume_stream_is_fenced_to_claim_captured_before_body_consumption(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class BlockingGraph:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def ainvoke(
            self, state: TracerState | None, config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            del state, config
            self.started.set()
            await self.release.wait()
            return {
                "events": [
                    {
                        "event_key": "recovery:old-claim:done",
                        "type": "done",
                        "data": {"source": "old-claim"},
                        "sequence": 1,
                    }
                ]
            }

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunEventRepository(pool)
        await seed_subject_conversation(pool)
        run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await repository.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        async with pool.connection() as connection:
            await connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET status = 'interrupted', owner_instance_id = ''
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run.run_id),
            )

    graph = BlockingGraph()
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        graph,
        resume_enabled=True,
    )
    async with app.router.lifespan_context(app):
        response = await v2_resume_endpoint(app)(
            run.run_id,
            stream_request(app),
            request_context=TrustedRequestContext(
                tenant_id="tenant-a", subject_id="subject-a"
            ),
        )
        await graph.started.wait()
        repository = RunEventRepository(app.state.langgraph_v2_postgres_pool)
        async with app.state.langgraph_v2_postgres_pool.connection() as connection:
            await connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run.run_id),
            )
        replacement = await repository.resume_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="replacement",
        )
        await repository.complete_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=replacement.owner_instance_id,
            execution_epoch=replacement.execution_epoch,
            event=EventInput(
                event_key="recovery:replacement:done",
                type="done",
                data={"source": "replacement"},
            ),
        )

        boundary = json.loads(
            (await anext(response.body_iterator)).removeprefix("data: ")
        )
        assert boundary["type"] == "error"
        assert boundary["data"] == {
            "status": "interrupted",
            "reason": "claim_expired",
        }
        with pytest.raises(StopAsyncIteration):
            await anext(response.body_iterator)
        await response.body_iterator.aclose()
        graph.release.set()
        for _ in range(100):
            if app.state.langgraph_v2_runtime.active_task_count == 0:
                break
            await asyncio.sleep(0.01)

        assert replacement.execution_epoch == 3


@pytest.mark.parametrize(
    "feature_flag",
    ["LANGGRAPH_V2_UAT_ENABLED", "LANGGRAPH_V2_TRACER_ENABLED"],
)
def test_main_registers_the_uat_route_set_only_when_a_supported_flag_is_enabled(
    monkeypatch,
    feature_flag: str,
) -> None:
    import app.main as main_module

    monkeypatch.setenv(feature_flag, "1")
    enabled_app = reload(main_module).app
    assert {
        getattr(route, "path", None)
        for route in enabled_app.routes
        if getattr(route, "path", "").startswith("/v2/")
    } == {
        "/v2/query/stream",
        "/v2/runs/{run_id}/stream",
        "/v2/runs/{run_id}/resume/stream",
        "/v2/runs/{run_id}/cancel",
    }

    monkeypatch.delenv(feature_flag)
    disabled_app = reload(main_module).app
    assert not {
        getattr(route, "path", None)
        for route in disabled_app.routes
        if getattr(route, "path", "").startswith("/v2/")
    }


def test_resume_route_is_default_off() -> None:
    app = FastAPI()
    register_v2_routes(app, enabled=True)
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
            await seed_subject_conversation(pool)
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
            headers={"X-Application-Id": "tenant-b", "X-Subject-Id": "subject-a"},
        )
        missing = client.post(
            f"/v2/runs/{uuid4()}/resume/stream",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
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
            await seed_subject_conversation(pool)
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
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
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
            await seed_subject_conversation(pool)
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
            f"/v2/runs/{run_id}/resume/stream?afterSequence=1",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
    delivered = parse_sse(response.text)
    assert [event["type"] for event in delivered] == ["error", "done"]
    assert [event["sequence"] for event in delivered] == [2, 3]

    async def read_events() -> list:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            return await RunEventRepository(pool).list_events("tenant-a", run_id)

    persisted = asyncio.run(read_events())
    assert [event.event_key for event in persisted] == [
        "phase:query:step_start:1",
        "lifecycle:interrupted:1",
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
            await seed_subject_conversation(pool)
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
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
    assert response.status_code == 200
    assert [event["type"] for event in parse_sse(response.text)] == (
        ["error", "done"] if initial_status == "stale_running" else ["done"]
    )


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
            await seed_subject_conversation(pool)
            run_id = uuid4()
            run = await repository.create_run(
                tenant_id="tenant-a",
                run_id=run_id,
                conversation_id="conversation-1",
                owner_instance_id="instance-a",
            )
            messages = ConversationMessageRepository(pool)
            await messages.resolve_conversation(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="conversation-1",
            )
            await messages.persist_user_message(
                tenant_id="tenant-a",
                conversation_id="conversation-1",
                run_id=run_id,
                content="authoritative",
                idempotency_key=f"run:{run_id}:user",
            )
            old_namespace = checkpoint_namespace_for(
                "tenant-a", str(run_id), run.execution_epoch
            )
            old_ids: list[str] = []

            async def old_pointer(checkpoint_id: str, _: str) -> None:
                old_ids.append(checkpoint_id)

            await AsyncPostgresSaver(cast(Any, pool)).setup()
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
            f"/v2/runs/{run_id}/resume/stream?afterSequence=10",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    delivered = parse_sse(response.text)
    assert response.status_code == 200
    assert [event["sequence"] for event in delivered] == [11, 12, 13, 14]

    async def read_run_and_events() -> tuple[str, int, list, Any]:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            recovered = await repository.get_run("tenant-a", run_id)
            events = await repository.list_events("tenant-a", run_id)
            phase = await PhaseResultRepository(pool).get_completed(
                "tenant-a", run_id, "query"
            )
            return recovered.status, recovered.execution_epoch, events, phase

    status, epoch, events, phase = asyncio.run(read_run_and_events())
    assert (status, epoch) == ("completed", 2)
    assert [event.sequence for event in events] == list(range(1, 15))
    assert phase is not None
    assert phase.normalized_result == {
        "query": "authoritative",
        "history_snapshot": [],
    }


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
            await seed_subject_conversation(pool)
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
                headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
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
    assert [event.event_key for event in events] == [
        "lifecycle:interrupted:1",
        "recovery:completed:1",
    ]


def test_completed_tracer_persists_its_run_and_every_delivered_event(
    langgraph_v2_migrated_database_url: str,
) -> None:
    asyncio.run(
        seed_subject_conversation(langgraph_v2_migrated_database_url, "conversation-1")
    )
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "conversation-1"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
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
            messages = ConversationMessageRepository(pool)
            return (
                await repository.get_run("tenant-a", run_id),
                await repository.list_events("tenant-a", run_id),
                await messages.list_messages(
                    context=TrustedRequestContext(
                        tenant_id="tenant-a", subject_id="subject-a"
                    ),
                    conversation_id="conversation-1",
                ),
            )

    run, persisted, messages = asyncio.run(load_persisted_result())

    assert run.status == "completed"
    assert run.terminal_outcome == delivered[-1]["data"]
    assert [event.sequence for event in persisted] == list(range(1, 14))
    assert [event.event_key for event in persisted] == [
        "phase:query:step_start:1",
        "phase:query:step_completed:1",
        "phase:pre_moderation:step_start:1",
        "phase:pre_moderation:step_completed:1",
        "phase:question_refinement:step_start:1",
        "phase:question_refinement:step_completed:1",
        "phase:retrieval:step_start:1",
        "phase:retrieval:step_completed:1",
        "phase:reranking:step_start:1",
        "phase:reranking:step_completed:1",
        "phase:finalization:step_start:1",
        "phase:finalization:step_completed:1",
        "lifecycle:completed:0",
    ]
    assert [event.type for event in persisted] == [event["type"] for event in delivered]
    assert [(message.role, message.content) for message in messages] == [
        ("user", "hello")
    ]


def test_long_running_request_refreshes_its_claim(
    langgraph_v2_migrated_database_url: str,
    monkeypatch,
) -> None:
    asyncio.run(
        seed_subject_conversation(langgraph_v2_migrated_database_url, "conversation-1")
    )
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
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
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
                    headers={
                        "X-Application-Id": "tenant-a",
                        "X-Subject-Id": "subject-a",
                    },
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
