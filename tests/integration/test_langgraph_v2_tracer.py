import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from importlib import reload
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

import app.langgraph_v2.api as api_module
from app.api.schemas import QueryResponse
from app.langgraph_v2.answer import AnswerActor
from app.langgraph_v2.api import TracerGraph, register_v2_routes
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.graph import TracerState, build_tracer_graph
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.pre_moderation import ModerationProvider
from app.langgraph_v2.question_refinement import QuestionRefinementActor
from app.langgraph_v2.reranking import Ranker
from app.langgraph_v2.retrieval import Retriever
from app.langgraph_v2.run_events import (
    ClaimFenced,
    EventInput,
    EventInvariantConflict,
    RunEventRepository,
)
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
    thread_resume_enabled: bool = False,
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
        thread_resume_enabled=thread_resume_enabled,
    )
    return app


def v2_stream_endpoint(app: FastAPI) -> Any:
    """Return the public stream endpoint for direct subscriber lifecycle tests."""
    return next(
        route.endpoint
        for route in app.router.routes
        if getattr(route, "path", None) == "/v2/query/stream"
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


async def close_stream_after_first_token(
    body_iterator: Any,
    blocked_on_followup_read: asyncio.Event,
) -> dict[str, Any]:
    """Consume one token, cancel the blocked follow-up read, and close cleanly."""
    token_frame: dict[str, Any] | None = None
    while token_frame is None or token_frame["type"] != "token":
        token_frame = json.loads(
            (await anext(body_iterator)).removeprefix("data: ").strip()
        )
    pending_read = asyncio.create_task(anext(body_iterator))
    await blocked_on_followup_read.wait()
    pending_read.cancel()
    with suppress(asyncio.CancelledError):
        await pending_read
    await body_iterator.aclose()
    return token_frame


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
        repeated = client.post(
            "/v2/query/stream",
            json={
                "query": "hello",
                "sessionId": generated.headers.get("x-conversation-id"),
                "clientRequestId": "request-1",
            },
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        invalid_client_id = client.post(
            "/v2/query/stream",
            json={"query": "hello", "clientRequestId": "not allowed"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert missing_tenant.status_code == 422
    assert generated.status_code == 200
    assert repeated.status_code == 200
    conversation_id = generated.headers["x-conversation-id"]
    UUID(conversation_id)
    assert repeated.headers["x-turn-id"] == generated.headers["x-turn-id"]

    async def read_turn():
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            return await ConversationMessageRepository(pool).get_turn(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id=conversation_id,
                turn_id=UUID(generated.headers["x-turn-id"]),
            )

    turn = asyncio.run(read_turn())
    assert turn.resume_deadline - turn.created_at == timedelta(hours=1)
    assert parse_sse(generated.text)[-1]["data"]["session_id"] == conversation_id
    assert invalid_client_id.status_code == 422


@pytest.mark.asyncio
async def test_done_answer_finalization_is_atomic_on_turn_validation(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        await seed_subject_conversation(pool)
        repository = RunEventRepository(pool)
        message_repository = ConversationMessageRepository(pool)

        async def persist_done(
            run_id: UUID,
            result_turn_id: UUID,
            expected_turn_id: UUID | None = None,
        ) -> None:
            result = {
                "turn_id": str(result_turn_id),
                "events": [
                    {
                        "event_key": "lifecycle:completed:0",
                        "type": "done",
                        "data": {"answer": "answer"},
                        "sequence": 1,
                    }
                ],
            }
            frames = api_module._persist_result_events(
                repository,
                message_repository,
                tenant_id="tenant-a",
                run_id=run_id,
                conversation_id="conversation-1",
                result=result,
                expected_turn_id=expected_turn_id,
                owner_instance_id="instance-a",
                execution_epoch=1,
            )
            async for _ in frames:
                pass

        missing_run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        missing_turn = uuid4()
        with pytest.raises(ClaimFenced):
            await persist_done(missing_run.run_id, missing_turn)
        assert (await repository.get_run("tenant-a", missing_run.run_id)).status == (
            "running"
        )
        assert await repository.list_events("tenant-a", missing_run.run_id) == []

        wrong_run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        actual_turn = uuid4()
        await message_repository.persist_user_message(
            tenant_id="tenant-a",
            conversation_id="conversation-1",
            run_id=wrong_run.run_id,
            turn_id=actual_turn,
            content="question",
            idempotency_key=f"turn:{actual_turn}:user",
        )
        unrelated_turn = uuid4()
        await message_repository.persist_user_message(
            tenant_id="tenant-a",
            conversation_id="conversation-1",
            run_id=uuid4(),
            turn_id=unrelated_turn,
            content="another question",
            idempotency_key=f"turn:{unrelated_turn}:user",
        )
        with pytest.raises(ValueError, match="does not match Run"):
            await persist_done(wrong_run.run_id, unrelated_turn, actual_turn)
        assert (await repository.get_run("tenant-a", wrong_run.run_id)).status == (
            "running"
        )
        assert await repository.list_events("tenant-a", wrong_run.run_id) == []

        valid_run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        valid_turn = uuid4()
        await message_repository.persist_user_message(
            tenant_id="tenant-a",
            conversation_id="conversation-1",
            run_id=valid_run.run_id,
            turn_id=valid_turn,
            content="question",
            idempotency_key=f"turn:{valid_turn}:user",
        )
        await persist_done(valid_run.run_id, valid_turn)

        assert (await repository.get_run("tenant-a", valid_run.run_id)).status == (
            "completed"
        )
        events = await repository.list_events("tenant-a", valid_run.run_id)
        assert [(event.event_key, event.type) for event in events] == [
            ("lifecycle:completed:0", "done")
        ]
        messages = await message_repository.list_messages(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="conversation-1",
        )
        assert [
            message.role for message in messages if message.turn_id == valid_turn
        ] == [
            "user",
            "assistant",
        ]

        conflict_run = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        conflict_turn = uuid4()
        await message_repository.persist_user_message(
            tenant_id="tenant-a",
            conversation_id="conversation-1",
            run_id=conflict_run.run_id,
            turn_id=conflict_turn,
            content="question",
            idempotency_key=f"turn:{conflict_turn}:user",
        )
        await repository.append_event(
            tenant_id="tenant-a",
            run_id=conflict_run.run_id,
            owner_instance_id="instance-a",
            execution_epoch=1,
            event=EventInput(
                event_key="lifecycle:completed:0",
                type="done",
                data={"answer": "previous answer"},
            ),
        )
        with pytest.raises(EventInvariantConflict):
            await persist_done(conflict_run.run_id, conflict_turn, conflict_turn)
        failed_conflict_run = await repository.get_run("tenant-a", conflict_run.run_id)
        assert failed_conflict_run.status == "failed"
        assert failed_conflict_run.terminal_outcome == {
            "error": "event_invariant_conflict",
            "event_key": "lifecycle:completed:0",
        }
        assert [
            event.event_key
            for event in await repository.list_events("tenant-a", conflict_run.run_id)
        ] == ["lifecycle:completed:0"]
        assert [
            message
            for message in await message_repository.list_messages(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="conversation-1",
            )
            if message.turn_id == conflict_turn and message.role == "assistant"
        ] == []


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

    async def read_run() -> tuple[str, list]:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            repository = RunEventRepository(pool)
            return (
                (await repository.get_run("tenant-a", run_id)).status,
                await repository.list_events("tenant-a", run_id),
            )

    status, persisted = asyncio.run(read_run())
    assert status == "interrupted"
    assert persisted == []


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

        def astream(self, state: TracerState | None, **options: Any) -> AsyncIterator:
            del options
            self.received_state = state

            async def stream() -> AsyncIterator:
                yield (
                    "updates",
                    {
                        "fake": {
                            "events": [
                                {
                                    "event_key": "phase:fake:step_start:1",
                                    "type": "step_start",
                                    "step": "fake",
                                },
                                {
                                    "event_key": "lifecycle:completed:0",
                                    "type": "done",
                                    "data": {"source": "fake"},
                                },
                            ]
                        }
                    },
                )

            return stream()

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
    enabled_routes = {
        getattr(route, "path", None)
        for route in enabled_app.routes
        if getattr(route, "path", "").startswith("/v2/")
    }
    assert enabled_routes == {
        "/v2/query/stream",
        "/v2/threads/{thread_id}/resume/stream",
    }
    assert {
        path for path in enabled_app.openapi()["paths"] if path.startswith("/v2/")
    } == enabled_routes

    monkeypatch.delenv(feature_flag)
    disabled_app = reload(main_module).app
    assert not {
        getattr(route, "path", None)
        for route in disabled_app.routes
        if getattr(route, "path", "").startswith("/v2/")
    }


def test_thread_resume_route_is_default_off() -> None:
    app = FastAPI()
    register_v2_routes(app, enabled=True)
    assert "/v2/threads/{thread_id}/resume/stream" not in {
        getattr(route, "path", None) for route in app.routes
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/v2/runs/00000000-0000-0000-0000-000000000001/resume/stream"),
        ("get", "/v2/runs/00000000-0000-0000-0000-000000000001/stream"),
        ("post", "/v2/runs/00000000-0000-0000-0000-000000000001/cancel"),
    ],
)
def test_removed_run_control_routes_are_404(method: str, path: str) -> None:
    app = FastAPI()
    register_v2_routes(app, enabled=True)

    v2_paths = {path for path in app.openapi()["paths"] if path.startswith("/v2/")}
    assert "/v2/query/stream" in v2_paths
    assert "/v2/runs/{run_id}/resume/stream" not in v2_paths
    assert "/v2/runs/{run_id}/stream" not in v2_paths
    assert "/v2/runs/{run_id}/cancel" not in v2_paths

    with TestClient(app) as client:
        response = getattr(client, method)(
            path,
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )

    assert response.status_code == 404


def test_completed_tracer_persists_only_later_phase_and_terminal_events(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "conversation-completed-tracer"
    asyncio.run(
        seed_subject_conversation(langgraph_v2_migrated_database_url, conversation_id)
    )
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": conversation_id},
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
                    conversation_id=conversation_id,
                ),
            )

    run, persisted, messages = asyncio.run(load_persisted_result())

    assert run.status == "completed"
    assert run.terminal_outcome == delivered[-1]["data"]
    assert [event.sequence for event in persisted] == list(range(1, 8))
    assert [event.event_key for event in persisted] == [
        "phase:retrieval:step_start:1",
        "phase:retrieval:step_completed:1",
        "phase:reranking:step_start:1",
        "phase:reranking:step_completed:1",
        "phase:finalization:step_start:1",
        "phase:finalization:step_completed:1",
        "lifecycle:completed:0",
    ]
    assert [event.type for event in persisted] == [
        event["type"] for event in delivered[6:]
    ]
    assert [(message.role, message.content) for message in messages] == [
        ("user", "hello"),
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
        def astream(self, state: TracerState | None, **options: Any) -> AsyncIterator:
            del state, options

            async def stream() -> AsyncIterator:
                await asyncio.sleep(0.05)
                yield (
                    "updates",
                    {
                        "slow": {
                            "events": [
                                {
                                    "event_key": "lifecycle:completed:0",
                                    "type": "done",
                                    "data": {"source": "slow"},
                                    "sequence": 1,
                                }
                            ]
                        }
                    },
                )

            return stream()

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
