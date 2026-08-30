import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from importlib import reload
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

from app.api.schemas import QueryResponse
from app.langgraph_v2.answer import AnswerActor
from app.langgraph_v2.api import TracerGraph, register_v2_routes
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
)
from app.langgraph_v2.graph import TracerState
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.pre_moderation import ModerationProvider
from app.langgraph_v2.question_refinement import QuestionRefinementActor
from app.langgraph_v2.reranking import Ranker
from app.langgraph_v2.retrieval import Retriever
from app.services.events import EventEmitter

FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "langgraph_v2" / "v1_minimal_wire.json"
)
UAT_CONTRACT_PATH = (
    Path(__file__).parents[1] / "fixtures" / "langgraph_v2" / "v2_uat_contract.json"
)


def _wire_fixture(name: str) -> dict[str, Any]:
    fixture: object = json.loads((FIXTURE_PATH.parent / name).read_text())
    if not isinstance(fixture, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return cast(dict[str, Any], fixture)


def _event_payload_type(event: dict[str, Any]) -> str:
    if "data" not in event:
        return "none"
    data = event["data"]
    if isinstance(data, str):
        return "string"
    if isinstance(data, list):
        return "array"
    if isinstance(data, dict):
        return "object"
    raise TypeError(f"unsupported captured event payload: {data!r}")


def parse_sse(response_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame in response_text.strip().split("\n\n"):
        payload: object = json.loads(frame.removeprefix("data: "))
        if not isinstance(payload, dict):
            raise TypeError("SSE payload must be a JSON object")
        events.append(cast(dict[str, Any], payload))
    return events


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
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
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
    for route in app.router.routes:
        if isinstance(route, APIRoute) and route.path == "/v2/query/stream":
            return route.endpoint
    raise LookupError("v2 stream endpoint is not registered")


def stream_request(app: FastAPI) -> Request:
    """Build only the ASGI Request boundary needed by the stream endpoint."""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v2/query/stream",
            "query_string": b"",
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
    """Seed the Conversation authorization required by v2 stream tests."""
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


def test_released_uat_contract_fixture_matches_public_http_shapes(
    langgraph_v2_migrated_database_url: str,
) -> None:
    fixture = _wire_fixture(UAT_CONTRACT_PATH.name)
    minimal = _wire_fixture("v1_minimal_wire.json")
    answer = _wire_fixture("v1_answer_wire.json")
    captured_events = [
        *minimal["events"],
        *answer["token_events"],
        answer["error_event"],
        _wire_fixture("v1_citations_wire.json")["event"],
        *_wire_fixture("v1_stopped_wire.json")["events"],
        *_wire_fixture("v1_progress_wire.json")["events"],
    ]
    samples_by_name = {
        event["type"]: cast(dict[str, Any], event) for event in captured_events
    }
    assert set(samples_by_name) == set(fixture["event_names"])
    for event_name, sample in samples_by_name.items():
        assert _event_payload_type(sample) == fixture["payload_types"][event_name]
    done_data = samples_by_name["done"]["data"]
    assert isinstance(done_data, dict)
    assert set(cast(dict[str, Any], done_data)) == set(fixture["done_fields"])

    request = V2QueryRequest.model_validate(fixture["request"])
    assert set(fixture["request"]) == set(fixture["query_request_fields"])
    assert isinstance(request.query, str)
    assert isinstance(request.conversation_id, str)
    assert isinstance(request.client_request_id, str)

    asyncio.run(
        seed_subject_conversation(
            langgraph_v2_migrated_database_url,
            "contract-conversation",
        )
    )
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)
    with TestClient(app) as client:
        success = client.post(
            "/v2/query/stream",
            json={**fixture["request"], "sessionId": "contract-conversation"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        error = client.post(
            "/v2/query/stream",
            json={"query": "blocked", "sessionId": "contract-conversation"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    for header in fixture["query_response_headers"]:
        assert header in success.headers
    assert fixture["resume_request_fields"] == {
        "thread_id": "string:path",
        "expectedTurnId": "uuid:query",
    }
    assert fixture["resume_response_headers"] == [
        "x-conversation-id",
        "x-turn-id",
        "x-thread-id",
    ]
    success_events = parse_sse(success.text)
    error_events = parse_sse(error.text)
    assert set(event["type"] for event in success_events) <= set(
        fixture["event_names"]
    )
    assert set(event["type"] for event in error_events) <= set(
        fixture["event_names"]
    )
    assert set(success_events[-1]["data"]) == set(fixture["done_fields"])
    assert isinstance(success_events[-1]["data"], dict)
    assert error_events[-1]["type"] == "error"
    assert isinstance(error_events[-1]["data"], str)


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

        def astream(
            self, state: object | None, **options: Any
        ) -> AsyncIterator[object]:
            del options
            self.received_state = (
                cast(TracerState, state) if isinstance(state, dict) else None
            )

            async def stream() -> AsyncIterator[object]:
                yield (
                    "custom",
                    {
                        "type": "step_start",
                        "step": "fake",
                    },
                )
                yield (
                    "custom",
                    {
                        "type": "done",
                        "data": {"source": "fake"},
                        "checkpoint_terminal": True,
                    },
                )
                yield ("updates", {"fake": {"final_response": {}}})

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
    }


@pytest.mark.parametrize(
    "feature_flag",
    ["LANGGRAPH_V2_UAT_ENABLED", "LANGGRAPH_V2_TRACER_ENABLED"],
)
def test_main_registers_the_uat_route_set_only_when_a_supported_flag_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
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


@pytest.mark.parametrize(
    ("path", "json_body"),
    [
        ("/v2/query/stream?afterSequence=3", {"query": "hello"}),
        (
            "/v2/threads/thread-a/resume/stream"
            "?expectedTurnId=00000000-0000-0000-0000-000000000001"
            "&afterSequence=3",
            None,
        ),
    ],
)
def test_removed_replay_cursor_is_rejected(
    path: str, json_body: dict[str, str] | None
) -> None:
    app = FastAPI()
    register_v2_routes(app, enabled=True, thread_resume_enabled=True)

    with TestClient(app) as client:
        response = client.post(
            path,
            json=json_body,
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Replay query parameter is no longer supported: afterSequence"
    }
