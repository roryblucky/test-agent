import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress
from importlib import reload
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

from app.api.schemas import QueryResponse
from app.config.models import (
    FlowConfig,
    LangGraphRuntimeMode,
    LLMConfig,
    TenantConfig,
)
from app.langgraph_v2.answer import AnswerActor
from app.langgraph_v2.api import (
    GraphRuntimeFactory,
    GraphStream,
    register_v2_routes,
)
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.conversations import ConversationRepository
from app.langgraph_v2.graph import LinearGraphState
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


class LinearTenantManager:
    """Minimal trusted Linear-mode configuration used by HTTP tests."""

    def __init__(self) -> None:
        self._config = TenantConfig(
            kms_app_name="Tenant A",
            application_id="tenant-a",
            ad_groups=[],
            llm_config=LLMConfig(models={}),
            flow_config=FlowConfig(),
        )

    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Return the fixed test Tenant configuration."""
        assert tenant_id == "tenant-a"
        return self._config


def configure_linear_tenant(app: FastAPI) -> None:
    """Install the trusted Linear-mode configuration on a test app."""
    app.state.tenant_manager = LinearTenantManager()


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


def _json_value_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, int | float):
        return "number"
    raise TypeError(f"unsupported JSON value: {value!r}")


def _assert_done_shape(data: object, contract: dict[str, Any]) -> None:
    assert isinstance(data, dict)
    done = cast(dict[str, Any], data)
    assert set(done) == set(contract["done_fields"])
    for field, expected_types in contract["done_field_types"].items():
        assert _json_value_type(done[field]) in expected_types.split("|")


def parse_sse(response_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame in response_text.strip().split("\n\n"):
        payload: object = json.loads(frame.removeprefix("data: "))
        if not isinstance(payload, dict):
            raise TypeError("SSE payload must be a JSON object")
        events.append(cast(dict[str, Any], payload))
    return events


def persistent_linear_app(
    database_url: str,
    graph: GraphStream | None = None,
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    moderation_provider: ModerationProvider | None = None,
    answer_actor: AnswerActor | None = None,
    agent_runtime_factory: GraphRuntimeFactory | None = None,
) -> FastAPI:
    """Create the test-only Linear Core with its real application database pool."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=database_url),
        ):
            yield

    app = FastAPI(lifespan=lifespan)
    configure_linear_tenant(app)
    register_v2_routes(
        app,
        enabled=True,
        linear_graph_override=graph,
        refinement_actor=refinement_actor,
        retriever=retriever,
        ranker=ranker,
        moderation_provider=moderation_provider,
        answer_actor=answer_actor,
        agent_runtime_factory=agent_runtime_factory,
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
    conversation_id: UUID | str = UUID("00000000-0000-0000-0000-000000000001"),
    *,
    runtime_mode: LangGraphRuntimeMode = LangGraphRuntimeMode.LINEAR,
) -> UUID:
    """Seed the Conversation authorization required by v2 stream tests."""
    if isinstance(pool_or_database_url, str):
        async with AsyncConnectionPool(
            pool_or_database_url, min_size=1, max_size=2
        ) as pool:
            return await seed_subject_conversation(
                pool, conversation_id, runtime_mode=runtime_mode
            )
    if isinstance(conversation_id, UUID):
        resolved_id = conversation_id
    else:
        try:
            resolved_id = UUID(conversation_id)
        except ValueError:
            resolved_id = uuid5(NAMESPACE_URL, conversation_id)
    async with pool_or_database_url.connection() as connection:
        await connection.execute(
            """
            INSERT INTO langgraph_v2.conversations (
                conversation_id, tenant_id, owner_subject_id, runtime_mode
            ) VALUES (%s, 'tenant-a', 'subject-a', %s)
            ON CONFLICT (conversation_id) DO NOTHING
            """,
            (resolved_id, runtime_mode.value),
        )
    return resolved_id


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


def test_enabled_linear_core_preserves_the_minimal_stream_contract(
    langgraph_v2_migrated_database_url: str,
) -> None:
    asyncio.run(
        seed_subject_conversation(
            langgraph_v2_migrated_database_url, "00000000-0000-0000-0000-000000000123"
        )
    )
    fixture = json.loads(FIXTURE_PATH.read_text())
    app = persistent_linear_app(langgraph_v2_migrated_database_url)

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
    assert "x-run-id" not in response.headers
    assert response.headers["x-conversation-id"] == "00000000-0000-0000-0000-000000000123"
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
                "evidence_ids": actual_events[7]["data"]["evidence_ids"],
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
    _assert_done_shape(samples_by_name["done"]["data"], fixture)

    request = V2QueryRequest.model_validate(fixture["request"])
    assert set(fixture["request"]) == set(fixture["query_request_fields"])
    assert isinstance(request.query, str)
    assert isinstance(request.conversation_id, UUID)
    assert isinstance(request.client_request_id, str)

    asyncio.run(
        seed_subject_conversation(
            langgraph_v2_migrated_database_url,
            "00000000-0000-0000-0000-000000000002",
        )
    )
    app = persistent_linear_app(langgraph_v2_migrated_database_url)
    with TestClient(app) as client:
        success = client.post(
            "/v2/query/stream",
            json={**fixture["request"], "sessionId": "00000000-0000-0000-0000-000000000002"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        error = client.post(
            "/v2/query/stream",
            json={"query": "blocked", "sessionId": "00000000-0000-0000-0000-000000000002"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    for header in fixture["query_response_headers"]:
        assert header in success.headers
    success_events = parse_sse(success.text)
    error_events = parse_sse(error.text)
    assert set(event["type"] for event in success_events) <= set(fixture["event_names"])
    assert set(event["type"] for event in error_events) <= set(fixture["event_names"])
    _assert_done_shape(success_events[-1]["data"], fixture)
    captured_error_events = _wire_fixture("v1_moderation_error_wire.json")["events"]
    assert error_events[-len(captured_error_events) :] == captured_error_events


def test_request_header_and_generated_conversation_variants(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(langgraph_v2_migrated_database_url)

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
    assert repeated.headers["x-request-id"] == generated.headers["x-request-id"]

    async def read_conversation():
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            return await ConversationRepository(pool).get_conversation(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id=UUID(conversation_id),
            )

    conversation = asyncio.run(read_conversation())
    assert conversation.updated_at > conversation.created_at

    assert parse_sse(generated.text)[-1]["data"]["session_id"] == conversation_id
    assert invalid_client_id.status_code == 422


def test_query_authorizes_existing_conversation_before_streaming(
    langgraph_v2_migrated_database_url: str,
) -> None:
    asyncio.run(seed_subject_conversation(langgraph_v2_migrated_database_url))
    app = persistent_linear_app(langgraph_v2_migrated_database_url)
    with TestClient(app) as client:
        owner = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "00000000-0000-0000-0000-000000000001"},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )
        missing = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "00000000-0000-0000-0000-000000000099"},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )
        cross_subject = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "00000000-0000-0000-0000-000000000001"},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-b",
            },
        )
        missing_subject = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "00000000-0000-0000-0000-000000000001"},
            headers={"X-Application-Id": "tenant-a"},
        )
        empty_subject = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "00000000-0000-0000-0000-000000000001"},
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
        seed_subject_conversation(langgraph_v2_migrated_database_url, "00000000-0000-0000-0000-000000000001")
    )
    app = persistent_linear_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "please blocked this", "sessionId": "00000000-0000-0000-0000-000000000001"},
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
        langgraph_v2_migrated_database_url, "00000000-0000-0000-0000-000000000001"
    )

    class DeterministicGraphFake:
        def __init__(self) -> None:
            self.received_state: LinearGraphState | None = None

        def astream(
            self, state: object | None, **options: Any
        ) -> AsyncIterator[object]:
            del options
            self.received_state = (
                cast(LinearGraphState, state) if isinstance(state, dict) else None
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
    app = persistent_linear_app(langgraph_v2_migrated_database_url, graph)

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "00000000-0000-0000-0000-000000000001"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert [event["type"] for event in parse_sse(response.text)] == [
        "step_start",
        "done",
    ]
    request_id = response.headers["X-Request-Id"]
    UUID(request_id)
    assert graph.received_state == {
        "query": "hello",
        "conversation_id": "00000000-0000-0000-0000-000000000001",
        "request_id": request_id,
        "conversation_messages": [],
    }


@pytest.mark.parametrize(
    "feature_flag",
    ["LANGGRAPH_V2_UAT_ENABLED", "LANGGRAPH_V2_LINEAR_CORE_ENABLED"],
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
    assert enabled_routes == {"/v2/query/stream"}
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


def test_thread_resume_route_is_not_registered() -> None:
    app = FastAPI()
    register_v2_routes(app, enabled=True)
    assert "/v2/threads/{thread_id}/resume/stream" not in {
        getattr(route, "path", None) for route in app.routes
    }


def test_query_openapi_preserves_released_camel_case_request_fields() -> None:
    app = FastAPI()
    register_v2_routes(app, enabled=True)

    request_schema = app.openapi()["components"]["schemas"]["V2QueryRequest"]
    assert set(request_schema["properties"]) == {
        "query",
        "sessionId",
        "clientRequestId",
    }
    assert (
        V2QueryRequest.model_validate(
            {
                "query": "hello",
                "session_id": "00000000-0000-0000-0000-000000000001",
            }
        ).conversation_id
        == UUID("00000000-0000-0000-0000-000000000001")
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/v2/runs/00000000-0000-0000-0000-000000000001/resume/stream"),
        (
            "post",
            "/v2/threads/thread-a/resume/stream"
            "?expectedRequestId=request-1",
        ),
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
    assert "/v2/artifacts/{artifact_id}" not in v2_paths

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
    ],
)
def test_removed_replay_cursor_is_rejected(
    path: str, json_body: dict[str, str] | None
) -> None:
    app = FastAPI()
    register_v2_routes(app, enabled=True)

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
