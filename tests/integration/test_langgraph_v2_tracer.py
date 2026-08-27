import json
from importlib import reload
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.langgraph_v2.api import register_tracer_routes

FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "langgraph_v2" / "v1_minimal_wire.json"
)


def test_enabled_tracer_preserves_the_minimal_stream_contract() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    app = FastAPI()
    register_tracer_routes(app, enabled=True)

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
    assert [
        json.loads(frame.removeprefix("data: "))
        for frame in response.text.strip().split("\n\n")
    ] == fixture["events"]


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
