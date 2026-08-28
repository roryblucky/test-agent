from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.langgraph_v2.pre_moderation import ModerationDecision
from app.langgraph_v2.question_refinement import MockQuestionRefinementActor
from app.services.exceptions import TenantNotFoundError
from tests.integration.test_langgraph_v2_tracer import parse_sse, persistent_tracer_app


class _FailingModeration:
    async def check(self, text: str) -> ModerationDecision:
        del text
        raise RuntimeError("moderation backend unavailable")


class _EmptyTenantManager:
    def get_providers(self, tenant_id: str):
        assert tenant_id == "tenant-a"
        return SimpleNamespace(retriever=None, ranker=None, moderation=None)


class _UnknownTenantManager:
    def get_providers(self, tenant_id: str):
        raise TenantNotFoundError(tenant_id)


def test_configured_provider_failure_is_persisted_as_v2_error(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=MockQuestionRefinementActor(),
        moderation_provider=_FailingModeration(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a"},
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-1]["type"] == "error"
    assert events[-1]["data"] == "moderation backend unavailable"
    assert not any(event["type"] == "done" for event in events)


def test_missing_tenant_providers_do_not_fall_back_to_mocks(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=MockQuestionRefinementActor(),
    )
    app.state.tenant_manager = _EmptyTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a"},
        )

    events = parse_sse(response.text)
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert "moderation provider is not configured" in events[-1]["data"]
    assert not any(event["type"] == "done" for event in events)


def test_unknown_tenant_is_rejected_before_run_creation(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(langgraph_v2_migrated_database_url)
    app.state.tenant_manager = _UnknownTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "missing"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Tenant not found"}
