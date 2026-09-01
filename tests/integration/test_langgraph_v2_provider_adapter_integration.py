from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config.models import (
    FlowConfig,
    GroundednessConfig,
    LLMConfig,
    ModerationConfig,
    RankingConfig,
    RetrieverConfig,
    RetrieverSourceConfig,
    TenantConfig,
)
from app.core.http_client_pool import HttpClientPool
from app.langgraph_v2.answer import AnswerResult, AnswerStreamChunk
from app.langgraph_v2.history import ConversationExchange
from app.langgraph_v2.pre_moderation import ModerationDecision
from app.langgraph_v2.question_refinement import MockQuestionRefinementActor
from app.models.domain import Document, GroundednessResult, ModerationResult
from app.providers.base import (
    BaseGroundednessProvider,
    BaseModerationProvider,
    BaseRankerProvider,
    BaseRetrieverProvider,
)
from app.providers.factory import ProviderFactory
from app.services.exceptions import TenantNotFoundError
from app.services.tenant_manager import TenantManager
from tests.integration.test_langgraph_v2_linear_core import (
    LinearTenantManager,
    parse_sse,
    persistent_linear_app,
    seed_subject_conversation,
)


class _FailingModeration:
    async def check(self, text: str) -> ModerationDecision:
        del text
        raise RuntimeError("moderation backend unavailable")


class _EmptyTenantManager(LinearTenantManager):
    def get_providers(self, tenant_id: str):
        assert tenant_id == "tenant-a"
        return SimpleNamespace(
            retriever=None,
            ranker=None,
            moderation=None,
            groundedness=None,
        )


class _UnknownTenantManager:
    def get_providers(self, tenant_id: str):
        raise TenantNotFoundError(tenant_id)


class _RecordingRetriever(BaseRetrieverProvider):
    def __init__(self, name: str, calls: list[tuple[str, int]]) -> None:
        self._name = name
        self._calls = calls

    async def retrieve(
        self, query: str, top_k: int = 10, filter_expr: str | None = None
    ) -> list[Document]:
        del query, filter_expr
        self._calls.append((self._name, top_k))
        return [Document(id=self._name, content=self._name)]


class _RecordingRanker(BaseRankerProvider):
    def __init__(self, calls: list[int]) -> None:
        self._calls = calls

    async def rank(
        self, query: str, documents: list[Document], top_n: int = 5
    ) -> list[Document]:
        del query
        self._calls.append(top_n)
        return documents[:top_n]


class _SafeModeration(BaseModerationProvider):
    async def check(self, text: str) -> ModerationResult:
        del text
        return ModerationResult(is_flagged=False)


class _RecordingGroundedness(BaseGroundednessProvider):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def check(
        self, answer: str, context: list[Document]
    ) -> GroundednessResult:
        del context
        self._calls.append(answer)
        return GroundednessResult(is_grounded=True, score=0.9)


class _AnswerActor:
    async def answer_stream(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AsyncIterator[AnswerStreamChunk]:
        del query, documents, history
        yield AnswerStreamChunk(delta="answer")
        yield AnswerStreamChunk(result=AnswerResult(answer="answer"))


def test_configured_provider_failure_is_persisted_as_v2_error(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=MockQuestionRefinementActor(),
        moderation_provider=_FailingModeration(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-1]["type"] == "error"
    assert events[-1]["data"] == "moderation backend unavailable"
    assert not any(event["type"] == "done" for event in events)


def test_missing_tenant_providers_do_not_fall_back_to_mocks(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=MockQuestionRefinementActor(),
    )
    app.state.tenant_manager = _EmptyTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    events = parse_sse(response.text)
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert "moderation provider is not configured" in events[-1]["data"]
    assert not any(event["type"] == "done" for event in events)


def test_unknown_tenant_is_rejected_before_run_creation(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(langgraph_v2_migrated_database_url)
    app.state.tenant_manager = _UnknownTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "missing", "X-Subject-Id": "subject-a"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Tenant not found"}


def test_tenant_config_limits_reach_v2_retriever_and_ranker(
    langgraph_v2_migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever_calls: list[tuple[str, int]] = []
    ranker_calls: list[int] = []
    groundedness_calls: list[str] = []
    tenant_config = TenantConfig(
        kms_app_name="tenant-a",
        application_id="tenant-a",
        ad_groups=[],
        llm_config=LLMConfig(models={}),
        retriever_config=RetrieverConfig(
            sources=[
                RetrieverSourceConfig(provider="first", top_k=2),
                RetrieverSourceConfig(provider="second", top_k=7),
            ]
        ),
        ranking_config=RankingConfig(provider="ranker", top_n=3),
        moderation_config=ModerationConfig(provider="moderation"),
        groundedness_config=GroundednessConfig(provider="groundedness"),
        flow_config=FlowConfig(),
    )

    def create_provider(
        component: str,
        provider: str,
        config: object,
        cloud_config: object,
        http_pool: object,
    ) -> Any:
        del config, cloud_config, http_pool
        if component == "retriever":
            return _RecordingRetriever(provider, retriever_calls)
        if component == "ranker":
            return _RecordingRanker(ranker_calls)
        if component == "moderation":
            return _SafeModeration()
        if component == "groundedness":
            return _RecordingGroundedness(groundedness_calls)
        raise AssertionError(component)

    monkeypatch.setattr(ProviderFactory, "create", staticmethod(create_provider))
    tenant_manager = TenantManager([tenant_config], HttpClientPool())

    class ProviderOnlyTenantManager:
        def get_providers(self, tenant_id: str):
            return tenant_manager.get_providers(tenant_id)

        def get_tenant_config(self, tenant_id: str) -> TenantConfig:
            return tenant_manager.get_tenant_config(tenant_id)

    asyncio.run(
        seed_subject_conversation(
            langgraph_v2_migrated_database_url,
            "00000000-0000-0000-0000-000000000001",
        )
    )
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=MockQuestionRefinementActor(),
        answer_actor=_AnswerActor(),
    )
    app.state.tenant_manager = ProviderOnlyTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "00000000-0000-0000-0000-000000000001"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert response.status_code == 200
    assert parse_sse(response.text)[-1]["type"] == "done"
    assert retriever_calls == [("first", 2), ("second", 7)]
    assert ranker_calls == [3]
    assert groundedness_calls == ["answer"]
