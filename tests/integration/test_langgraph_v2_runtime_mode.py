from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config.models import (
    FlowConfig,
    LangGraphRuntimeMode,
    LLMConfig,
    TenantConfig,
)
from app.langgraph_v2.api import GraphRuntimeAdapter
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.stream import RequestOwnedGraph
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)


class _AgentGraph:
    def __init__(self) -> None:
        self.inputs: list[Mapping[str, Any]] = []

    def astream(
        self,
        graph_input: Mapping[str, Any],
        /,
        **options: Any,
    ) -> AsyncIterator[object]:
        del options
        self.inputs.append(graph_input)

        async def stream() -> AsyncIterator[object]:
            yield (
                "custom",
                {
                    "type": "done",
                    "data": {"answer": "agent answer"},
                    "checkpoint_terminal": True,
                },
            )
            yield ("updates", {"agent_finalization": {"answer": "agent answer"}})

        return stream()


class _AgentRuntime:
    def __init__(self) -> None:
        self.graph = _AgentGraph()

    @property
    def runtime_mode(self) -> LangGraphRuntimeMode:
        return LangGraphRuntimeMode.AGENT

    @property
    def checkpoint_state_adapter(self) -> _EmptyCheckpointStateAdapter:
        return _EmptyCheckpointStateAdapter()

    def build_graph(self, *, request_id: str) -> RequestOwnedGraph:
        del request_id
        return self.graph

    def initial_state_fields(
        self,
        *,
        payload: V2QueryRequest,
    ) -> Mapping[str, Any]:
        del payload
        return {}


class _EmptyCheckpointStateAdapter:
    def validate_checkpoint_state(
        self,
        channel_values: Mapping[str, object],
    ) -> list[BaseMessage]:
        if channel_values:
            raise TypeError("test Agent checkpoint state is invalid")
        return []


class _AgentRuntimeFactory:
    def __init__(self, runtime: _AgentRuntime) -> None:
        self.runtime = runtime

    def __call__(
        self,
        *,
        app: FastAPI,
        request_context: TrustedRequestContext,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> GraphRuntimeAdapter:
        del app, request_context, checkpointer
        return self.runtime


class _IdentityOverridingAgentRuntime(_AgentRuntime):
    def initial_state_fields(
        self,
        *,
        payload: V2QueryRequest,
    ) -> Mapping[str, Any]:
        del payload
        return {"request_id": "forged-request"}


class _AgentTenantManager:
    def __init__(self) -> None:
        self.config = TenantConfig(
            kms_app_name="Agent Tenant",
            application_id="tenant-a",
            ad_groups=[],
            runtime_mode=LangGraphRuntimeMode.AGENT,
            llm_config=LLMConfig(models={}),
            flow_config=FlowConfig(),
        )

    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        assert tenant_id == "tenant-a"
        return self.config

    def get_providers(self, tenant_id: str) -> object:
        assert tenant_id == "tenant-a"
        return object()


def test_agent_tenant_query_ignores_client_mode_override(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = UUID("00000000-0000-0000-0000-000000000031")
    runtime = _AgentRuntime()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=_AgentRuntimeFactory(runtime),
    )
    app.state.tenant_manager = _AgentTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={
                "query": "agent query",
                "sessionId": str(conversation_id),
                "mode": "linear",
            },
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert response.status_code == 200
    assert parse_sse(response.text)[-1]["data"]["answer"] == "agent answer"
    assert runtime.graph.inputs == [
        {
            "query": "agent query",
            "conversation_id": str(conversation_id),
            "request_id": response.headers["X-Request-Id"],
            "conversation_messages": [],
        }
    ]


def test_agent_runtime_cannot_override_shared_query_identity(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = UUID("00000000-0000-0000-0000-000000000032")
    runtime = _IdentityOverridingAgentRuntime()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        agent_runtime_factory=_AgentRuntimeFactory(runtime),
    )
    app.state.tenant_manager = _AgentTenantManager()

    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "agent query", "sessionId": str(conversation_id)},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )

    assert response.status_code == 200
    assert parse_sse(response.text) == [
        {
            "type": "error",
            "data": "Runtime redefined shared Query state: request_id",
        }
    ]
    assert runtime.graph.inputs == []


def test_query_requires_trusted_tenant_runtime_configuration(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        del app.state.tenant_manager
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Tenant runtime configuration is not available"
    }


def test_query_requires_official_postgres_checkpointer(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(langgraph_v2_migrated_database_url)

    with TestClient(app) as client:
        app.state.langgraph_v2_checkpointer = None
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "LangGraph v2 checkpointer is not configured"
    }
