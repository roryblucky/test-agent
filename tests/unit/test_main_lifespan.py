from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI

import app.api.router as router_module
import app.config.config_reloader as config_reloader_module
import app.core.audit as audit_module
import app.core.rate_limiter as rate_limiter_module
import app.core.telemetry as telemetry_module
import app.langgraph_v2.output_assessments as output_assessments_module
import app.main as main_module
from app.langgraph_v2.agent_batch import (
    AgentToolRegistry,
    EvidenceToolRegistration,
)
from app.langgraph_v2.agent_evidence import EvidenceEnvelope


class _AsyncCloseTracker:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _HttpPool(_AsyncCloseTracker):
    async def close_all(self) -> None:
        await self.close()


@pytest.mark.asyncio
async def test_lifespan_checks_the_pinned_pydantic_ai_version_before_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def require_version() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("PydanticAI version is not pinned")

    monkeypatch.setattr(
        main_module, "require_pinned_pydantic_ai_version", require_version
    )

    with pytest.raises(RuntimeError, match="not pinned"):
        async with main_module.lifespan(FastAPI()):
            pass

    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_from_body", [False, True])
async def test_lifespan_always_closes_bigquery_assessment_audit(
    monkeypatch: pytest.MonkeyPatch,
    raise_from_body: bool,
) -> None:
    assessment_audit = _AsyncCloseTracker()
    audit_logger = _AsyncCloseTracker()
    rate_limiter = _AsyncCloseTracker()
    session_store = _AsyncCloseTracker()
    http_pool = _HttpPool()
    specialist_catalogs = {"tenant-a": object()}
    specialist_load_calls: list[
        tuple[object, Path, object, dict[str, frozenset[str]]]
    ] = []

    class AssessmentAuditFactory:
        def __new__(cls, *, project_id: str) -> _AsyncCloseTracker:
            assert project_id == "project-a"
            return assessment_audit

    @asynccontextmanager
    async def postgres_lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        yield

    class Manager:
        tenant_ids = ["tenant-a"]

        def get_agent_tool_ids(self, tenant_id: str) -> frozenset[str]:
            assert tenant_id == "tenant-a"
            return frozenset({"filing_reader", "unregistered"})

    manager_instance = Manager()

    def tenant_manager(_configs: object, _http_pool: object) -> Manager:
        return manager_instance

    async def load_local_specialist_catalogs(
        manager: object,
        *,
        root: Path,
        tool_registry: object,
        tenant_allowed_tool_ids: dict[str, frozenset[str]],
    ) -> dict[str, object]:
        specialist_load_calls.append(
            (manager, root, tool_registry, tenant_allowed_tool_ids)
        )
        return specialist_catalogs

    def audit_logger_factory(*, sinks: list[object]) -> _AsyncCloseTracker:
        assert len(sinks) == 2
        return audit_logger

    def http_pool_factory() -> _HttpPool:
        return http_pool

    def load_config(_path: str) -> dict[str, object]:
        return {}

    def config_reloader(*_args: object) -> None:
        return None

    def telemetry_service(_name: str) -> None:
        return None

    def rate_limiter_factory(_url: str | None) -> _AsyncCloseTracker:
        return rate_limiter

    def file_audit_sink() -> object:
        return object()

    def bigquery_audit_sink(*, project_id: str) -> object:
        assert project_id == "project-a"
        return object()

    def session_store_factory() -> _AsyncCloseTracker:
        return session_store

    monkeypatch.setenv("GCP_PROJECT_ID", "project-a")
    monkeypatch.setenv("SPECIALIST_DEFINITIONS_LOCAL_ROOT", "/definitions")
    monkeypatch.setattr(main_module, "HttpClientPool", http_pool_factory)
    monkeypatch.setattr(main_module, "load_config", load_config)
    monkeypatch.setattr(main_module, "TenantManager", tenant_manager)
    monkeypatch.setattr(
        main_module,
        "load_local_specialist_catalogs",
        load_local_specialist_catalogs,
        raising=False,
    )
    monkeypatch.setattr(main_module, "postgres_lifespan", postgres_lifespan)
    monkeypatch.setattr(config_reloader_module, "ConfigReloader", config_reloader)
    monkeypatch.setattr(telemetry_module, "TelemetryService", telemetry_service)
    monkeypatch.setattr(
        rate_limiter_module, "create_rate_limiter", rate_limiter_factory
    )
    monkeypatch.setattr(audit_module, "FileAuditSink", file_audit_sink)
    monkeypatch.setattr(audit_module, "BigQueryAuditSink", bigquery_audit_sink)
    monkeypatch.setattr(audit_module, "AuditLogger", audit_logger_factory)
    monkeypatch.setattr(
        output_assessments_module,
        "BigQueryOutputAssessmentAudit",
        AssessmentAuditFactory,
    )
    monkeypatch.setattr(router_module, "get_session_store", session_store_factory)

    lifespan_app = FastAPI()
    async def filing_provider(_source: str, _query: str) -> EvidenceEnvelope:
        raise AssertionError("startup must not execute business Tools")

    expected_tool_registry = AgentToolRegistry(
        evidence_registrations=(
            EvidenceToolRegistration(
                id="filing_reader",
                provider=filing_provider,
                allowed_sources=frozenset(),
            ),
        )
    )
    configured_lifespan = main_module.create_lifespan(expected_tool_registry)
    if raise_from_body:
        with pytest.raises(RuntimeError, match="body failed"):
            async with configured_lifespan(lifespan_app):
                assert (
                    lifespan_app.state.langgraph_v2_specialist_catalogs
                    == specialist_catalogs
                )
                raise RuntimeError("body failed")
    else:
        async with configured_lifespan(lifespan_app):
            assert (
                lifespan_app.state.langgraph_v2_specialist_catalogs
                == specialist_catalogs
            )

    assert len(specialist_load_calls) == 1
    manager, root, tool_registry, tenant_tool_ids = specialist_load_calls[0]
    assert manager is lifespan_app.state.tenant_manager
    assert root == Path("/definitions")
    assert tool_registry is expected_tool_registry
    assert tenant_tool_ids == {"tenant-a": frozenset({"filing_reader"})}
    assert assessment_audit.closed
    assert audit_logger.closed
    assert rate_limiter.closed
    assert session_store.closed
    assert http_pool.closed
