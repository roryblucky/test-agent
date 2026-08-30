from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

import app.api.router as router_module
import app.config.config_reloader as config_reloader_module
import app.core.audit as audit_module
import app.core.rate_limiter as rate_limiter_module
import app.core.telemetry as telemetry_module
import app.langgraph_v2.output_assessments as output_assessments_module
import app.main as main_module


class _AsyncCloseTracker:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _HttpPool(_AsyncCloseTracker):
    async def close_all(self) -> None:
        await self.close()


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

    class AssessmentAuditFactory:
        def __new__(cls, *, project_id: str) -> _AsyncCloseTracker:
            assert project_id == "project-a"
            return assessment_audit

    @asynccontextmanager
    async def postgres_lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        yield

    def tenant_manager(_configs: object, _http_pool: object) -> SimpleNamespace:
        return SimpleNamespace(tenant_ids=[])

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
    monkeypatch.setattr(main_module, "HttpClientPool", http_pool_factory)
    monkeypatch.setattr(main_module, "load_config", load_config)
    monkeypatch.setattr(main_module, "TenantManager", tenant_manager)
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

    if raise_from_body:
        with pytest.raises(RuntimeError, match="body failed"):
            async with main_module.lifespan(FastAPI()):
                raise RuntimeError("body failed")
    else:
        async with main_module.lifespan(FastAPI()):
            pass

    assert assessment_audit.closed
    assert audit_logger.closed
    assert rate_limiter.closed
    assert session_store.closed
    assert http_pool.closed
