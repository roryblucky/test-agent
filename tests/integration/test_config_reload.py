"""Integration tests for configuration reload behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never

import pytest
from fastapi import FastAPI

from app.config.config_reloader import ConfigReloader
from app.config.models import FlowConfig, LLMConfig, TenantConfig
from app.core.http_client_pool import HttpClientPool


class FakeTenantManager:
    """Tiny tenant-manager double used by reload tests."""

    def __init__(self, configs: list[TenantConfig], http_pool: HttpClientPool) -> None:
        self.configs = configs
        self.http_pool = http_pool
        self.tenant_ids = [cfg.application_id for cfg in configs]


def _config(application_id: str) -> TenantConfig:
    return TenantConfig(
        kms_app_name=f"{application_id} app",
        application_id=application_id,
        ad_groups=[],
        llm_config=LLMConfig(models={}),
        flow_config=FlowConfig(),
    )


@pytest.mark.asyncio
async def test_config_reload_swaps_manager_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful reload replaces the active tenant manager."""
    app = FastAPI()
    app.state.tenant_manager = SimpleNamespace(tenant_ids=["old-tenant"])
    http_pool = HttpClientPool()

    def _load_new(_config_path: str | Path) -> list[TenantConfig]:
        return [_config("new-tenant")]

    monkeypatch.setattr(
        "app.config.config_reloader.load_config",
        _load_new,
    )
    monkeypatch.setattr(
        "app.config.config_reloader.TenantManager",
        FakeTenantManager,
    )

    reloader = ConfigReloader(app, http_pool)
    result = await reloader.reload("config.json")

    assert result.status == "success"
    assert result.previous_tenants == ["old-tenant"]
    assert result.current_tenants == ["new-tenant"]
    assert result.reload_count == 1
    assert app.state.tenant_manager.tenant_ids == ["new-tenant"]
    assert reloader.reload_count == 1
    assert reloader.last_reload is not None


@pytest.mark.asyncio
async def test_config_reload_keeps_old_manager_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed reload leaves the previous tenant manager in place."""
    app = FastAPI()
    old_manager = SimpleNamespace(tenant_ids=["old-tenant"])
    app.state.tenant_manager = old_manager
    http_pool = HttpClientPool()

    def _raise(*args: Any, **kwargs: Any) -> Never:
        raise ValueError("invalid config")

    def _load_broken(_config_path: str | Path) -> list[TenantConfig]:
        return [_config("broken-tenant")]

    monkeypatch.setattr(
        "app.config.config_reloader.load_config",
        _load_broken,
    )
    monkeypatch.setattr(
        "app.config.config_reloader.TenantManager",
        _raise,
    )

    reloader = ConfigReloader(app, http_pool)
    result = await reloader.reload("config.json")

    assert result.status == "failed"
    assert result.previous_tenants == ["old-tenant"]
    assert result.current_tenants == ["old-tenant"]
    assert "ValueError" in (result.error or "")
    assert app.state.tenant_manager is old_manager
    assert reloader.reload_count == 0
    assert reloader.last_reload is None
