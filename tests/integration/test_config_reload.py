"""Integration tests for configuration reload behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Never

import pytest
from fastapi import FastAPI

from app.config.config_reloader import ConfigReloader
from app.config.models import (
    FlowConfig,
    LangGraphRuntimeMode,
    LLMConfig,
    TenantConfig,
)
from app.core.http_client_pool import HttpClientPool


class FakeTenantManager:
    """Tiny tenant-manager double used by reload tests."""

    def __init__(self, configs: list[TenantConfig], http_pool: HttpClientPool) -> None:
        self.configs = configs
        self.http_pool = http_pool
        self.tenant_ids = [cfg.application_id for cfg in configs]

    def get_tenant_config(self, application_id: str) -> TenantConfig:
        return next(
            config
            for config in self.configs
            if config.application_id == application_id
        )


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
    http_pool = HttpClientPool()
    app.state.tenant_manager = FakeTenantManager([_config("old-tenant")], http_pool)

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
    http_pool = HttpClientPool()
    old_manager = FakeTenantManager([_config("old-tenant")], http_pool)
    app.state.tenant_manager = old_manager

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


@pytest.mark.asyncio
async def test_config_reload_rejects_runtime_mode_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    http_pool = HttpClientPool()
    old_manager = FakeTenantManager([_config("tenant-a")], http_pool)
    app.state.tenant_manager = old_manager
    changed = _config("tenant-a").model_copy(
        update={"runtime_mode": LangGraphRuntimeMode.AGENT}
    )

    def _load_changed(_config_path: str | Path) -> list[TenantConfig]:
        return [changed]

    monkeypatch.setattr(
        "app.config.config_reloader.load_config",
        _load_changed,
    )
    monkeypatch.setattr(
        "app.config.config_reloader.TenantManager",
        FakeTenantManager,
    )

    result = await ConfigReloader(app, http_pool).reload("config.json")

    assert result.status == "failed"
    assert "runtime_mode cannot change" in (result.error or "")
    assert app.state.tenant_manager is old_manager


@pytest.mark.asyncio
async def test_config_reload_cannot_change_mode_by_removing_and_readding_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    http_pool = HttpClientPool()
    app.state.tenant_manager = FakeTenantManager([_config("tenant-a")], http_pool)
    agent_config = _config("tenant-a").model_copy(
        update={"runtime_mode": LangGraphRuntimeMode.AGENT}
    )
    loaded_configs = iter([[], [agent_config]])

    def _load_next(_config_path: str | Path) -> list[TenantConfig]:
        return next(loaded_configs)

    monkeypatch.setattr("app.config.config_reloader.load_config", _load_next)
    monkeypatch.setattr(
        "app.config.config_reloader.TenantManager",
        FakeTenantManager,
    )
    reloader = ConfigReloader(app, http_pool)

    removed = await reloader.reload("config.json")
    readded = await reloader.reload("config.json")

    assert removed.status == "success"
    assert readded.status == "failed"
    assert "runtime_mode cannot change" in (readded.error or "")
    assert app.state.tenant_manager.tenant_ids == []
