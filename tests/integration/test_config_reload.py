"""Integration tests for configuration reload behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from app.config.config_reloader import ConfigReloader


class FakeTenantManager:
    """Tiny tenant-manager double used by reload tests."""

    def __init__(self, configs, http_pool) -> None:
        self.configs = configs
        self.http_pool = http_pool
        self.tenant_ids = [cfg.kmsAppName for cfg in configs]


@pytest.mark.asyncio
async def test_config_reload_swaps_manager_atomically(monkeypatch) -> None:
    """A successful reload replaces the active tenant manager."""
    app = FastAPI()
    app.state.tenant_manager = SimpleNamespace(tenant_ids=["old-tenant"])
    http_pool = SimpleNamespace(label="pool")

    monkeypatch.setattr(
        "app.config.config_reloader.load_config",
        lambda config_path: [SimpleNamespace(kmsAppName="new-tenant")],
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
async def test_config_reload_keeps_old_manager_on_failure(monkeypatch) -> None:
    """A failed reload leaves the previous tenant manager in place."""
    app = FastAPI()
    old_manager = SimpleNamespace(tenant_ids=["old-tenant"])
    app.state.tenant_manager = old_manager
    http_pool = SimpleNamespace(label="pool")

    def _raise(*args, **kwargs):
        raise ValueError("invalid config")

    monkeypatch.setattr(
        "app.config.config_reloader.load_config",
        lambda config_path: [SimpleNamespace(kmsAppName="broken-tenant")],
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
