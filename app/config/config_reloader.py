"""Configuration hot-reloader with build-then-swap strategy.

Loads and validates a new ``config.json``, builds a complete new
:class:`TenantManager`, and atomically swaps it on ``app.state``.
In-flight requests continue using the old manager uninterrupted.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from fastapi import FastAPI

from app.config.loader import load_config
from app.config.models import LangGraphRuntimeMode, TenantConfig
from app.core.http_client_pool import HttpClientPool
from app.services.tenant_manager import TenantManager

logger = logging.getLogger(__name__)


class _TenantModeSource(Protocol):
    @property
    def tenant_ids(self) -> list[str]: ...

    def get_tenant_config(self, tenant_id: str) -> TenantConfig: ...


def _runtime_modes(manager: _TenantModeSource) -> dict[str, LangGraphRuntimeMode]:
    """Snapshot configured modes that must remain fixed for this process."""
    return {
        tenant_id: manager.get_tenant_config(tenant_id).runtime_mode
        for tenant_id in manager.tenant_ids
    }


@dataclass
class ReloadResult:
    """Outcome of a configuration reload attempt."""

    status: str  # "success" | "failed"
    reload_count: int = 0
    previous_tenants: list[str] = field(default_factory=list[str])
    current_tenants: list[str] = field(default_factory=list[str])
    timestamp: str = ""
    error: str | None = None


class ConfigReloader:
    """Thread-safe configuration hot-reloader.

    Strategy: **Build-then-swap** (blue-green)

    1. Load and validate new config from file.
    2. Build a completely new ``TenantManager`` with new configs.
    3. Atomically swap ``app.state.tenant_manager``.
    4. Old ``TenantManager``'s resources are garbage collected.

    Safety guarantees:

    - In-flight requests continue using the old ``TenantManager``.
    - New requests immediately use the new config.
    - If new config is invalid, swap never happens (safe rollback).
    - ``asyncio.Lock`` prevents concurrent reloads.
    """

    def __init__(self, app: FastAPI, http_pool: HttpClientPool) -> None:
        self._app = app
        self._http_pool = http_pool
        self._lock = asyncio.Lock()
        self._reload_count: int = 0
        self._last_reload: datetime | None = None
        self._fixed_runtime_modes = _runtime_modes(app.state.tenant_manager)

    async def reload(self, config_path: str = "config.json") -> ReloadResult:
        """Attempt to reload configuration.

        Returns:
            ReloadResult with status, tenant IDs, and any errors.
        """
        async with self._lock:
            old_manager: TenantManager = self._app.state.tenant_manager
            previous_tenants = old_manager.tenant_ids

            try:
                # 1. Load & validate
                new_configs = load_config(config_path)
                logger.info(
                    "Config reload: loaded %d tenant(s) from %s",
                    len(new_configs),
                    config_path,
                )
                for config in new_configs:
                    previous_mode = self._fixed_runtime_modes.get(
                        config.application_id
                    )
                    if previous_mode is None:
                        continue
                    if config.runtime_mode is not previous_mode:
                        raise ValueError(
                            "runtime_mode cannot change during config reload: "
                            f"{config.application_id}"
                        )

                # 2. Build new manager (may raise on invalid config)
                new_manager = TenantManager(new_configs, self._http_pool)

                # 3. Atomic swap
                self._app.state.tenant_manager = new_manager
                for config in new_configs:
                    self._fixed_runtime_modes.setdefault(
                        config.application_id,
                        config.runtime_mode,
                    )

                # 4. Record
                self._reload_count += 1
                self._last_reload = datetime.now(tz=UTC)

                result = ReloadResult(
                    status="success",
                    reload_count=self._reload_count,
                    previous_tenants=previous_tenants,
                    current_tenants=new_manager.tenant_ids,
                    timestamp=self._last_reload.isoformat(),
                )
                logger.info(
                    "Config reload #%d successful: %s → %s",
                    self._reload_count,
                    previous_tenants,
                    new_manager.tenant_ids,
                )
                return result

            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error("Config reload failed: %s", error_msg)
                return ReloadResult(
                    status="failed",
                    reload_count=self._reload_count,
                    previous_tenants=previous_tenants,
                    current_tenants=previous_tenants,  # unchanged
                    timestamp=datetime.now(tz=UTC).isoformat(),
                    error=error_msg,
                )

    @property
    def reload_count(self) -> int:
        """Total number of successful reloads."""
        return self._reload_count

    @property
    def last_reload(self) -> datetime | None:
        """Timestamp of the last successful reload."""
        return self._last_reload
