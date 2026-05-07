"""Admin API — configuration management and operational endpoints.

Separated from the main API router with a dedicated prefix.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config.config_reloader import ConfigReloader

admin_router = APIRouter(prefix="/admin", tags=["Admin"])


class ReloadResponse(BaseModel):
    """Response from a config reload operation."""

    status: str
    reload_count: int = Field(alias="reloadCount")
    previous_tenants: list[str] = Field(alias="previousTenants")
    current_tenants: list[str] = Field(alias="currentTenants")
    timestamp: str = ""
    error: str | None = None

    model_config = {"populate_by_name": True}


class ConfigStatusResponse(BaseModel):
    """Current configuration status."""

    tenant_count: int = Field(alias="tenantCount")
    tenant_ids: list[str] = Field(alias="tenantIds")
    reload_count: int = Field(alias="reloadCount")
    last_reload: str | None = Field(None, alias="lastReload")

    model_config = {"populate_by_name": True}


@admin_router.post("/reload", response_model=ReloadResponse)
async def reload_config(request: Request) -> ReloadResponse:
    """Hot-reload tenant configuration from config.json.

    Validates new config, builds a new TenantManager, and atomically
    swaps it.  In-flight requests are not interrupted.
    """
    reloader: ConfigReloader | None = getattr(
        request.app.state, "config_reloader", None
    )
    if not reloader:
        raise HTTPException(500, "Config reloader not initialized")

    result = await reloader.reload()
    return ReloadResponse(
        status=result.status,
        reload_count=result.reload_count,
        previous_tenants=result.previous_tenants,
        current_tenants=result.current_tenants,
        timestamp=result.timestamp,
        error=result.error,
    )


@admin_router.get("/config/status", response_model=ConfigStatusResponse)
async def config_status(request: Request) -> ConfigStatusResponse:
    """Return current configuration metadata."""
    tenant_manager = request.app.state.tenant_manager
    reloader: ConfigReloader | None = getattr(
        request.app.state, "config_reloader", None
    )

    return ConfigStatusResponse(
        tenant_count=len(tenant_manager.tenant_ids),
        tenant_ids=tenant_manager.tenant_ids,
        reload_count=reloader.reload_count if reloader else 0,
        last_reload=(
            reloader.last_reload.isoformat()
            if reloader and reloader.last_reload
            else None
        ),
    )
