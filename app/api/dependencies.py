"""FastAPI dependency injection — tenant resolution and validation."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request

from app.services.exceptions import TenantNotFoundError
from app.services.tenant_manager import TenantManager


@dataclass
class TenantContext:
    """Resolved tenant context available to route handlers."""

    app_id: str
    manager: TenantManager


def get_tenant_manager(request: Request) -> TenantManager:
    """Retrieve the :class:`TenantManager` from app state."""
    return request.app.state.tenant_manager


async def get_tenant(
    x_application_id: str = Header(
        ..., alias="X-Application-Id", description="Tenant application ID"
    ),
    x_user_groups: str = Header(
        "", alias="X-User-Groups", description="Comma-separated AD groups from gateway"
    ),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
) -> TenantContext:
    """Resolve and validate a tenant from request headers.

    - Looks up the tenant by ``X-Application-Id``.
    - Validates user AD groups against the tenant's allowed groups.
    """
    try:
        # Check tenant exists
        tenant_manager.get_tenant_config(x_application_id)
    except TenantNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Tenant not found: {x_application_id}"
        )

    # Validate AD groups (skip if header is empty — for local dev)
    if x_user_groups:
        user_groups = [g.strip() for g in x_user_groups.split(",") if g.strip()]
        if not tenant_manager.validate_ad_group(x_application_id, user_groups):
            raise HTTPException(
                status_code=403, detail="Access denied: AD group mismatch"
            )

    return TenantContext(app_id=x_application_id, manager=tenant_manager)
