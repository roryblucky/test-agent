"""RAG KMS Application — FastAPI entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.router import router
from app.config.loader import load_config
from app.core.http_client_pool import HttpClientPool
from app.services.tenant_manager import TenantManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — initialise and tear down shared resources."""
    # Startup
    http_pool = HttpClientPool()
    configs = load_config("config.json")
    app.state.tenant_manager = TenantManager(configs, http_pool)
    app.state.http_pool = http_pool

    yield

    # Shutdown
    await http_pool.close_all()


app = FastAPI(
    title="RAG KMS",
    description="Multi-tenant RAG Knowledge Management System",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
