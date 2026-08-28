"""PostgreSQL configuration and lifecycle for the v2 runtime."""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Self

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field, model_validator

from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.run_events import RunEventRepository

_INSTANCE_ID = os.environ.get("LANGGRAPH_V2_INSTANCE_ID", socket.gethostname())


class V2PostgresConfig(BaseModel):
    """Explicit configuration for the bounded v2 PostgreSQL pool."""

    database_url: str = Field(min_length=1)
    min_size: int = Field(default=1, ge=0)
    max_size: int = Field(default=10, gt=0)

    @property
    def conninfo(self) -> str:
        """Return the connection string expected by psycopg."""
        return self.database_url

    @model_validator(mode="after")
    def validate_pool_bounds(self) -> Self:
        """Reject configurations whose lower bound exceeds the upper bound."""
        if self.max_size < self.min_size:
            raise ValueError("max_size must be greater than or equal to min_size")
        return self

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] = os.environ,
    ) -> Self | None:
        """Load v2 settings only when its dedicated database URL is present."""
        database_url = environment.get("LANGGRAPH_V2_DATABASE_URL")
        if not database_url:
            return None
        return cls.model_validate(
            {
                "database_url": database_url,
                "min_size": environment.get("LANGGRAPH_V2_DATABASE_POOL_MIN_SIZE", "1"),
                "max_size": environment.get(
                    "LANGGRAPH_V2_DATABASE_POOL_MAX_SIZE", "10"
                ),
            }
        )


PoolFactory = Callable[..., Any]
CheckpointerFactory = Callable[[Any], AsyncPostgresSaver]


@asynccontextmanager
async def postgres_lifespan(
    app: FastAPI,
    *,
    config: V2PostgresConfig | None = None,
    pool_factory: PoolFactory = AsyncConnectionPool,
    checkpointer_factory: CheckpointerFactory = AsyncPostgresSaver,
) -> AsyncIterator[None]:
    """Open the configured pool for the application lifespan and always close it."""
    resolved_config = config or V2PostgresConfig.from_environment()
    app.state.langgraph_v2_postgres_pool = None
    app.state.langgraph_v2_checkpointer = None
    app.state.langgraph_v2_live_events = LiveEventWakeups(
        redis_url=os.environ.get("LANGGRAPH_V2_REDIS_URL"),
        instance_id=_INSTANCE_ID,
    )
    wakeups = app.state.langgraph_v2_live_events
    pool: Any | None = None
    try:
        await wakeups.start()
        if resolved_config is None:
            yield
            return

        pool = pool_factory(
            conninfo=resolved_config.conninfo,
            min_size=resolved_config.min_size,
            max_size=resolved_config.max_size,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await pool.open(wait=True)
        app.state.langgraph_v2_postgres_pool = pool
        checkpointer = checkpointer_factory(pool)
        await checkpointer.setup()
        app.state.langgraph_v2_checkpointer = checkpointer
        yield
    finally:
        try:
            if pool is not None:
                try:
                    runtime = getattr(app.state, "langgraph_v2_runtime", None)
                    if (
                        runtime is not None
                        and app.state.langgraph_v2_postgres_pool is pool
                    ):
                        await runtime.stop_and_wait_for_checkpoint_boundary()
                        await RunEventRepository(pool).interrupt_runs_owned_by(
                            _INSTANCE_ID
                        )
                finally:
                    await pool.close()
        finally:
            app.state.langgraph_v2_postgres_pool = None
            app.state.langgraph_v2_checkpointer = None
            await wakeups.close()
