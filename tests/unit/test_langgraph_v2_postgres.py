from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan


def test_v2_postgres_config_reads_explicit_bounded_pool_settings() -> None:
    config = V2PostgresConfig.from_environment(
        {
            "LANGGRAPH_V2_DATABASE_URL": "postgresql://app:secret@db/v2",
            "LANGGRAPH_V2_DATABASE_POOL_MIN_SIZE": "2",
            "LANGGRAPH_V2_DATABASE_POOL_MAX_SIZE": "12",
        }
    )

    assert config is not None
    assert config.conninfo == "postgresql://app:secret@db/v2"
    assert (config.min_size, config.max_size) == (2, 12)


def test_v2_postgres_config_is_disabled_without_an_explicit_url() -> None:
    assert V2PostgresConfig.from_environment({}) is None


def test_v2_postgres_config_rejects_an_inverted_pool_bound() -> None:
    with pytest.raises(ValidationError, match="max_size must be greater"):
        V2PostgresConfig.from_environment(
            {
                "LANGGRAPH_V2_DATABASE_URL": "postgresql://app:secret@db/v2",
                "LANGGRAPH_V2_DATABASE_POOL_MIN_SIZE": "4",
                "LANGGRAPH_V2_DATABASE_POOL_MAX_SIZE": "3",
            }
        )


@pytest.mark.asyncio
async def test_postgres_lifespan_opens_and_closes_the_configured_pool() -> None:
    app = FastAPI()
    pools: list[FakeAsyncPool] = []

    def pool_factory(**kwargs: Any) -> FakeAsyncPool:
        pool = FakeAsyncPool(**kwargs)
        pools.append(pool)
        return pool

    config = V2PostgresConfig(
        database_url="postgresql://app:secret@db/v2",
        min_size=2,
        max_size=12,
    )

    async with postgres_lifespan(app, config=config, pool_factory=pool_factory):
        assert app.state.langgraph_v2_postgres_pool is pools[0]
        assert pools[0].opened is True
        assert pools[0].options == {
            "conninfo": "postgresql://app:secret@db/v2",
            "min_size": 2,
            "max_size": 12,
            "open": False,
        }

    assert pools[0].closed is True
    assert app.state.langgraph_v2_postgres_pool is None


class FakeAsyncPool:
    """Controllable substitute for the external psycopg pool boundary."""

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.opened = False
        self.closed = False

    async def open(self, *, wait: bool) -> None:
        assert wait is True
        self.opened = True

    async def close(self) -> None:
        self.closed = True
