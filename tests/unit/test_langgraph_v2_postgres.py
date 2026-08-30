from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

import app.langgraph_v2.postgres as postgres_module
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan


def test_v2_postgres_config_reads_explicit_bounded_pool_settings() -> None:
    config = V2PostgresConfig.from_environment(
        {
            "LANGGRAPH_V2_DATABASE_URL": "postgresql://app:secret@db/v2",
            "LANGGRAPH_V2_DATABASE_POOL_MIN_SIZE": "2",
            "LANGGRAPH_V2_DATABASE_POOL_MAX_SIZE": "12",
            "LANGGRAPH_V2_RESUME_TTL_SECONDS": "900",
        }
    )

    assert config is not None
    assert config.conninfo == "postgresql://app:secret@db/v2"
    assert (config.min_size, config.max_size) == (2, 12)
    assert config.resume_ttl_seconds == 900


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
    checkpointers: list[FakeCheckpointer] = []

    def pool_factory(**kwargs: Any) -> FakeAsyncPool:
        pool = FakeAsyncPool(**kwargs)
        pools.append(pool)
        return pool

    def checkpointer_factory(pool: Any) -> FakeCheckpointer:
        checkpointer = FakeCheckpointer(pool)
        checkpointers.append(checkpointer)
        return checkpointer

    config = V2PostgresConfig(
        database_url="postgresql://app:secret@db/v2",
        min_size=2,
        max_size=12,
    )

    async with postgres_lifespan(
        app,
        config=config,
        pool_factory=pool_factory,
        checkpointer_factory=checkpointer_factory,
    ):
        assert app.state.langgraph_v2_postgres_pool is pools[0]
        assert app.state.langgraph_v2_checkpointer is checkpointers[0]
        assert pools[0].opened is True
        assert checkpointers[0].setup_called is True
        assert pools[0].options == {
            "conninfo": "postgresql://app:secret@db/v2",
            "min_size": 2,
            "max_size": 12,
            "kwargs": {"autocommit": True, "prepare_threshold": 0},
            "open": False,
        }

    assert pools[0].closed is True
    assert app.state.langgraph_v2_postgres_pool is None
    assert app.state.langgraph_v2_checkpointer is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["pool_open", "checkpointer_setup"])
async def test_postgres_startup_failure_closes_all_started_resources(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = FastAPI()
    pool = FailingAsyncPool(fail_open=failure == "pool_open")
    wakeups = FakeWakeups()

    def wakeups_factory(*, redis_url: str | None, instance_id: str) -> FakeWakeups:
        return wakeups

    def pool_factory(**kwargs: Any) -> FailingAsyncPool:
        return pool

    def checkpointer_factory(pool: Any) -> FailingCheckpointer:
        return FailingCheckpointer(fail_setup=failure == "checkpointer_setup")

    monkeypatch.setattr(
        postgres_module,
        "LiveEventWakeups",
        wakeups_factory,
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url="postgresql://app:secret@db/v2"),
            pool_factory=pool_factory,
            checkpointer_factory=checkpointer_factory,
        ):
            pass

    assert wakeups.started is True
    assert wakeups.closed is True
    assert pool.closed is True
    assert app.state.langgraph_v2_postgres_pool is None
    assert app.state.langgraph_v2_checkpointer is None


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


class FakeCheckpointer:
    """Controllable substitute for the official saver setup boundary."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool
        self.setup_called = False

    async def setup(self) -> None:
        self.setup_called = True


class FailingAsyncPool(FakeAsyncPool):
    def __init__(self, *, fail_open: bool) -> None:
        super().__init__()
        self._fail_open = fail_open

    async def open(self, *, wait: bool) -> None:
        await super().open(wait=wait)
        if self._fail_open:
            raise RuntimeError("startup failed")


class FailingCheckpointer(FakeCheckpointer):
    def __init__(self, *, fail_setup: bool) -> None:
        super().__init__(None)
        self._fail_setup = fail_setup

    async def setup(self) -> None:
        await super().setup()
        if self._fail_setup:
            raise RuntimeError("startup failed")


class FakeWakeups:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True
