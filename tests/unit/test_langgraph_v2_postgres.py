from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import ValidationError

from app.langgraph_v2.postgres import (
    V2PostgresConfig,
    postgres_lifespan,
    strict_checkpoint_serializer,
)


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
    checkpointers: list[FakeCheckpointer] = []

    def pool_factory(**kwargs: Any) -> FakeAsyncPool:
        pool = FakeAsyncPool(**kwargs)
        pools.append(pool)
        return pool

    def checkpointer_factory(conn: Any, **_kwargs: Any) -> FakeCheckpointer:
        checkpointer = FakeCheckpointer(conn)
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


def test_strict_checkpoint_serializer_round_trips_json_and_messages() -> None:
    serializer = strict_checkpoint_serializer()

    json_value = {"nested": ["value", {"count": 2}]}
    message = HumanMessage(content="saved conversation")

    assert isinstance(serializer, JsonPlusSerializer)
    assert serializer.loads_typed(serializer.dumps_typed(json_value)) == json_value
    restored_message = serializer.loads_typed(serializer.dumps_typed(message))
    assert isinstance(restored_message, HumanMessage)
    assert restored_message.text == message.text


def test_strict_checkpoint_serializer_rejects_pickle_payloads() -> None:
    serializer = strict_checkpoint_serializer()

    with pytest.raises(NotImplementedError, match="Unknown serialization type: pickle"):
        serializer.loads_typed(("pickle", b"not a pickle"))


@pytest.mark.asyncio
async def test_postgres_lifespan_injects_the_explicit_strict_serializer() -> None:
    app = FastAPI()
    observed: list[JsonPlusSerializer] = []

    def pool_factory(**kwargs: Any) -> FakeAsyncPool:
        return FakeAsyncPool(**kwargs)

    def checkpointer_factory(
        conn: Any,
        *,
        serde: JsonPlusSerializer,
    ) -> FakeCheckpointer:
        observed.append(serde)
        return FakeCheckpointer(conn)

    async with postgres_lifespan(
        app,
        config=V2PostgresConfig(database_url="postgresql://app:secret@db/v2"),
        pool_factory=pool_factory,
        checkpointer_factory=checkpointer_factory,
    ):
        pass

    assert len(observed) == 1
    assert observed[0].pickle_fallback is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["pool_open", "checkpointer_setup"])
async def test_postgres_startup_failure_closes_all_started_resources(
    failure: str,
) -> None:
    app = FastAPI()
    pool = FailingAsyncPool(fail_open=failure == "pool_open")

    def pool_factory(**kwargs: Any) -> FailingAsyncPool:
        return pool

    def checkpointer_factory(conn: Any, **_kwargs: Any) -> FailingCheckpointer:
        del conn
        return FailingCheckpointer(fail_setup=failure == "checkpointer_setup")

    with pytest.raises(RuntimeError, match="startup failed"):
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url="postgresql://app:secret@db/v2"),
            pool_factory=pool_factory,
            checkpointer_factory=checkpointer_factory,
        ):
            pass

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
