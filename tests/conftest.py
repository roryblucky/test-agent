"""Pytest configuration and fixtures."""

import os
import unittest.mock
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from psycopg import sql

from alembic import command
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.migrations import build_alembic_config
from app.providers.base import BaseRankerProvider, BaseRetrieverProvider
from app.services.events import EventEmitter
from app.services.flow_context import FlowContext
from tests.postgres import (
    require_disposable_postgres_url,
    scoped_disposable_postgres_url,
)


@pytest.fixture(scope="session")
def langgraph_v2_test_database_url() -> Iterator[str]:
    """Provide an empty test-only database and remove migration artifacts."""
    database_url = require_disposable_postgres_url(os.environ)
    schema_name = os.environ.get("LANGGRAPH_V2_TEST_SCHEMA", "")
    scoped_url = scoped_disposable_postgres_url(database_url, schema_name)
    try:
        with psycopg.connect(database_url, connect_timeout=3) as connection:
            if schema_name:
                connection.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(schema_name)
                    )
                )
                connection.commit()
        with psycopg.connect(scoped_url, connect_timeout=3) as connection:
            existing_object = connection.execute(
                """
                SELECT object_kind, object_name
                FROM (
                    SELECT 'relation' AS object_kind, c.relname AS object_name
                    FROM pg_class AS c
                    JOIN pg_namespace AS n ON n.oid = c.relnamespace
                    WHERE n.nspname = current_schema()
                    UNION ALL
                    SELECT 'function' AS object_kind, p.proname AS object_name
                    FROM pg_proc AS p
                    JOIN pg_namespace AS n ON n.oid = p.pronamespace
                    WHERE n.nspname = current_schema()
                    UNION ALL
                    SELECT 'type' AS object_kind, t.typname AS object_name
                    FROM pg_type AS t
                    JOIN pg_namespace AS n ON n.oid = t.typnamespace
                    WHERE n.nspname = current_schema()
                ) AS user_objects
                LIMIT 1
                """
            ).fetchone()
    except psycopg.OperationalError as error:
        pytest.fail(
            "LANGGRAPH_V2_TEST_DATABASE_URL points to an unavailable disposable "
            f"PostgreSQL database: {error}"
        )
    if existing_object is not None:
        pytest.fail(
            "LANGGRAPH_V2_TEST_DATABASE_URL must point to an empty disposable "
            f"database; found {existing_object[0]} {existing_object[1]!r}."
        )

    try:
        yield scoped_url
    finally:
        with psycopg.connect(database_url, autocommit=True) as connection:
            if schema_name:
                connection.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(schema_name)
                    )
                )
                connection.execute(
                    sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
                )
            else:
                connection.execute("DROP TABLE IF EXISTS public.alembic_version")
                connection.execute("DROP TABLE IF EXISTS public.checkpoint_writes")
                connection.execute("DROP TABLE IF EXISTS public.checkpoint_blobs")
                connection.execute("DROP TABLE IF EXISTS public.checkpoints")
                connection.execute("DROP TABLE IF EXISTS public.checkpoint_migrations")


@pytest.fixture
def langgraph_v2_migrated_database_url(
    langgraph_v2_test_database_url: str,
) -> Iterator[str]:
    """Apply all application migrations for one integration test."""
    config = build_alembic_config(
        langgraph_v2_test_database_url,
        fixture_schema=os.environ.get("LANGGRAPH_V2_TEST_SCHEMA") or None,
    )
    command.upgrade(config, "head")
    try:
        yield langgraph_v2_test_database_url
    finally:
        command.downgrade(config, "base")


@pytest.fixture
def mock_registry():
    """Mock ModelRegistry."""
    registry = MagicMock(spec=ModelRegistry)
    # Mock create_agent to return an AsyncMock that has a run method
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock()
    registry.create_agent.return_value = mock_agent
    return registry


@pytest.fixture
def mock_retriever():
    """Mock Retriever Provider."""
    retriever = AsyncMock(spec=BaseRetrieverProvider)
    return retriever


@pytest.fixture
def mock_ranker():
    """Mock Ranker Provider."""
    ranker = AsyncMock(spec=BaseRankerProvider)
    return ranker


@pytest.fixture
def mock_emitter():
    """Mock EventEmitter."""
    emitter = AsyncMock(spec=EventEmitter)
    emitter.is_cancelled = False
    return emitter


@pytest.fixture
def flow_context(mock_emitter: unittest.mock.AsyncMock):
    """Fixture for FlowContext."""
    return FlowContext(
        query="test query",
        emitter=mock_emitter,
        session_id="test-session",
    )
