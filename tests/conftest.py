"""Pytest configuration and fixtures."""

import asyncio
import os
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest

from app.core.model_registry import ModelRegistry
from app.providers.base import BaseRankerProvider, BaseRetrieverProvider
from app.services.events import EventEmitter
from app.services.flow_context import FlowContext
from tests.postgres import require_disposable_postgres_url


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def langgraph_v2_test_database_url() -> Iterator[str]:
    """Provide an empty test-only database and remove migration artifacts."""
    database_url = require_disposable_postgres_url(os.environ)
    try:
        with psycopg.connect(database_url, connect_timeout=3) as connection:
            existing_object = connection.execute(
                """
                SELECT object_kind, object_name
                FROM (
                    SELECT 'schema' AS object_kind, nspname AS object_name
                    FROM pg_namespace
                    WHERE nspname NOT IN ('public', 'information_schema')
                      AND nspname NOT LIKE 'pg_%'
                    UNION ALL
                    SELECT 'relation', c.relname
                    FROM pg_class AS c
                    JOIN pg_namespace AS n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public'
                    UNION ALL
                    SELECT 'function', p.proname
                    FROM pg_proc AS p
                    JOIN pg_namespace AS n ON n.oid = p.pronamespace
                    WHERE n.nspname = 'public'
                    UNION ALL
                    SELECT 'type', t.typname
                    FROM pg_type AS t
                    JOIN pg_namespace AS n ON n.oid = t.typnamespace
                    WHERE n.nspname = 'public'
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
        yield database_url
    finally:
        with psycopg.connect(database_url, autocommit=True) as connection:
            connection.execute("DROP SCHEMA IF EXISTS langgraph_v2 CASCADE")
            connection.execute("DROP TABLE IF EXISTS public.alembic_version")


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
def flow_context(mock_emitter):
    """Fixture for FlowContext."""
    return FlowContext(
        query="test query",
        emitter=mock_emitter,
        session_id="test-session",
    )
