"""Pytest configuration and fixtures."""

import asyncio
import os
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
def langgraph_v2_test_database_url() -> str:
    """Require an explicitly named, disposable PostgreSQL test database."""
    database_url = require_disposable_postgres_url(os.environ)
    try:
        with psycopg.connect(database_url, connect_timeout=3):
            pass
    except psycopg.OperationalError as error:
        pytest.fail(
            "LANGGRAPH_V2_TEST_DATABASE_URL points to an unavailable disposable "
            f"PostgreSQL database: {error}"
        )
    return database_url


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
