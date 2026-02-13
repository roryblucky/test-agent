"""Pytest configuration and fixtures."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.model_registry import ModelRegistry
from app.providers.base import BaseRankerProvider, BaseRetrieverProvider
from app.services.events import EventEmitter
from app.services.flow_context import FlowContext


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


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
    return emitter


@pytest.fixture
def flow_context(mock_emitter):
    """Fixture for FlowContext."""
    return FlowContext(
        query="test query",
        emitter=mock_emitter,
        session_id="test-session",
    )
