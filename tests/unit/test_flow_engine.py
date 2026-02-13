"""Unit tests for FlowEngine."""

from unittest.mock import MagicMock, AsyncMock
import pytest
from app.config.models import (
    FlowStep,
    FlowStepType,
    TenantConfig,
    FlowConfig,
    LLMConfig,
)
from app.services.flow_engine import FlowEngine
from app.services.flow_context import FlowContext


@pytest.fixture
def mock_providers():
    """Mock TenantProviders."""
    providers = MagicMock()
    providers.moderation = AsyncMock()
    providers.retriever = AsyncMock()
    providers.ranker = AsyncMock()
    providers.groundedness = AsyncMock()
    return providers


@pytest.fixture
def flow_engine(mock_registry, mock_providers):
    """Create FlowEngine instance with mocks."""
    config = TenantConfig(
        id="test-tenant",
        kmsAppName="Test App",
        applicationId="app-123",
        adGroups=["group1"],
        flow_config=FlowConfig(steps=[]),
        llm_config=LLMConfig(models={}),
    )
    return FlowEngine(config, mock_registry, mock_providers)


@pytest.mark.asyncio
async def test_run_moderation_pre(flow_engine, mock_providers, mock_emitter):
    """Test moderation step (pre-check)."""
    step = FlowStep(type=FlowStepType.MODERATION, mode="pre")
    ctx = FlowContext(query="bad query", emitter=mock_emitter)

    mock_providers.moderation.check.return_value.is_flagged = False

    await flow_engine._run_moderation(ctx, step)

    mock_providers.moderation.check.assert_awaited_with("bad query")
    mock_emitter.emit_step_completed.assert_awaited()


@pytest.mark.asyncio
async def test_run_retriever(flow_engine, mock_providers, mock_emitter):
    """Test retriever step."""
    step = FlowStep(type=FlowStepType.RETRIEVER)
    ctx = FlowContext(query="test", emitter=mock_emitter)

    mock_providers.retriever.config.top_k = 3
    mock_providers.retriever.retrieve.return_value = []

    await flow_engine._run_retriever(ctx, step)

    mock_providers.retriever.retrieve.assert_awaited_with("test", 3)
    mock_emitter.emit_step_completed.assert_awaited()
