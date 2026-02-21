"""Unit tests for FlowEngine."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.models import (
    FlowConfig,
    FlowStep,
    FlowStepType,
    LLMConfig,
    TenantConfig,
)
from app.services.flow_engine import FlowEngine
from app.services.handlers.base import StepHandler


@pytest.fixture
def mock_handlers():
    """Mock StepHandlers."""
    handlers = {}
    for step_type in FlowStepType:
        mock_handler = MagicMock(spec=StepHandler)
        mock_handler.handle = AsyncMock(side_effect=lambda ctx, step: ctx)
        handlers[step_type] = mock_handler
    return handlers


@pytest.fixture
def flow_engine(mock_handlers):
    """Create FlowEngine instance with mocks."""
    config = TenantConfig(
        id="test-tenant",
        kmsAppName="Test App",
        applicationId="app-123",
        adGroups=["group1"],
        flow_config=FlowConfig(
            steps=[
                FlowStep(type=FlowStepType.MODERATION, mode="pre"),
                FlowStep(type=FlowStepType.RETRIEVER),
            ]
        ),
        llm_config=LLMConfig(models={}),
    )
    return FlowEngine(config, mock_handlers)


@pytest.mark.asyncio
async def test_execute_pipeline(flow_engine, mock_handlers, mock_emitter):
    """Test full pipeline execution."""
    ctx = await flow_engine.execute("test query", emitter=mock_emitter)
    assert ctx is not None

    # Verify handlers were called
    mock_handlers[FlowStepType.MODERATION].handle.assert_awaited_once()
    mock_handlers[FlowStepType.RETRIEVER].handle.assert_awaited_once()

    # Verify emitter events emitted
    assert mock_emitter.emit_step_start.call_count == 2
