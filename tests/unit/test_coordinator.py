"""Unit tests for Agent tool functions."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import RunContext

from app.agents.agent_deps import AgentDeps
from app.agents.tools import (
    decompose_question_tool,
    search_documents_tool,
)
from app.models.domain import Document
from app.services.flow_context import FlowContext
from app.services.tenant_manager import TenantProviders


@pytest.fixture
def deps(
    mock_registry: MagicMock,
    mock_retriever: AsyncMock,
    mock_ranker: AsyncMock,
    mock_emitter: AsyncMock,
    flow_context: FlowContext,
) -> AgentDeps:
    return AgentDeps(
        registry=mock_registry,
        providers=TenantProviders(
            retriever=mock_retriever,
            ranker=mock_ranker,
        ),
        emitter=mock_emitter,
        tenant_id="test-tenant",
        flow_context=flow_context,
    )


@pytest.fixture
def ctx(deps: AgentDeps) -> RunContext[AgentDeps]:
    """Mock RunContext."""
    context = MagicMock(spec=RunContext)
    context.deps = deps
    context.usage = MagicMock()
    return cast(RunContext[AgentDeps], context)


@pytest.mark.asyncio
async def test_search_documents_tool(
    ctx: RunContext[AgentDeps],
    mock_retriever: AsyncMock,
    mock_emitter: AsyncMock,
    flow_context: FlowContext,
) -> None:
    """Test search_documents tool."""
    mock_retriever.retrieve.return_value = [
        Document(id="doc1", content="Content 1", score=0.9),
        Document(id="doc2", content="Content 2", score=0.8),
    ]

    result = await search_documents_tool(ctx, "test query", task_id="search-task")

    assert result.status == "success"
    assert result.result_count == 2
    mock_retriever.retrieve.assert_awaited_once_with("test query", filter_expr=None)
    mock_emitter.emit_step_start.assert_awaited_with("search_documents")
    mock_emitter.emit_step_completed.assert_awaited()
    assert len(flow_context.tool_results) == 1
    assert len(flow_context.tool_results[0].normalized_items) == 2
    assert flow_context.tool_calls[0].tool_name == "search_documents"
    assert flow_context.tool_calls[0].task_id == "search-task"
    assert flow_context.tool_calls[0].tenant_id == "test-tenant"
    assert flow_context.tool_results[0].task_id == "search-task"
    assert flow_context.tool_observations[0].result_count == 2
    assert flow_context.tool_observations[0].task_status_hint == "completed"


@pytest.mark.asyncio
async def test_search_documents_no_results(
    ctx: RunContext[AgentDeps], mock_retriever: AsyncMock
) -> None:
    """Test search_documents with no results."""
    mock_retriever.retrieve.return_value = []

    result = await search_documents_tool(ctx, "test query")

    assert result.status == "empty"
    assert result.task_status_hint == "missing"


@pytest.mark.asyncio
async def test_decompose_question_tool(
    ctx: RunContext[AgentDeps], mock_registry: MagicMock
) -> None:
    """Test decompose_question tool."""
    # Mock the inner agent run result
    mock_agent = mock_registry.create_agent.return_value
    mock_agent.run.return_value.output = ["Q1", "Q2"]

    result = await decompose_question_tool(ctx, "Complex question")

    assert result.status == "success"
    assert result.result_count == 2
    mock_registry.create_agent.assert_called_once()
    mock_agent.run.assert_awaited_once()
