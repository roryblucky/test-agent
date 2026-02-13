"""Unit tests for Coordinator Agent tools."""

from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from app.agents.coordinator import CoordinatorDeps
from app.agents.tools import (
    search_documents_tool,
    decompose_question_tool,
)
from app.models.domain import Document


@pytest.fixture
def deps(mock_registry, mock_retriever, mock_ranker, mock_emitter):
    return CoordinatorDeps(
        registry=mock_registry,
        retriever=mock_retriever,
        ranker=mock_ranker,
        emitter=mock_emitter,
    )


@pytest.fixture
def ctx(deps):
    """Mock RunContext."""
    context = MagicMock(spec=RunContext)
    context.deps = deps
    context.usage = MagicMock()
    return context


@pytest.mark.asyncio
async def test_search_documents_tool(ctx, mock_retriever):
    """Test search_documents tool."""
    mock_retriever.retrieve.return_value = [
        Document(id="doc1", content="Content 1", score=0.9),
        Document(id="doc2", content="Content 2", score=0.8),
    ]

    result = await search_documents_tool(ctx, "test query")

    assert "Content 1" in result
    assert "Content 2" in result
    mock_retriever.retrieve.assert_awaited_once_with("test query")
    ctx.deps.emitter.emit_step_start.assert_awaited_with("search_documents")
    ctx.deps.emitter.emit_step_completed.assert_awaited()


@pytest.mark.asyncio
async def test_search_documents_no_results(ctx, mock_retriever):
    """Test search_documents with no results."""
    mock_retriever.retrieve.return_value = []

    result = await search_documents_tool(ctx, "test query")

    assert "No documents found" in result


@pytest.mark.asyncio
async def test_decompose_question_tool(ctx, mock_registry):
    """Test decompose_question tool."""
    # Mock the inner agent run result
    mock_agent = mock_registry.create_agent.return_value
    mock_agent.run.return_value.output = ["Q1", "Q2"]

    result = await decompose_question_tool(ctx, "Complex question")

    assert result == ["Q1", "Q2"]
    mock_registry.create_agent.assert_called_once()
    mock_agent.run.assert_awaited_once()
