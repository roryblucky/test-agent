"""Unit tests for FlowEngine."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.models import (
    FlowConfig,
    FlowStep,
    FlowStepType,
    LLMConfig,
    TenantConfig,
)
from app.models.domain import Document, ModerationResult
from app.services.flow_engine import FlowEngine
from app.services.handlers.base import StepHandler
from app.services.handlers.moderation import ModerationHandler
from app.services.handlers.ranking import RankingHandler
from app.services.handlers.retriever import RetrieverHandler


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
    assert ctx.metadata["streaming_policy"] == "token"

    # Verify emitter events emitted
    assert mock_emitter.emit_step_start.call_count == 2


class FakeRetrieverProvider:
    """Minimal retriever provider for legacy RAG regression coverage."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(top_k=3)
        self.calls: list[tuple[str, int]] = []

    async def retrieve(
        self, query: str, top_k: int = 10, filter_expr: str | None = None
    ) -> list[Document]:
        self.calls.append((query, top_k))
        return [
            Document(id="doc1", content=f"Retrieved for {query}", score=0.4),
            Document(id="doc2", content="Lower ranked content", score=0.2),
        ]


class FakeRankerProvider:
    """Minimal ranker provider for legacy RAG regression coverage."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def rank(
        self, query: str, documents: list[Document], top_n: int = 5
    ) -> list[Document]:
        self.calls.append((query, [doc.id for doc in documents]))
        return [
            Document(id=doc.id, content=doc.content, metadata=doc.metadata, score=1.0)
            for doc in documents[:top_n]
        ]


class FakeModerationProvider:
    """Minimal moderation provider for legacy RAG regression coverage."""

    def __init__(self) -> None:
        self.checked_texts: list[str] = []

    async def check(self, text: str) -> ModerationResult:
        self.checked_texts.append(text)
        return ModerationResult(is_flagged=False)


class FakeLLMHandler:
    """Small LLM handler double that preserves old mode semantics."""

    async def handle(self, ctx, step):
        if step.mode == "refine_question":
            ctx.refined_query = "refined legacy query"
        elif step.mode == "answer":
            ctx.llm_response = "Legacy RAG answer"
        else:
            raise ValueError(f"Unexpected LLM mode in regression test: {step.mode}")

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(f"llm:{step.mode}", {})
        return ctx


@pytest.mark.asyncio
async def test_legacy_rag_flow_regression(mock_emitter):
    """Cover the existing moderation -> llm -> retriever -> ranking -> answer chain."""
    config = TenantConfig(
        kmsAppName="Legacy RAG App",
        applicationId="legacy-rag",
        adGroups=["group1"],
        flow_config=FlowConfig(
            steps=[
                FlowStep(type=FlowStepType.MODERATION, mode="pre"),
                FlowStep(type=FlowStepType.LLM, mode="refine_question"),
                FlowStep(type=FlowStepType.RETRIEVER),
                FlowStep(type=FlowStepType.RANKING),
                FlowStep(type=FlowStepType.LLM, mode="answer"),
            ]
        ),
        llm_config=LLMConfig(models={}),
    )
    retriever = FakeRetrieverProvider()
    ranker = FakeRankerProvider()
    moderation = FakeModerationProvider()
    handlers = {
        FlowStepType.MODERATION: ModerationHandler(moderation),
        FlowStepType.LLM: FakeLLMHandler(),
        FlowStepType.RETRIEVER: RetrieverHandler(retriever),
        FlowStepType.RANKING: RankingHandler(ranker),
    }

    ctx = await FlowEngine(config, handlers).execute(
        "legacy query",
        emitter=mock_emitter,
    )

    assert ctx.metadata["steps_executed"] == [
        "moderation:pre",
        "llm:refine_question",
        "retriever",
        "ranking",
        "llm:answer",
    ]
    assert ctx.refined_query == "refined legacy query"
    assert ctx.llm_response == "Legacy RAG answer"
    assert [doc.id for doc in ctx.ranked_documents] == ["doc1", "doc2"]
    assert retriever.calls == [("refined legacy query", 3)]
    assert ranker.calls == [("refined legacy query", ["doc1", "doc2"])]
    assert moderation.checked_texts == ["legacy query"]


@pytest.mark.asyncio
async def test_flow_engine_marks_compliance_review_streaming_policy(mock_handlers):
    """A flow with compliance review uses approved-answer-only streaming."""
    config = TenantConfig(
        kmsAppName="Compliance App",
        applicationId="compliance-app",
        adGroups=["group1"],
        flow_config=FlowConfig(
            steps=[
                FlowStep(type=FlowStepType.LLM, mode="answer"),
                FlowStep(type=FlowStepType.LLM, mode="compliance_review"),
            ]
        ),
        llm_config=LLMConfig(models={}),
    )

    ctx = await FlowEngine(config, mock_handlers).execute("test query")

    assert ctx.metadata["streaming_policy"] == "approved_answer_only"
