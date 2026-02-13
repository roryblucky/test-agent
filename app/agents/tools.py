"""Coordinator Agent Tools — Standalone functions for unit testing.

Extracted from ``coordinator.py`` to allow direct testing of tool logic
without mocking the entire PydanticAI Agent machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import logging
from app.models.domain import Document

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.agents.coordinator import CoordinatorDeps
    from pydantic_ai import RunContext


async def search_documents_tool(ctx: RunContext[CoordinatorDeps], query: str) -> str:
    """Search the knowledge base for documents relevant to a query.

    Args:
        query: The search query, can be a sub-question or refined query.

    Returns:
        Formatted text of retrieved documents.
    """
    if not ctx.deps.retriever:
        return "No retriever configured for this tenant."

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start("search_documents")

    docs = await ctx.deps.retriever.retrieve(query)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            "search_documents",
            {
                "query": query,
                "document_count": len(docs),
                "documents": [{"id": d.id, "score": d.score} for d in docs],
            },
        )

    if not docs:
        return "No documents found for this query."

    return "\n\n---\n\n".join(
        f"[Document {d.id} | score={d.score}]\n{d.content}" for d in docs
    )


async def rank_documents_tool(
    ctx: RunContext[CoordinatorDeps], query: str, document_texts: list[str]
) -> str:
    """Re-rank documents by relevance to a specific query.

    Args:
        query: The ranking query.
        document_texts: List of document texts to rank.

    Returns:
        The top-ranked documents as formatted text.
    """
    if not ctx.deps.ranker:
        return "No ranker configured for this tenant."

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start("rank_documents")

    # Wrap raw text into Document objects for the ranker
    docs = [
        Document(id=f"doc_{i}", content=text) for i, text in enumerate(document_texts)
    ]
    ranked = await ctx.deps.ranker.rank(query, docs)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            "rank_documents",
            {
                "query": query,
                "input_count": len(docs),
                "output_count": len(ranked),
            },
        )

    return "\n\n---\n\n".join(
        f"[Ranked #{i + 1} | score={d.score}]\n{d.content}"
        for i, d in enumerate(ranked)
    )


async def decompose_question_tool(
    ctx: RunContext[CoordinatorDeps], complex_question: str
) -> list[str]:
    """Break a complex question into 2-5 focused sub-questions.

    Args:
        complex_question: The complex question to decompose.

    Returns:
        A list of focused sub-questions.
    """
    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start("decompose_question")

    # Note: This creates a nested agent. In a real enterprise app, we might
    # want to inject this agent or use a lighter-weight mechanism.
    # For now, we keep the existing logic but make it testable via mocks.
    decompose_agent = ctx.deps.registry.create_agent(
        "fast",
        output_type=list[str],
        instructions=(
            "Break the given question into 2-5 specific, focused sub-questions "
            "that can each be answered independently. Each sub-question should "
            "target a distinct aspect of the original question."
        ),
    )
    # Use the parent context's usage limits if available
    result = await decompose_agent.run(complex_question, usage=ctx.usage)
    sub_questions = result.output

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            "decompose_question",
            {"sub_questions": sub_questions},
        )

    return sub_questions


async def analyze_section_tool(
    ctx: RunContext[CoordinatorDeps],
    question: str,
    context: str,
) -> str:
    """Analyze specific content to answer a focused question.

    Args:
        question: The specific question to analyze.
        context: The reference text to analyze.

    Returns:
        A focused analysis.
    """
    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start("analyze_section")

    analysis_agent = ctx.deps.registry.create_agent(
        "fast",
        output_type=str,
        instructions=(
            "You are a specialist analyst. Answer the given question "
            "based strictly on the provided context. Be precise and "
            "cite specific data points when available."
        ),
    )
    prompt = f"Question: {question}\n\nContext:\n{context}"
    result = await analysis_agent.run(prompt, usage=ctx.usage)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            "analyze_section",
            {"question": question, "analysis_length": len(result.output)},
        )

    return result.output
