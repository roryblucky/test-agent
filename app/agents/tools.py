"""Coordinator Agent Tools — Standalone functions for unit testing.

Extracted from ``coordinator.py`` to allow direct testing of tool logic
without mocking the entire PydanticAI Agent machinery.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.api.schemas import QuestionAnswerSelector
from app.models.domain import Document

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from app.agents.agent_deps import AgentDeps


async def search_documents_tool(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search the knowledge base for documents relevant to a query.

    Args:
        ctx: Tools context with access to dependencies.
        query: The search query, can be a sub-question or refined query.

    Returns:
        Formatted text of retrieved documents.
    """
    if not ctx.deps.providers.retriever:
        return "No retriever configured for this tenant."

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start("search_documents")

    docs = await ctx.deps.providers.retriever.retrieve(query)

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
    ctx: RunContext[AgentDeps], query: str, document_texts: list[str]
) -> str:
    """Re-rank documents by relevance to a specific query.

    Args:
        ctx: Tools context with access to dependencies.
        query: The ranking query.
        document_texts: List of document texts to rank.

    Returns:
        The top-ranked documents as formatted text.
    """
    if not ctx.deps.providers.ranker:
        return "No ranker configured for this tenant."

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start("rank_documents")

    # Wrap raw text into Document objects for the ranker
    docs = [
        Document(id=f"doc_{i}", content=text) for i, text in enumerate(document_texts)
    ]
    ranked = await ctx.deps.providers.ranker.rank(query, docs)

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
    ctx: RunContext[AgentDeps], complex_question: str
) -> list[str]:
    """Break a complex question into 2-5 focused sub-questions.

    Args:
        ctx: Tools context with access to dependencies.
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
    ctx: RunContext[AgentDeps],
    question: str,
    context: str,
) -> str:
    """Analyze specific content to answer a focused question.

    Args:
        ctx: Tools context with access to dependencies.
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


async def plan_and_reason_tool(ctx: RunContext[Any], reasoning: str) -> str:
    """Use this tool to structure your response to the user with reasoning.
    
    Call this tool to think out loud and record your internal planning.
    
    Args:
        ctx: Tools context.
        reasoning: Your detailed reasoning or plan.
    """
    logger.info(f"Plan and Reasoning: {reasoning}")
    return "Reasoning updated and recorded."


async def get_user_classification_tool(
    ctx: RunContext[Any],
    response: str,
    quick_questions: list[QuestionAnswerSelector] | None = None,
) -> str:
    """Get clarification from user if the user's query is too broad.
    
    Use this tool when you need the user to clarify their intent by selecting from specific options.
    You MUST output the exact string returned by this tool as your final answer.
    
    Args:
        ctx: Tools context.
        response: The explanatory response or question directed at the user.
        quick_questions: A list of specific questions and their options for the user to choose from.
    """
    logger.info(f"Clarification Question: {response}")
    
    def _build_quick_questions_markdown(questions: list[QuestionAnswerSelector]) -> str:
        """Build quick question payload in markdown fenced block for UI parsing."""
        if not questions:
            return ""

        lines: list[str] = ["```selection"]
        for index, item in enumerate(questions, start=1):
            question = item.question.strip()
            options = [option.strip() for option in item.options if option and option.strip()]

            lines.append(f"{index}. {question}")
            for option in options:
                lines.append(f"- {option}")

        lines.append("```")
        return "\n".join(lines)

    quick_questions_markdown = _build_quick_questions_markdown(quick_questions or [])
    answer = response if not quick_questions_markdown else f"{response}\n\n{quick_questions_markdown}"
    
    return answer
