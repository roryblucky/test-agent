"""Agent Tools — domain tools + agentskills.io skill-loading tools.

All tools are registered as standard Pydantic AI tool functions.
Pydantic AI auto-generates function-calling schema from type hints + docstrings.

Skill-system tools (agentskills.io progressive disclosure)
-----------------------------------------------------------
activate_skill_tool        Tier 2: LLM calls this to load a skill's full instructions.
load_skill_references_tool Tier 3: LLM calls this to load a skill's reference documents.

Domain tools (RAG operations)
------------------------------
search_documents_tool      Vector/semantic search in knowledge base.
rank_documents_tool        Re-rank retrieved documents by relevance.
decompose_question_tool    Break a complex question into focused sub-questions.
analyze_section_tool       Analyze a specific document section.
plan_and_reason_tool       Lightweight reasoning scratchpad.
get_user_classification_tool Request clarification from user.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import RunContext

from app.agents.agent_deps import AgentDeps
from app.api.schemas import QuestionAnswerSelector
from app.models.domain import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill system tools — agentskills.io Tier 2 & Tier 3
# The LLM calls these autonomously at runtime based on the discovery index
# (name + description) injected into the system prompt.
# -------------------------------------------------------------------------
async def activate_skill_tool(ctx: RunContext[AgentDeps], skill_name: str) -> str:
    """Activate a skill to get its full instructions and tool guidance.

    Call this FIRST when you determine that a specific skill is needed.
    Returns the complete instructions from the skill's SKILL.md wrapped in
    <skill_content> tags, along with a list of available resource files.

    IMPORTANT: Only call this with skill names from the <available_skills>
    list in the system prompt. Do NOT invent skill names.

    After activation, if you need additional context (schemas, examples,
    data dictionaries), call load_skill_references_tool(skill_name) to load
    the reference documents listed in <skill_resources>.

    Args:
        ctx: Tools context with access to skill registry.
        skill_name: The exact skill name from <available_skills> in system prompt.

    Returns:
        Skill instructions wrapped in <skill_content> tags, including skill
        directory path and <skill_resources> listing of available references.
    """
    registry = ctx.deps.skill_registry
    tenant_id = ctx.deps.tenant_id

    if registry is None:
        return f"Skill system not configured for tenant '{tenant_id}'."

    # Deduplication: return cached instructions if already activated this run
    if skill_name in ctx.deps.activated_skill_names:
        cached = registry.get_activated_skill(tenant_id, skill_name)
        if cached:
            # Return cached content without re-listing resources
            skill_dir = cached.source_path.rsplit("/", 1)[0]
            return (
                f'<skill_content name="{skill_name}">\n'
                f"{cached.instructions}\n\n"
                f"Skill directory: {skill_dir}\n"
                f"</skill_content>"
            )

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start(f"skill:activate:{skill_name}")

    # Tier 2: load full SKILL.md instructions
    activated = await registry.activate(tenant_id, [skill_name])
    if not activated:
        # Build hint from known names for the model
        known = [s.name for s in registry.get_summaries(tenant_id)]
        return (
            f"Skill '{skill_name}' not found. "
            f"Valid skill names are: {', '.join(known)}"
        )

    skill = activated[0]
    ctx.deps.activated_skill_names.append(skill_name)

    # List available resource files WITHOUT loading them (spec requirement)
    resource_files = await registry.get_resource_files(tenant_id, skill_name)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            f"skill:activate:{skill_name}",
            {"tools": skill.metadata.allowed_tools, "resources": resource_files},
        )

    # Build structured response per agentskills.io spec
    skill_dir = skill.source_path.rsplit("/", 1)[0]
    parts = [f'<skill_content name="{skill_name}">']
    parts.append(skill.instructions)
    parts.append(f"\nSkill directory: {skill_dir}")

    if resource_files:
        parts.append("\n<skill_resources>")
        for fname in resource_files:
            parts.append(f"  <file>references/{fname}</file>")
        parts.append("</skill_resources>")

    parts.append("</skill_content>")

    logger.info(
        f"[{tenant_id}] Agent activated skill: '{skill_name}' "
        f"(tools: {skill.metadata.allowed_tools}, "
        f"resources: {resource_files})"
    )
    return "\n".join(parts)


async def load_skill_references_tool(
    ctx: RunContext[AgentDeps], skill_name: str
) -> str:
    """Load detailed reference documents for a previously activated skill.

    Call this when you need additional technical context beyond the skill's
    instructions — for example: database schemas, API specifications, data
    dictionaries, or domain-specific lookup tables.

    The available reference files are listed in <skill_resources> inside
    the <skill_content> returned by activate_skill_tool(). Only call this
    AFTER activating the skill.

    Args:
        ctx: Tools context with access to skill registry.
        skill_name: The name of a previously activated skill.

    Returns:
        All reference documents concatenated as formatted text.
    """
    registry = ctx.deps.skill_registry
    tenant_id = ctx.deps.tenant_id

    if registry is None:
        return "Skill system not configured."

    skill = registry.get_activated_skill(tenant_id, skill_name)
    if skill is None:
        return (
            f"Skill '{skill_name}' has not been activated yet. "
            f"Call activate_skill_tool('{skill_name}') first."
        )

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start(f"skill:references:{skill_name}")

    # Tier 3: load reference document contents
    refs = await registry.load_references(skill)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            f"skill:references:{skill_name}",
            {"reference_count": len(refs)},
        )

    if not refs:
        return f"No reference documents available for skill '{skill_name}'."

    parts = [f"# Reference Documents for '{skill_name}'\n"]
    for ref in refs:
        parts.append(f"## {ref.filename}\n\n{ref.content}\n")

    logger.info(
        f"[{tenant_id}] Agent loaded {len(refs)} reference(s) "
        f"for skill '{skill_name}'"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Domain tools (RAG operations)
# ---------------------------------------------------------------------------


async def search_documents_tool(
    ctx: RunContext[AgentDeps],
    query: str,
    filter_expr: str | None = None,
) -> str:
    """Search the knowledge base for documents relevant to a query.

    Args:
        query: The search query text.
        filter_expr: Optional metadata filter expression string.
    """
    if not ctx.deps.providers.retriever:
        return "No retriever configured for this tenant."

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start("search_documents")

    docs = await ctx.deps.providers.retriever.retrieve(query, filter_expr=filter_expr)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            "search_documents",
            {
                "query": query,
                "filter_expr": filter_expr,
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
