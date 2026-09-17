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
import time
from datetime import UTC, datetime
from typing import Any

from pydantic_ai import RunContext

from app.agents.agent_deps import AgentDeps
from app.api.schemas import QuestionAnswerSelector
from app.models.domain import Document
from app.models.workflow import (
    NormalizedToolResultItem,
    TaskStatusHint,
    ToolCallRecord,
    ToolObservation,
    ToolObservationStatus,
    ToolResultRecord,
)
from app.services.flow_context import FlowContext

logger = logging.getLogger(__name__)


def _get_flow_context(ctx: RunContext[Any]) -> FlowContext | None:
    """Return workflow context when the tool is running inside FlowEngine."""
    deps = getattr(ctx, "deps", None)
    flow_context = getattr(deps, "flow_context", None)
    return flow_context if isinstance(flow_context, FlowContext) else None


def _latency_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _record_tool_call(
    ctx: RunContext[Any],
    *,
    tool_name: str,
    task_id: str | None,
    input_payload: dict[str, Any],
    status: ToolObservationStatus,
    result_count: int = 0,
    compiled_filter: str | None = None,
    latency_ms: int | None = None,
    error_code: str | None = None,
    error: str | None = None,
) -> str:
    flow_ctx = _get_flow_context(ctx)
    if flow_ctx is None:
        return f"{tool_name}:external"

    tool_call_id = f"{tool_name}:{len(flow_ctx.tool_calls) + 1}"

    deps = getattr(ctx, "deps", None)
    flow_ctx.tool_calls.append(
        ToolCallRecord(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            task_id=task_id,
            input_payload=input_payload,
            compiled_filter=compiled_filter,
            status=status,
            result_count=result_count,
            latency_ms=latency_ms,
            error_code=error_code,
            error=error,
            tenant_id=getattr(deps, "tenant_id", None),
            user_id=None,
        )
    )
    return tool_call_id


def _record_tool_observation(
    ctx: RunContext[Any],
    observation: ToolObservation,
) -> None:
    flow_ctx = _get_flow_context(ctx)
    if flow_ctx is not None:
        flow_ctx.tool_observations.append(observation)


def _store_document_results(
    ctx: RunContext[Any],
    *,
    tool_call_id: str,
    tool_name: str,
    task_id: str | None,
    documents: list[Document],
) -> None:
    flow_ctx = _get_flow_context(ctx)
    if flow_ctx is None:
        return

    items: list[NormalizedToolResultItem] = []
    for doc in documents:
        metadata = dict(doc.metadata)
        metadata["document_id"] = doc.id
        items.append(
            NormalizedToolResultItem(
                item_id=doc.id,
                title=str(doc.metadata["title"]) if "title" in doc.metadata else None,
                content=doc.content,
                score=doc.score,
                metadata=metadata,
            )
        )

    flow_ctx.tool_results.append(
        ToolResultRecord(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            task_id=task_id,
            source=tool_name,
            normalized_items=items,
        )
    )


def _store_text_result(
    ctx: RunContext[Any],
    *,
    tool_call_id: str,
    tool_name: str,
    task_id: str | None,
    item_id: str,
    content: str,
) -> None:
    flow_ctx = _get_flow_context(ctx)
    if flow_ctx is None:
        return

    flow_ctx.tool_results.append(
        ToolResultRecord(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            task_id=task_id,
            source=tool_name,
            normalized_items=[
                NormalizedToolResultItem(
                    item_id=item_id,
                    content=content,
                    metadata={"recorded_at": datetime.now(UTC).isoformat()},
                )
            ],
        )
    )


def _observation(
    *,
    tool_name: str,
    status: ToolObservationStatus,
    result_count: int = 0,
    warnings: list[str] | None = None,
    error_code: str | None = None,
) -> ToolObservation:
    task_status_hints: dict[ToolObservationStatus, TaskStatusHint] = {
        "success": "completed",
        "empty": "missing",
        "partial": "partial",
        "stale": "stale",
        "error": "failed",
    }
    return ToolObservation(
        tool_name=tool_name,
        status=status,
        task_status_hint=task_status_hints[status],
        result_count=result_count,
        warnings=warnings or [],
        error_code=error_code,
    )


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
    skill = await registry.activate(tenant_id, skill_name)
    if skill is None:
        # Build hint from known names for the model
        known = [s.name for s in registry.get_summaries(tenant_id)]
        return (
            f"Skill '{skill_name}' not found. Valid skill names are: {', '.join(known)}"
        )

    ctx.deps.activated_skill_names.append(skill_name)

    # List available resource files WITHOUT loading them (spec requirement)
    resource_files = await registry.get_resource_files(tenant_id, skill_name)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            f"skill:activate:{skill_name}",
            {
                "tools": skill.metadata.allowed_tools,
                "required_tools": skill.metadata.required_tools,
                "risk_level": skill.metadata.risk_level.value,
                "resources": resource_files,
            },
        )

    # Build structured response per agentskills.io spec
    skill_dir = skill.source_path.rsplit("/", 1)[0]
    parts = [f'<skill_content name="{skill_name}">']
    parts.append(skill.instructions)

    if (
        skill.metadata.allowed_tools
        or skill.metadata.required_tools
        or skill.metadata.tool_constraints
    ):
        parts.append("\n<skill_tool_policy>")
        parts.append(f"risk_level: {skill.metadata.risk_level.value}")
        parts.append(
            f"allowed_tools: {', '.join(skill.metadata.allowed_tools) or 'none'}"
        )
        parts.append(
            f"required_tools: {', '.join(skill.metadata.required_tools) or 'none'}"
        )
        if skill.metadata.tool_constraints:
            parts.append("tool_constraints:")
            for tool_name, constraints in skill.metadata.tool_constraints.items():
                parts.append(f"  {tool_name}: {constraints}")
        parts.append("</skill_tool_policy>")

    parts.append(f"\nSkill directory: {skill_dir}")

    if resource_files:
        parts.append("\n<skill_resources>")
        parts.extend(f"  <file>references/{fname}</file>" for fname in resource_files)
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
    parts.extend(f"## {ref.filename}\n\n{ref.content}\n" for ref in refs)

    logger.info(
        f"[{tenant_id}] Agent loaded {len(refs)} reference(s) for skill '{skill_name}'"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Domain tools (RAG operations)
# ---------------------------------------------------------------------------


async def search_documents_tool(
    ctx: RunContext[AgentDeps],
    query: str,
    filter_expr: str | None = None,
    task_id: str | None = None,
) -> ToolObservation:
    """Search the knowledge base for documents relevant to a query.

    Args:
        ctx: Tools context with access to dependencies.
        query: The search query text.
        filter_expr: Optional metadata filter expression string.
        task_id: Checklist task identifier assigned by the planner.

    Returns:
        A lightweight observation. Full normalized results are stored on
        ``FlowContext.tool_results`` for aggregation.
    """
    start = time.perf_counter()
    task_id = task_id or "search_documents"
    input_payload = {"query": query, "filter_expr": filter_expr, "task_id": task_id}

    if not ctx.deps.providers.retriever:
        message = "No retriever configured for this tenant."
        _record_tool_call(
            ctx,
            tool_name="search_documents",
            task_id=task_id,
            input_payload=input_payload,
            compiled_filter=filter_expr,
            status="error",
            latency_ms=_latency_ms(start),
            error=message,
        )
        observation = _observation(
            tool_name="search_documents",
            status="error",
            warnings=[message],
            error_code="retriever_not_configured",
        )
        _record_tool_observation(ctx, observation)
        return observation

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start("search_documents")

    try:
        docs = await ctx.deps.providers.retriever.retrieve(
            query, filter_expr=filter_expr
        )
    except Exception as exc:
        _record_tool_call(
            ctx,
            tool_name="search_documents",
            task_id=task_id,
            input_payload=input_payload,
            compiled_filter=filter_expr,
            status="error",
            latency_ms=_latency_ms(start),
            error_code="retriever_error",
            error=str(exc),
        )
        observation = _observation(
            tool_name="search_documents",
            status="error",
            warnings=[str(exc)],
            error_code="retriever_error",
        )
        _record_tool_observation(ctx, observation)
        raise

    status = "success" if docs else "empty"
    tool_call_id = _record_tool_call(
        ctx,
        tool_name="search_documents",
        task_id=task_id,
        input_payload=input_payload,
        compiled_filter=filter_expr,
        status=status,
        result_count=len(docs),
        latency_ms=_latency_ms(start),
    )
    _store_document_results(
        ctx,
        tool_call_id=tool_call_id,
        tool_name="search_documents",
        task_id=task_id,
        documents=docs,
    )
    observation = _observation(
        tool_name="search_documents",
        status=status,
        result_count=len(docs),
    )
    _record_tool_observation(ctx, observation)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            "search_documents",
            {
                "query": query,
                "filter_expr": filter_expr,
                "document_count": len(docs),
                "tool_call_id": tool_call_id,
                "task_id": task_id,
            },
        )

    return observation


async def rank_documents_tool(
    ctx: RunContext[AgentDeps],
    query: str,
    document_texts: list[str],
    task_id: str | None = None,
) -> ToolObservation:
    """Re-rank documents by relevance to a specific query.

    Args:
        ctx: Tools context with access to dependencies.
        query: The ranking query.
        document_texts: List of document texts to rank.
        task_id: Checklist task identifier assigned by the planner.

    Returns:
        A lightweight observation. Ranked documents are stored on
        ``FlowContext.tool_results`` for aggregation.
    """
    start = time.perf_counter()
    task_id = task_id or "rank_documents"
    input_payload = {
        "query": query,
        "document_count": len(document_texts),
        "task_id": task_id,
    }

    if not ctx.deps.providers.ranker:
        message = "No ranker configured for this tenant."
        _record_tool_call(
            ctx,
            tool_name="rank_documents",
            task_id=task_id,
            input_payload=input_payload,
            status="error",
            latency_ms=_latency_ms(start),
            error_code="ranker_not_configured",
            error=message,
        )
        observation = _observation(
            tool_name="rank_documents",
            status="error",
            warnings=[message],
            error_code="ranker_not_configured",
        )
        _record_tool_observation(ctx, observation)
        return observation

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_start("rank_documents")

    # Wrap raw text into Document objects for the ranker
    docs = [
        Document(id=f"doc_{i}", content=text) for i, text in enumerate(document_texts)
    ]
    try:
        ranked = await ctx.deps.providers.ranker.rank(query, docs)
    except Exception as exc:
        _record_tool_call(
            ctx,
            tool_name="rank_documents",
            task_id=task_id,
            input_payload=input_payload,
            status="error",
            latency_ms=_latency_ms(start),
            error_code="ranker_error",
            error=str(exc),
        )
        observation = _observation(
            tool_name="rank_documents",
            status="error",
            warnings=[str(exc)],
            error_code="ranker_error",
        )
        _record_tool_observation(ctx, observation)
        raise

    status = "success" if ranked else "empty"
    tool_call_id = _record_tool_call(
        ctx,
        tool_name="rank_documents",
        task_id=task_id,
        input_payload=input_payload,
        status=status,
        result_count=len(ranked),
        latency_ms=_latency_ms(start),
    )
    _store_document_results(
        ctx,
        tool_call_id=tool_call_id,
        tool_name="rank_documents",
        task_id=task_id,
        documents=ranked,
    )
    observation = _observation(
        tool_name="rank_documents",
        status=status,
        result_count=len(ranked),
    )
    _record_tool_observation(ctx, observation)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            "rank_documents",
            {
                "query": query,
                "input_count": len(docs),
                "output_count": len(ranked),
                "tool_call_id": tool_call_id,
                "task_id": task_id,
            },
        )

    return observation


async def decompose_question_tool(
    ctx: RunContext[AgentDeps],
    complex_question: str,
    task_id: str | None = None,
) -> ToolObservation:
    """Break a complex question into 2-5 focused sub-questions.

    Args:
        ctx: Tools context with access to dependencies.
        complex_question: The complex question to decompose.
        task_id: Checklist task identifier assigned by the planner.

    Returns:
        A lightweight observation. Sub-questions are stored on
        ``FlowContext.tool_results``.
    """
    start = time.perf_counter()
    task_id = task_id or "decompose_question"
    input_payload = {"complex_question": complex_question, "task_id": task_id}

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
    try:
        result = await decompose_agent.run(complex_question, usage=ctx.usage)
    except Exception as exc:
        _record_tool_call(
            ctx,
            tool_name="decompose_question",
            task_id=task_id,
            input_payload=input_payload,
            status="error",
            latency_ms=_latency_ms(start),
            error_code="decomposition_error",
            error=str(exc),
        )
        observation = _observation(
            tool_name="decompose_question",
            status="error",
            warnings=[str(exc)],
            error_code="decomposition_error",
        )
        _record_tool_observation(ctx, observation)
        raise
    sub_questions = result.output
    status = "success" if sub_questions else "empty"
    tool_call_id = _record_tool_call(
        ctx,
        tool_name="decompose_question",
        task_id=task_id,
        input_payload=input_payload,
        status=status,
        result_count=len(sub_questions),
        latency_ms=_latency_ms(start),
    )
    _store_text_result(
        ctx,
        tool_call_id=tool_call_id,
        tool_name="decompose_question",
        task_id=task_id,
        item_id="sub_questions",
        content="\n".join(sub_questions),
    )
    observation = _observation(
        tool_name="decompose_question",
        status=status,
        result_count=len(sub_questions),
    )
    _record_tool_observation(ctx, observation)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            "decompose_question",
            {
                "result_count": len(sub_questions),
                "tool_call_id": tool_call_id,
                "task_id": task_id,
            },
        )

    return observation


async def analyze_section_tool(
    ctx: RunContext[AgentDeps],
    question: str,
    context: str,
    task_id: str | None = None,
) -> ToolObservation:
    """Analyze specific content to answer a focused question.

    Args:
        ctx: Tools context with access to dependencies.
        question: The specific question to analyze.
        context: The reference text to analyze.
        task_id: Checklist task identifier assigned by the planner.

    Returns:
        A lightweight observation. The generated analysis is stored on
        ``FlowContext.tool_results``.
    """
    start = time.perf_counter()
    task_id = task_id or "analyze_section"
    input_payload = {
        "question": question,
        "context_length": len(context),
        "task_id": task_id,
    }

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
    try:
        result = await analysis_agent.run(prompt, usage=ctx.usage)
    except Exception as exc:
        _record_tool_call(
            ctx,
            tool_name="analyze_section",
            task_id=task_id,
            input_payload=input_payload,
            status="error",
            latency_ms=_latency_ms(start),
            error_code="analysis_error",
            error=str(exc),
        )
        observation = _observation(
            tool_name="analyze_section",
            status="error",
            warnings=[str(exc)],
            error_code="analysis_error",
        )
        _record_tool_observation(ctx, observation)
        raise

    tool_call_id = _record_tool_call(
        ctx,
        tool_name="analyze_section",
        task_id=task_id,
        input_payload=input_payload,
        status="success",
        result_count=1,
        latency_ms=_latency_ms(start),
    )
    _store_text_result(
        ctx,
        tool_call_id=tool_call_id,
        tool_name="analyze_section",
        task_id=task_id,
        item_id="analysis",
        content=result.output,
    )
    observation = _observation(
        tool_name="analyze_section",
        status="success",
        result_count=1,
    )
    _record_tool_observation(ctx, observation)

    if ctx.deps.emitter:
        await ctx.deps.emitter.emit_step_completed(
            "analyze_section",
            {
                "question": question,
                "analysis_length": len(result.output),
                "tool_call_id": tool_call_id,
                "task_id": task_id,
            },
        )

    return observation


async def plan_and_reason_tool(
    ctx: RunContext[Any],
    reasoning: str,
    task_id: str | None = None,
) -> ToolObservation:
    """Use this tool to structure your response to the user with reasoning.

    Call this tool to record a lightweight planning note.

    Args:
        ctx: Tools context.
        reasoning: Your detailed reasoning or plan.
        task_id: Checklist task identifier assigned by the planner.
    """
    start = time.perf_counter()
    logger.info(f"Plan and Reasoning: {reasoning}")
    task_id = task_id or "plan_and_reason"
    tool_call_id = _record_tool_call(
        ctx,
        tool_name="plan_and_reason",
        task_id=task_id,
        input_payload={"reasoning": reasoning, "task_id": task_id},
        status="success",
        result_count=1,
        latency_ms=_latency_ms(start),
    )
    _store_text_result(
        ctx,
        tool_call_id=tool_call_id,
        tool_name="plan_and_reason",
        task_id=task_id,
        item_id="reasoning",
        content=reasoning,
    )
    observation = _observation(
        tool_name="plan_and_reason",
        status="success",
        result_count=1,
    )
    _record_tool_observation(ctx, observation)
    return observation


async def get_user_classification_tool(
    ctx: RunContext[Any],
    response: str,
    quick_questions: list[QuestionAnswerSelector] | None = None,
    task_id: str | None = None,
) -> ToolObservation:
    """Get clarification from user if the user's query is too broad.

    Use this tool when you need the user to clarify their intent by selecting
    from specific options. The clarification payload is stored on the workflow
    context; the tool returns only an execution observation.

    Args:
        ctx: Tools context.
        response: The explanatory response or question directed at the user.
        quick_questions: Specific questions and options for the user.
        task_id: Checklist task identifier assigned by the planner.
    """
    start = time.perf_counter()
    logger.info(f"Clarification Question: {response}")
    task_id = task_id or "get_user_classification"

    def _build_quick_questions_markdown(questions: list[QuestionAnswerSelector]) -> str:
        """Build quick question payload in markdown fenced block for UI parsing."""
        if not questions:
            return ""

        lines: list[str] = ["```selection"]
        for index, item in enumerate(questions, start=1):
            question = item.question.strip()
            options = [
                option.strip() for option in item.options if option and option.strip()
            ]

            lines.append(f"{index}. {question}")
            lines.extend(f"- {option}" for option in options)

        lines.append("```")
        return "\n".join(lines)

    quick_questions_markdown = _build_quick_questions_markdown(quick_questions or [])
    answer = (
        response
        if not quick_questions_markdown
        else f"{response}\n\n{quick_questions_markdown}"
    )
    tool_call_id = _record_tool_call(
        ctx,
        tool_name="get_user_classification",
        task_id=task_id,
        input_payload={
            "response": response,
            "question_count": len(quick_questions or []),
            "task_id": task_id,
        },
        status="success",
        result_count=1,
        latency_ms=_latency_ms(start),
    )
    _store_text_result(
        ctx,
        tool_call_id=tool_call_id,
        tool_name="get_user_classification",
        task_id=task_id,
        item_id="clarification",
        content=answer,
    )
    observation = _observation(
        tool_name="get_user_classification",
        status="success",
        result_count=1,
    )
    _record_tool_observation(ctx, observation)

    return observation
