"""Coordinator Agent — Agent Delegation pattern.

Implements pydantic-ai's **Agent Delegation** pattern: a single
Coordinator Agent owns multiple specialist agents as **tools**, and
the LLM autonomously decides which tools to call, in what order,
and how many times.

Architecture (Option 2: Delegation + Fixed Guardrails)::

    ┌──────────────────────────────────────────────────────┐
    │  Fixed Pre-step   │  LLM-Driven Core  │  Fixed Post │
    │                   │                    │             │
    │  moderation ──────▶ Coordinator Agent ──▶ groundedness│
    │  (always runs)    │  (LLM decides      │ (always runs│
    │                   │   which tools)     │             │
    └──────────────────────────────────────────────────────┘

The Coordinator Agent has access to specialist tools:

- ``search_documents`` — retrieve docs from the knowledge base
- ``rank_documents``   — re-rank a doc set against a query
- ``decompose_question`` — break complex questions into sub-questions
- ``analyze_section``  — call an LLM to analyze specific content

The LLM decides the execution plan dynamically.  For example:
- Simple factual question → search → answer
- Complex comparison → decompose → search × N → synthesize
- Multi-section analysis → decompose → search × N → analyze × N → synthesize

This module is **self-contained** and does NOT modify any existing
components (FlowEngine, FlowContext, providers, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.core.model_registry import ModelRegistry
from app.models.domain import Document, GroundednessResult, ModerationResult
from app.providers.base import (
    BaseRankerProvider,
    BaseRetrieverProvider,
)
from app.services.events import EventEmitter
from app.services.exceptions import ContentFlaggedError
from app.services.flow_context import FlowContext


# ---------------------------------------------------------------------------
# Coordinator dependencies — injected into every tool via RunContext
# ---------------------------------------------------------------------------


@dataclass
class CoordinatorDeps:
    """Dependencies available to the Coordinator Agent's tools.

    This is passed as ``deps`` to ``Agent.run()`` and is accessible
    inside every tool via ``ctx.deps``.
    """

    registry: ModelRegistry
    retriever: BaseRetrieverProvider
    ranker: BaseRankerProvider
    emitter: EventEmitter | None = None


# ---------------------------------------------------------------------------
# Coordinator output
# ---------------------------------------------------------------------------


class CoordinatorOutput(BaseModel):
    """Structured output from the Coordinator Agent."""

    answer: str = Field(
        description="The final comprehensive answer to the user's question"
    )
    sources_used: int = Field(0, description="Number of source documents referenced")
    reasoning: str = Field(
        "", description="Brief explanation of the analysis approach taken"
    )


# ---------------------------------------------------------------------------
# Build the Coordinator Agent + specialist tools
# ---------------------------------------------------------------------------


def create_coordinator_agent(
    registry: ModelRegistry,
) -> Agent[CoordinatorDeps, CoordinatorOutput]:
    """Create the Coordinator Agent with specialist tools.

    The Coordinator uses the ``pro`` model for complex reasoning.
    Each tool is a specialist that the LLM can invoke autonomously.
    """

    coordinator = Agent[CoordinatorDeps, CoordinatorOutput](
        registry.get_model("pro").model,
        output_type=CoordinatorOutput,
        model_settings=registry.get_model("pro").settings,
        instructions=(
            "You are an expert analysis coordinator. Your job is to answer the user's "
            "question by strategically using the available tools.\n\n"
            "Available strategies:\n"
            "- For simple questions: search_documents → use results to answer\n"
            "- For complex questions: decompose_question → search per sub-question → "
            "  analyze_section per topic → synthesize into final answer\n"
            "- For comparisons: search each aspect separately → compare results\n\n"
            "Guidelines:\n"
            "- Always search for relevant documents before answering\n"
            "- Use rank_documents if you have many documents to filter\n"
            "- Use decompose_question for multi-part or complex questions\n"
            "- Use analyze_section to get focused analysis on specific topics\n"
            "- Cite specific documents in your final answer\n"
            "- Be thorough but concise"
        ),
    )

    # -- Tool: search_documents -----------------------------------------

    @coordinator.tool
    async def search_documents(ctx: RunContext[CoordinatorDeps], query: str) -> str:
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

    # -- Tool: rank_documents ------------------------------------------

    @coordinator.tool
    async def rank_documents(
        ctx: RunContext[CoordinatorDeps], query: str, document_texts: list[str]
    ) -> str:
        """Re-rank documents by relevance to a specific query.

        Use this when you have many documents and want to focus on the
        most relevant ones for a particular aspect of the question.

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
            Document(id=f"doc_{i}", content=text)
            for i, text in enumerate(document_texts)
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

    # -- Tool: decompose_question --------------------------------------

    @coordinator.tool
    async def decompose_question(
        ctx: RunContext[CoordinatorDeps], complex_question: str
    ) -> list[str]:
        """Break a complex question into 2-5 focused sub-questions.

        Use this for multi-part questions, comparisons, or questions
        that need information from different angles.

        Args:
            complex_question: The complex question to decompose.

        Returns:
            A list of focused sub-questions.
        """
        if ctx.deps.emitter:
            await ctx.deps.emitter.emit_step_start("decompose_question")

        decompose_agent = ctx.deps.registry.create_agent(
            "fast",
            output_type=list[str],
            instructions=(
                "Break the given question into 2-5 specific, focused sub-questions "
                "that can each be answered independently. Each sub-question should "
                "target a distinct aspect of the original question."
            ),
        )
        result = await decompose_agent.run(complex_question, usage=ctx.usage)
        sub_questions = result.output

        if ctx.deps.emitter:
            await ctx.deps.emitter.emit_step_completed(
                "decompose_question",
                {"sub_questions": sub_questions},
            )

        return sub_questions

    # -- Tool: analyze_section -----------------------------------------

    @coordinator.tool
    async def analyze_section(
        ctx: RunContext[CoordinatorDeps],
        question: str,
        context: str,
    ) -> str:
        """Analyze specific content to answer a focused question.

        Use this as a specialist to get deeper analysis on a particular
        topic, using the provided context.

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

    return coordinator


# ---------------------------------------------------------------------------
# DelegationOrchestrator — the public API
# ---------------------------------------------------------------------------


class DelegationOrchestrator:
    """Agent Delegation orchestrator with fixed safety guardrails.

    Execution flow::

        1. [FIXED] Moderation check          ← always runs
        2. [LLM]   Coordinator Agent          ← LLM decides tools
        3. [FIXED] Groundedness check         ← always runs

    This class is self-contained and shares the same ``execute()``
    interface as :class:`~app.services.flow_engine.FlowEngine` and
    :class:`~app.agents.orchestrator.AgentOrchestrator`, so it can
    be used interchangeably.

    Usage::

        delegator = DelegationOrchestrator(registry, providers)
        ctx = await delegator.execute(query, emitter=emitter)
    """

    def __init__(
        self,
        registry: ModelRegistry,
        providers: Any,  # TenantProviders — duck-typed to avoid import
    ) -> None:
        self.registry = registry
        self.providers = providers
        self._coordinator = create_coordinator_agent(registry)

    async def execute(
        self,
        query: str,
        emitter: EventEmitter | None = None,
    ) -> FlowContext:
        """Run the delegation pipeline."""
        ctx = FlowContext(query=query, emitter=emitter)

        # ── Step 1: Fixed guard — Moderation ────────────────────────
        await self._run_moderation(ctx)

        # ── Step 2: LLM-driven core — Coordinator Agent ─────────────
        await self._run_coordinator(ctx)

        # ── Step 3: Fixed guard — Groundedness ──────────────────────
        await self._run_groundedness(ctx)

        return ctx

    # ------------------------------------------------------------------
    # Fixed guardrail steps
    # ------------------------------------------------------------------

    async def _run_moderation(self, ctx: FlowContext) -> None:
        """Always-run moderation check (skipped if provider not configured)."""
        if not self.providers.moderation:
            return  # no moderation provider configured

        if ctx.emitter:
            await ctx.emitter.emit_step_start("moderation")

        result: ModerationResult = await self.providers.moderation.check(ctx.query)
        ctx.moderation_result = result

        if result.is_flagged:
            if ctx.emitter:
                await ctx.emitter.emit_step_completed(
                    "moderation", {"is_flagged": True, "reason": result.reason}
                )
            raise ContentFlaggedError(result)

        if ctx.emitter:
            await ctx.emitter.emit_step_completed("moderation", {"is_flagged": False})

    async def _run_groundedness(self, ctx: FlowContext) -> None:
        """Always-run groundedness check (skipped if provider not configured)."""
        if not self.providers.groundedness:
            return  # no groundedness provider configured
        if not ctx.llm_response or not ctx.ranked_documents:
            return  # nothing to check

        if ctx.emitter:
            await ctx.emitter.emit_step_start("groundedness")

        result: GroundednessResult = await self.providers.groundedness.check(
            ctx.llm_response, ctx.ranked_documents
        )
        ctx.groundedness_result = result

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "groundedness",
                {"is_grounded": result.is_grounded, "score": result.score},
            )

    # ------------------------------------------------------------------
    # LLM-driven core
    # ------------------------------------------------------------------

    async def _run_coordinator(self, ctx: FlowContext) -> None:
        """Run the Coordinator Agent — LLM autonomously calls tools."""
        if ctx.emitter:
            await ctx.emitter.emit_step_start("coordinator")

        deps = CoordinatorDeps(
            registry=self.registry,
            retriever=self.providers.retriever,
            ranker=self.providers.ranker,
            emitter=ctx.emitter,
        )

        # Run the coordinator — LLM decides which tools to invoke
        result = await self._coordinator.run(ctx.query, deps=deps)

        # Extract results
        output: CoordinatorOutput = result.output
        ctx.llm_response = output.answer
        ctx.metadata["sources_used"] = output.sources_used
        ctx.metadata["reasoning"] = output.reasoning

        # Collect documents retrieved during tool execution
        # (they were stored by the tools, available via usage tracking)
        ctx.metadata["coordinator_usage"] = {
            "requests": result.usage().requests,
            "input_tokens": result.usage().input_tokens,
            "output_tokens": result.usage().output_tokens,
        }

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "coordinator",
                {
                    "sources_used": output.sources_used,
                    "reasoning": output.reasoning,
                    "usage": ctx.metadata["coordinator_usage"],
                },
            )
