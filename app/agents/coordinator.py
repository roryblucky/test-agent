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

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ThinkingPart
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import UsageLimits

from app.config.models import MCPServerConfig, UsageLimitConfig
from app.core.model_registry import ModelRegistry
from app.models.domain import GroundednessResult, ModerationResult
from app.providers.base import (
    BaseRankerProvider,
    BaseRetrieverProvider,
    TenantProvidersProtocol,
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
    extra_toolsets: list[AbstractToolset] | None = None,
) -> Agent[CoordinatorDeps, CoordinatorOutput]:
    """Create the Coordinator Agent with specialist tools.

    The Coordinator uses the ``pro`` model for complex reasoning.
    Each tool is a specialist that the LLM can invoke autonomously.
    """
    from app.agents.history_processors import filter_thinking, trim_history

    coordinator = Agent[CoordinatorDeps, CoordinatorOutput](
        registry.get_model("pro").model,
        output_type=CoordinatorOutput,
        model_settings=registry.get_model("pro").settings,
        toolsets=extra_toolsets or [],
        history_processors=[
            trim_history(max_messages=20),
            filter_thinking(),
        ],
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

    # Register tools from tools.py
    from app.agents.tools import (
        analyze_section_tool,
        decompose_question_tool,
        rank_documents_tool,
        search_documents_tool,
    )

    coordinator.tool(search_documents_tool)
    coordinator.tool(rank_documents_tool)
    coordinator.tool(decompose_question_tool)
    coordinator.tool(analyze_section_tool)

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
        providers: TenantProvidersProtocol,
        usage_limit_config: UsageLimitConfig | None = None,
        mcp_configs: list[MCPServerConfig] | None = None,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self._coordinator = create_coordinator_agent(
            registry,
            extra_toolsets=_build_mcp_toolsets(mcp_configs or []),
        )
        self._usage_limits = _build_usage_limits(usage_limit_config)

    async def execute(
        self,
        query: str,
        emitter: EventEmitter | None = None,
        session_id: str | None = None,
        message_history: list | None = None,
    ) -> FlowContext:
        """Run the delegation pipeline."""
        ctx = FlowContext(
            query=query,
            emitter=emitter,
            session_id=session_id,
            message_history=message_history or [],
        )

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
        result = await self._coordinator.run(
            ctx.query,
            deps=deps,
            usage_limits=self._usage_limits,
            message_history=ctx.message_history or None,
        )

        # Store new messages for session persistence
        ctx.new_messages = result.all_messages()

        # ── Extract thinking / reasoning traces ─────────────────────
        thinking_parts: list[str] = []
        for msg in result.all_messages():
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ThinkingPart) and part.content:
                        thinking_parts.append(part.content)

        if thinking_parts:
            ctx.metadata["thinking"] = thinking_parts
            if ctx.emitter:
                for thought in thinking_parts:
                    await ctx.emitter.emit_thinking(thought)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mcp_toolsets(
    configs: list[MCPServerConfig],
) -> list[AbstractToolset]:
    """Convert MCP server configs to pydantic-ai toolset instances."""
    if not configs:
        return []

    toolsets: list[AbstractToolset] = []
    for cfg in configs:
        if cfg.url:
            # SSE transport (legacy) vs Streamable HTTP
            if cfg.url.rstrip("/").endswith("/sse"):
                from pydantic_ai.mcp import MCPServerSSE

                toolsets.append(MCPServerSSE(cfg.url, tool_prefix=cfg.name))
            else:
                from pydantic_ai.mcp import MCPServerStreamableHTTP

                toolsets.append(MCPServerStreamableHTTP(cfg.url, tool_prefix=cfg.name))
        elif cfg.command:
            from pydantic_ai.mcp import MCPServerStdio

            toolsets.append(
                MCPServerStdio(
                    cfg.command,
                    args=cfg.args,
                    env=cfg.env,
                    tool_prefix=cfg.name,
                )
            )
    return toolsets


def _build_usage_limits(cfg: UsageLimitConfig | None) -> UsageLimits | None:
    """Convert config to pydantic-ai ``UsageLimits``, or ``None`` for defaults."""
    if cfg is None:
        return None
    return UsageLimits(
        request_limit=cfg.request_limit,
        tool_calls_limit=cfg.tool_calls_limit,
        input_tokens_limit=cfg.input_tokens_limit,
        output_tokens_limit=cfg.output_tokens_limit,
        total_tokens_limit=cfg.total_tokens_limit,
    )
