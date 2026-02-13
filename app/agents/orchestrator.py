"""Complex multi-agent orchestrator — expert-chain pattern.

Uses pydantic-ai's *programmatic agent hand-off* pattern to run
multiple *expert agents*, each with a distinct responsibility.

Predefined graph names
----------------------
``flowConfig.mode = "agent"`` + ``flowConfig.agentGraph`` selects a
pre-defined expert chain.  Currently supported:

- ``"rag_with_intent_branching"`` — moderation → refine → intent → branch
- ``"annual_report_analysis"`` — multi-expert chain for document analysis

To add a new expert pipeline, register it in ``_GRAPHS`` and create the
corresponding agents in ``app/agents/``.
"""

from __future__ import annotations


from app.providers.base import TenantProvidersProtocol
from app.agents.intent_recognition import create_intent_agent
from app.agents.rag_answer import create_rag_answer_agent
from app.agents.refine_question import create_refine_agent
from app.core.model_registry import ModelRegistry
from app.config.models import MCPServerConfig, UsageLimitConfig
from app.services.events import EventEmitter
from app.services.flow_context import FlowContext


class AgentOrchestrator:
    """Complex multi-agent orchestration via programmatic hand-off.

    Each expert agent handles one phase of the pipeline.
    Step events are emitted throughout.

    Usage::

        orchestrator = AgentOrchestrator(registry, providers,
                                        graph_name="rag_with_intent_branching")
        emitter = EventEmitter()
        ctx = await orchestrator.execute(query, emitter=emitter)
    """

    def __init__(
        self,
        registry: ModelRegistry,
        providers: TenantProvidersProtocol,
        graph_name: str = "rag_with_intent_branching",
        usage_limit_config: UsageLimitConfig | None = None,
        mcp_configs: list[MCPServerConfig] | None = None,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self.graph_name = graph_name
        self._usage_limit_config = usage_limit_config
        self._mcp_configs = mcp_configs

    async def execute(
        self,
        query: str,
        emitter: EventEmitter | None = None,
        session_id: str | None = None,
        message_history: list | None = None,
    ) -> FlowContext:
        """Dispatch to the named expert-chain graph."""
        ctx = FlowContext(
            query=query,
            emitter=emitter,
            session_id=session_id,
            message_history=message_history or [],
        )

        match self.graph_name:
            case "rag_with_intent_branching":
                return await self._rag_with_intent(ctx)
            case "annual_report_analysis":
                return await self._annual_report_analysis(ctx)
            case _:
                raise ValueError(
                    f"Unknown agent graph: '{self.graph_name}'. "
                    f"Available: rag_with_intent_branching, annual_report_analysis"
                )

    # ------------------------------------------------------------------
    # Graph: rag_with_intent_branching
    #   Expert A (intent) → Expert B (refine) → branch(rag / chitchat)
    # ------------------------------------------------------------------

    async def _rag_with_intent(self, ctx: FlowContext) -> FlowContext:
        """Moderation → refine → intent → branch(rag / chitchat)."""
        from app.services.exceptions import ContentFlaggedError

        # Expert 0: Moderation guard
        if ctx.emitter:
            await ctx.emitter.emit_step_start("moderation")
        moderation_result = await self.providers.moderation.check(ctx.query)
        if moderation_result.is_flagged:
            raise ContentFlaggedError(moderation_result)
        ctx.moderation_result = moderation_result
        if ctx.emitter:
            await ctx.emitter.emit_step_completed("moderation", {"is_flagged": False})

        # Expert A: Question refinement
        if ctx.emitter:
            await ctx.emitter.emit_step_start("refine_question")
        refine_agent = create_refine_agent(self.registry)
        async with refine_agent.run_stream(ctx.query) as stream:
            refined = await stream.get_output()
        ctx.refined_query = refined.refined_query
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "refine_question",
                {"refined_query": refined.refined_query, "keywords": refined.keywords},
            )

        # Expert B: Intent recognition
        if ctx.emitter:
            await ctx.emitter.emit_step_start("intent_recognition")
        intent_agent = create_intent_agent(self.registry)
        async with intent_agent.run_stream(ctx.refined_query) as stream:
            intent = await stream.get_output()
        ctx.intent = intent
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "intent_recognition",
                {"intent": intent.intent, "confidence": intent.confidence},
            )

        # Branch by intent
        match intent.intent:
            case "knowledge_query" | "comparison" | "summarization":
                return await self._rag_branch(ctx)
            case "chitchat":
                return await self._chitchat_branch(ctx)
            case _:
                return await self._rag_branch(ctx)

    async def _rag_branch(self, ctx: FlowContext) -> FlowContext:
        """Full RAG pipeline: retrieve → rank → answer → groundedness."""
        effective_query = ctx.refined_query or ctx.query

        # Expert C: Retrieval
        if ctx.emitter:
            await ctx.emitter.emit_step_start("retriever")
        ctx.documents = await self.providers.retriever.retrieve(effective_query)
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "retriever",
                {
                    "document_count": len(ctx.documents),
                    "documents": [
                        {"id": d.id, "score": d.score} for d in ctx.documents
                    ],
                },
            )

        # Expert D: Ranking
        if ctx.emitter:
            await ctx.emitter.emit_step_start("ranking")
        ctx.ranked_documents = await self.providers.ranker.rank(
            effective_query, ctx.documents
        )
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "ranking",
                {
                    "document_count": len(ctx.ranked_documents),
                    "documents": [
                        {"id": d.id, "score": d.score} for d in ctx.ranked_documents
                    ],
                },
            )

        # Expert E: Answer generation (streaming tokens)
        if ctx.emitter:
            await ctx.emitter.emit_step_start("llm")
        context_text = "\n\n---\n\n".join(
            f"[Document {d.id}]\n{d.content}" for d in ctx.ranked_documents
        )
        prompt = (
            f"Reference Documents:\n{context_text}\n\nUser Question: {effective_query}"
        )
        answer_agent = create_rag_answer_agent(self.registry)
        async with answer_agent.run_stream(prompt) as stream:
            chunks: list[str] = []
            async for chunk in stream.stream_text():
                chunks.append(chunk)
                if ctx.emitter:
                    await ctx.emitter.emit_token(chunk)
            ctx.llm_response = "".join(chunks)
        if ctx.emitter:
            await ctx.emitter.emit_step_completed("llm", {"model": "pro"})

        # Expert F: Groundedness check
        if ctx.emitter:
            await ctx.emitter.emit_step_start("groundedness")
        ctx.groundedness_result = await self.providers.groundedness.check(
            ctx.llm_response, ctx.ranked_documents
        )
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "groundedness",
                {
                    "is_grounded": ctx.groundedness_result.is_grounded,
                    "score": ctx.groundedness_result.score,
                },
            )
        return ctx

    async def _chitchat_branch(self, ctx: FlowContext) -> FlowContext:
        """Simple chat without RAG — direct LLM response."""
        if ctx.emitter:
            await ctx.emitter.emit_step_start("llm")
        agent = self.registry.create_agent(
            "fast",
            output_type=str,
            instructions="You are a friendly assistant. Respond conversationally.",
        )
        async with agent.run_stream(ctx.refined_query or ctx.query) as stream:
            chunks: list[str] = []
            async for chunk in stream.stream_text():
                chunks.append(chunk)
                if ctx.emitter:
                    await ctx.emitter.emit_token(chunk)
            ctx.llm_response = "".join(chunks)
        if ctx.emitter:
            await ctx.emitter.emit_step_completed("llm", {"model": "fast"})
        return ctx

    # ------------------------------------------------------------------
    # Graph: delegation-based analysis
    #   Uses Agent Delegation pattern — Coordinator LLM decides tools
    # ------------------------------------------------------------------

    async def _annual_report_analysis(self, ctx: FlowContext) -> FlowContext:
        """Delegation-based analysis — Coordinator Agent decides the plan.

        Delegates to :class:`~app.agents.coordinator.DelegationOrchestrator`
        which wraps fixed guardrails (moderation + groundedness) around
        a Coordinator Agent that autonomously invokes specialist tools.

        See ``app/agents/coordinator.py`` for the full implementation.
        """
        from app.agents.coordinator import DelegationOrchestrator

        delegator = DelegationOrchestrator(
            self.registry,
            self.providers,
            usage_limit_config=self._usage_limit_config,
            mcp_configs=self._mcp_configs,
        )
        return await delegator.execute(
            ctx.query,
            emitter=ctx.emitter,
            session_id=ctx.session_id,
            message_history=ctx.message_history,
        )
