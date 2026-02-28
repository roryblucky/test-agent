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

from app.agents.adaptive_router import IntentConfigStore
from app.agents.intent_recognition import create_intent_agent
from app.agents.rag_answer import create_rag_answer_agent
from app.agents.refine_question import create_refine_agent
from app.config.models import MCPServerConfig, UsageLimitConfig
from app.core.model_registry import ModelRegistry
from app.models.domain import Document
from app.providers.base import TenantProvidersProtocol
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
            case "adaptive_rag":
                return await self._adaptive_rag_branch(ctx)
            case "dynamic_router":
                return await self._dynamic_router_branch(ctx)
            case _:
                raise ValueError(
                    f"Unknown agent graph: '{self.graph_name}'. "
                    f"Available: rag_with_intent_branching, annual_report_analysis, adaptive_rag, dynamic_router"
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
        async with refine_agent.run_stream(
            ctx.query, message_history=ctx.message_history or None
        ) as stream:
            refined = await stream.get_output()
            ctx.add_usage(stream.usage())
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
        async with intent_agent.run_stream(
            ctx.refined_query, message_history=ctx.message_history or None
        ) as stream:
            intent = await stream.get_output()
            ctx.add_usage(stream.usage())
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
        answer_agent = create_rag_answer_agent(self.registry)
        from app.agents.rag_answer import RAGAgentDeps

        # Inject context text as a system prompt dependency, not user query,
        # to avoid polluting the user's conversation history with large docs.
        deps = RAGAgentDeps(system_prompt=f"Reference Documents:\n{context_text}")

        async with answer_agent.run_stream(
            effective_query, deps=deps, message_history=ctx.message_history or None
        ) as stream:
            chunks: list[str] = []
            async for chunk in stream.stream_text():
                chunks.append(chunk)
                if ctx.emitter:
                    await ctx.emitter.emit_token(chunk)
            ctx.llm_response = "".join(chunks)
            ctx.new_messages = stream.new_messages()
            ctx.add_usage(stream.usage())
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
        async with agent.run_stream(
            ctx.refined_query or ctx.query, message_history=ctx.message_history or None
        ) as stream:
            chunks: list[str] = []
            async for chunk in stream.stream_text():
                chunks.append(chunk)
                if ctx.emitter:
                    await ctx.emitter.emit_token(chunk)
            ctx.llm_response = "".join(chunks)
            ctx.new_messages = stream.new_messages()
            ctx.add_usage(stream.usage())
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

    # ------------------------------------------------------------------

    async def _adaptive_rag_branch(self, ctx: FlowContext) -> FlowContext:
        """Adaptive RAG route (V2) — Intent-driven dynamic configuration pipeline."""
        from app.services.exceptions import ContentFlaggedError
        from app.agents.rag_answer import RAGAgentDeps

        # 0. Fixed Pre-step: Moderation guard
        if ctx.emitter:
            await ctx.emitter.emit_step_start("moderation")
        if self.providers.moderation:
            moderation_result = await self.providers.moderation.check(ctx.query)
            if moderation_result.is_flagged:
                raise ContentFlaggedError(moderation_result)
            ctx.moderation_result = moderation_result
        if ctx.emitter:
            await ctx.emitter.emit_step_completed("moderation", {"is_flagged": False})

        # 1. Intent Recognition
        if ctx.emitter:
            await ctx.emitter.emit_step_start("intent_recognition")
        intent_agent = create_intent_agent(self.registry)
        async with intent_agent.run_stream(
            ctx.query, message_history=ctx.message_history or None
        ) as stream:
            intent = await stream.get_output()
            ctx.add_usage(stream.usage())
        ctx.intent = intent

        # Load dynamic config based on intent
        config = IntentConfigStore.get_config(intent.intent)
        ctx.metadata["intent_config"] = {
            "intent": config.intent_name,
            "data_sources": [
                {
                    "source": ds.source_name,
                    "rerank_model": ds.rerank_model,
                    "is_mcp": ds.is_mcp,
                }
                for ds in config.data_sources
            ],
        }

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "intent_recognition",
                {
                    "intent": intent.intent,
                    "confidence": intent.confidence,
                    "config_used": config.intent_name,
                },
            )

        # 2. Refine Question (Conditional)
        effective_query = ctx.query
        if config.needs_refine:
            if ctx.emitter:
                await ctx.emitter.emit_step_start("refine_question")
            refine_agent = create_refine_agent(self.registry)
            async with refine_agent.run_stream(
                ctx.query, message_history=ctx.message_history or None
            ) as stream:
                refined = await stream.get_output()
                ctx.add_usage(stream.usage())
            effective_query = refined.refined_query
            ctx.refined_query = effective_query
            if ctx.emitter:
                await ctx.emitter.emit_step_completed(
                    "refine_question",
                    {"refined_query": effective_query, "keywords": refined.keywords},
                )

        # 3. Retrieve & Rerank (Conditional based on source config)
        all_ranked_documents: list[Document] = []
        is_mcp_intent = False

        for ds_config in config.data_sources:
            if ds_config.is_mcp:
                is_mcp_intent = True
                continue

            # Conceptually, self.providers.retriever should dispatch based on ds_config.source_name.
            # Here we pass the specific source config to the retriever if supported,
            # otherwise we just call the default configured retriever for the tenant.
            if ctx.emitter:
                await ctx.emitter.emit_step_start("retriever")

            # Using standard retrieve for now. If TenantProviders gets updated to support
            # multi-retrievers, this would be `self.providers.get_retriever(ds_config.source_name)`
            retrieved_docs = await self.providers.retriever.retrieve(effective_query)
            ctx.documents.extend(retrieved_docs)

            if ctx.emitter:
                await ctx.emitter.emit_step_completed(
                    "retriever",
                    {
                        "document_count": len(retrieved_docs),
                        "source": ds_config.source_name,
                    },
                )

            # Conditional Reranking if configured for this intent
            if ds_config.rerank_model and self.providers.ranker:
                if ctx.emitter:
                    await ctx.emitter.emit_step_start("ranking")
                ranked_docs = await self.providers.ranker.rank(
                    effective_query, retrieved_docs
                )
                if ctx.emitter:
                    await ctx.emitter.emit_step_completed(
                        "ranking",
                        {
                            "document_count": len(ranked_docs),
                            "model": ds_config.rerank_model,
                        },
                    )
                all_ranked_documents.extend(ranked_docs)
            else:
                all_ranked_documents.extend(retrieved_docs)

        # Optional: Further global re-ranking could happen here if needed.
        # But for now, we just aggregate the re-ranked docs from all sources.
        ctx.ranked_documents = all_ranked_documents

        # 4. Answer generation using Dynamic Prompt
        if ctx.emitter:
            await ctx.emitter.emit_step_start("llm")

        # Build context if documents exist
        context_text = ""
        if ctx.ranked_documents:
            context_text = "\n\n---\n\n".join(
                f"[Document {d.id}]\n{d.content}" for d in ctx.ranked_documents
            )

        # Inject dynamic context directly into RAG deps along with the intent-specific system prompt
        system_prompt = (
            f"{config.system_prompt}\n\nReference Documents:\n{context_text}"
            if context_text
            else config.system_prompt
        )
        deps = RAGAgentDeps(system_prompt=system_prompt)

        # If the intent requires MCP tools, inject them into the RAG agent
        mcp_servers = self._mcp_configs if is_mcp_intent else None
        answer_agent = create_rag_answer_agent(self.registry, mcp_configs=mcp_servers)

        async with answer_agent.run_stream(
            effective_query, deps=deps, message_history=ctx.message_history or None
        ) as stream:
            chunks: list[str] = []
            async for chunk in stream.stream_text():
                chunks.append(chunk)
                if ctx.emitter:
                    await ctx.emitter.emit_token(chunk)
            ctx.llm_response = "".join(chunks)
            ctx.new_messages = stream.new_messages()
            ctx.add_usage(stream.usage())

        if ctx.emitter:
            await ctx.emitter.emit_step_completed("llm", {"model": "pro"})

        # 5. Fixed Post-step: Groundedness check (Only if context exists)
        if ctx.ranked_documents and self.providers.groundedness:
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

    # ------------------------------------------------------------------
    # Graph: dynamic router (ReAct / Agentic Handoff)
    # ------------------------------------------------------------------

    async def _dynamic_router_branch(self, ctx: FlowContext) -> FlowContext:
        """Dynamic ReAct execution with sandwich guardrails."""
        from pydantic_ai.usage import UsageLimits

        from app.agents.router_agent import SharedState, create_router_agent
        from app.services.exceptions import ContentFlaggedError

        # 1. Fixed Pre-step: Moderation guard
        if ctx.emitter:
            await ctx.emitter.emit_step_start("moderation")
        if self.providers.moderation:
            moderation_result = await self.providers.moderation.check(ctx.query)
            if moderation_result.is_flagged:
                if ctx.emitter:
                    await ctx.emitter.emit_step_completed(
                        "moderation",
                        {"is_flagged": True, "reason": moderation_result.reason},
                    )
                raise ContentFlaggedError(moderation_result)
            ctx.moderation_result = moderation_result
        if ctx.emitter:
            await ctx.emitter.emit_step_completed("moderation", {"is_flagged": False})

        # 2. Dynamic Core: Router Agent (ReAct Loop)
        if ctx.emitter:
            await ctx.emitter.emit_step_start("router_agent_core")

        shared_state = SharedState(
            registry=self.registry,
            retriever=self.providers.retriever,
            ranker=self.providers.ranker,
            emitter=ctx.emitter,
        )

        # Re-use MCP toolsets if provided
        from app.core.mcp import build_mcp_toolsets

        mcp_toolsets = build_mcp_toolsets(self._mcp_configs or [])

        router_agent = create_router_agent(self.registry, extra_toolsets=mcp_toolsets)

        # Determine usage limits for loop control (prevent infinite ReAct loops)
        if self._usage_limit_config:
            usage_limits = UsageLimits(
                request_limit=self._usage_limit_config.request_limit or 5,
                total_tokens_limit=self._usage_limit_config.total_tokens_limit,
            )
        else:
            usage_limits = UsageLimits(request_limit=5)

        async with router_agent.run_stream(
            ctx.query,
            deps=shared_state,
            message_history=ctx.message_history or None,
            usage_limits=usage_limits,
        ) as stream:
            chunks: list[str] = []
            async for chunk in stream.stream_text():
                chunks.append(chunk)
                if ctx.emitter:
                    await ctx.emitter.emit_token(chunk)
            ctx.llm_response = "".join(chunks)
            ctx.new_messages = stream.new_messages()
            ctx.add_usage(stream.usage())

        # Push retrieved docs into ctx for groundedness and final return
        ctx.ranked_documents = shared_state.retrieved_documents
        # Distinctly identify that dynamic ReAct was used
        ctx.metadata["dynamic_react_used"] = True

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "router_agent_core",
                {"model": "pro", "retrieved_docs_count": len(ctx.ranked_documents)},
            )

        # 3. Fixed Post-step: Groundedness & Content Moderation
        # (Assuming groundedness serves as a post-step validation)
        if ctx.ranked_documents and self.providers.groundedness:
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
