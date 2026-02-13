"""Linear pipeline flow engine — config-driven step execution.

Reads ``flowConfig.steps`` from the tenant config and executes them in
order.  Each step type maps to a handler method.  LLM-related steps
reference a named model from the :class:`ModelRegistry`.

Every step emits ``step_start`` and ``step_completed`` SSE events with
result payloads.  LLM steps additionally emit ``token`` events.
On any step failure the pipeline **terminates immediately** (raises).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.agents.intent_recognition import create_intent_agent
from app.agents.rag_answer import create_rag_answer_agent
from app.agents.refine_question import create_refine_agent
from app.config.models import FlowStep, FlowStepType, TenantConfig
from app.core.model_registry import ModelRegistry
from app.services.events import EventEmitter
from app.services.exceptions import ContentFlaggedError
from app.services.flow_context import FlowContext


class FlowEngine:
    """Executes a linear pipeline defined by ``flowConfig.steps``.

    Each step emits ``step_start`` / ``step_completed`` events via the
    :class:`EventEmitter` on the :class:`FlowContext`.

    Usage::

        engine = FlowEngine(tenant_config, model_registry, providers)
        emitter = EventEmitter()
        ctx = await engine.execute("What is RAG?", emitter=emitter)
    """

    def __init__(
        self,
        tenant_config: TenantConfig,
        registry: ModelRegistry,
        providers: Any,  # TenantProviders (forward ref)
    ) -> None:
        self.steps = tenant_config.flow_config.steps
        self.registry = registry
        self.providers = providers

        self._handlers: dict[
            FlowStepType,
            Callable[[FlowContext, FlowStep], Awaitable[FlowContext]],
        ] = {
            FlowStepType.MODERATION: self._run_moderation,
            FlowStepType.REFINE_QUESTION: self._run_refine_question,
            FlowStepType.INTENT_RECOGNITION: self._run_intent_recognition,
            FlowStepType.RETRIEVER: self._run_retriever,
            FlowStepType.RANKING: self._run_ranking,
            FlowStepType.LLM: self._run_llm,
            FlowStepType.GROUNDEDNESS: self._run_groundedness,
        }

    async def execute(
        self,
        query: str,
        emitter: EventEmitter | None = None,
        session_id: str | None = None,
        message_history: list | None = None,
    ) -> FlowContext:
        """Run the pipeline end-to-end.

        Raises on first error (fail-fast).
        """
        ctx = FlowContext(
            query=query,
            emitter=emitter,
            session_id=session_id,
            message_history=message_history or [],
        )
        for step in self.steps:
            handler = self._handlers.get(step.type)
            if handler is None:
                raise ValueError(f"Unknown flow step type: {step.type}")

            # Emit step_start
            if ctx.emitter:
                await ctx.emitter.emit_step_start(step.type.value)

            ctx = await handler(ctx, step)

            # Emit step_completed (handler populates the result payload)
            # — each handler returns the result via ctx; we extract
            #   the relevant piece for the completed event below.

        return ctx

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    async def _run_moderation(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        if not self.providers.moderation:
            raise ValueError(
                "Flow step 'moderation' requires 'moderationConfig' in tenant config"
            )
        result = await self.providers.moderation.check(ctx.query)
        if result.is_flagged:
            raise ContentFlaggedError(result)
        ctx.moderation_result = result
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "moderation",
                {"is_flagged": result.is_flagged},
            )
        return ctx

    async def _run_refine_question(
        self, ctx: FlowContext, step: FlowStep
    ) -> FlowContext:
        model_name = step.model or "fast"
        agent = create_refine_agent(self.registry, model_name)
        async with agent.run_stream(ctx.query) as stream:
            result = await stream.get_output()
        ctx.refined_query = result.refined_query
        ctx.metadata["keywords"] = result.keywords
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "refine_question",
                {
                    "refined_query": result.refined_query,
                    "keywords": result.keywords,
                },
            )
        return ctx

    async def _run_intent_recognition(
        self, ctx: FlowContext, step: FlowStep
    ) -> FlowContext:
        model_name = step.model or "intent"
        agent = create_intent_agent(self.registry, model_name)
        effective_query = ctx.refined_query or ctx.query
        async with agent.run_stream(effective_query) as stream:
            result = await stream.get_output()
        ctx.intent = result
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "intent_recognition",
                {
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "sub_intents": result.sub_intents,
                },
            )
        return ctx

    async def _run_retriever(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        if not self.providers.retriever:
            raise ValueError(
                "Flow step 'retriever' requires 'retrieverConfig' in tenant config"
            )
        effective_query = ctx.refined_query or ctx.query
        ctx.documents = await self.providers.retriever.retrieve(
            effective_query, self.providers.retriever.config.top_k
        )
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
        return ctx

    async def _run_ranking(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        if not self.providers.ranker:
            raise ValueError(
                "Flow step 'ranking' requires 'rankingConfig' in tenant config"
            )
        effective_query = ctx.refined_query or ctx.query
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
        return ctx

    async def _run_llm(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        model_name = step.model or "pro"
        agent = create_rag_answer_agent(self.registry, model_name)

        context_docs = ctx.ranked_documents or ctx.documents
        context_text = "\n\n---\n\n".join(
            f"[Document {d.id}]\n{d.content}" for d in context_docs
        )
        effective_query = ctx.refined_query or ctx.query
        prompt = (
            f"Reference Documents:\n{context_text}\n\nUser Question: {effective_query}"
        )

        # All AI calls use streaming — emit per-token events
        async with agent.run_stream(prompt) as stream:
            chunks: list[str] = []
            async for chunk in stream.stream_text():
                chunks.append(chunk)
                if ctx.emitter:
                    await ctx.emitter.emit_token(chunk)
            ctx.llm_response = "".join(chunks)

        if ctx.emitter:
            await ctx.emitter.emit_step_completed("llm", {"model": model_name})
        return ctx

    async def _run_groundedness(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        if not self.providers.groundedness:
            raise ValueError(
                "Flow step 'groundedness' requires 'groundednessConfig' in tenant config"
            )
        if ctx.llm_response is None:
            raise ValueError("Groundedness step requires a prior LLM response")
        context_docs = ctx.ranked_documents or ctx.documents
        ctx.groundedness_result = await self.providers.groundedness.check(
            ctx.llm_response, context_docs
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
