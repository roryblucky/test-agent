"""Linear pipeline flow engine — config-driven step execution.

Reads ``flowConfig.steps`` from the tenant config and executes them in
order.  Each step ``type`` maps to a *module handler*, and ``mode``
selects the specific action within that module.

Module types
------------
- **moderation** — ``pre`` (check query) / ``post`` (check answer)
- **llm** — unified LLM dispatcher; ``mode`` selects the agent factory
  (``refine_question``, ``intent``, ``answer``, …)
- **retriever** — document retrieval
- **ranking** — document re-ranking
- **groundedness** — answer groundedness checking
- **analysis** — pipeline observability (token usage, timing, storage)
- **memory** — session / long-term memory persistence (future)

Every step emits ``step_start`` and ``step_completed`` SSE events with
result payloads.  LLM steps additionally emit ``token`` events.
On any step failure the pipeline **terminates immediately** (raises).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai import Agent

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
            FlowStepType.LLM: self._run_llm,
            FlowStepType.RETRIEVER: self._run_retriever,
            FlowStepType.RANKING: self._run_ranking,
            FlowStepType.GROUNDEDNESS: self._run_groundedness,
            FlowStepType.ANALYSIS: self._run_analysis,
            FlowStepType.MEMORY: self._run_memory,
        }

        # Agent cache: keyed by (mode, model_name) — pydantic-ai Agent
        # is stateless & thread-safe, safe to reuse across requests.
        self._agent_cache: dict[tuple[str, str], Agent] = {}

        # Pre-warm cache for agents declared in config steps
        _AGENT_FACTORIES: dict[str, Callable] = {
            "refine_question": create_refine_agent,
            "intent": create_intent_agent,
            "answer": create_rag_answer_agent,
        }
        _DEFAULT_MODELS: dict[str, str] = {
            "refine_question": "fast",
            "intent": "intent",
            "answer": "pro",
        }
        for step in self.steps:
            if step.type == FlowStepType.LLM:
                mode = step.mode or "answer"
                model_name = step.model or _DEFAULT_MODELS.get(mode, "pro")
                factory = _AGENT_FACTORIES.get(mode)
                if factory and (mode, model_name) not in self._agent_cache:
                    self._agent_cache[(mode, model_name)] = factory(
                        registry, model_name
                    )

    def _get_agent(self, mode: str, model_name: str) -> Agent:
        """Get or create a cached Agent for (mode, model_name)."""
        key = (mode, model_name)
        if key not in self._agent_cache:
            factories: dict[str, Callable] = {
                "refine_question": create_refine_agent,
                "intent": create_intent_agent,
                "answer": create_rag_answer_agent,
            }
            factory = factories.get(mode)
            if factory is None:
                raise ValueError(f"No agent factory for LLM mode: {mode!r}")
            self._agent_cache[key] = factory(self.registry, model_name)
        return self._agent_cache[key]

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
        ctx.metadata["pipeline_start"] = time.time()

        for step in self.steps:
            handler = self._handlers.get(step.type)
            if handler is None:
                raise ValueError(f"Unknown flow step type: {step.type}")

            # Build a human-readable step name for SSE events
            step_name = (
                f"{step.type.value}:{step.mode}" if step.mode else step.type.value
            )

            # Emit step_start
            if ctx.emitter:
                await ctx.emitter.emit_step_start(step_name)

            ctx = await handler(ctx, step)

        return ctx

    # ------------------------------------------------------------------
    # Module handlers
    # ------------------------------------------------------------------

    # ── Moderation ────────────────────────────────────────────────────

    async def _run_moderation(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Run content moderation.

        - ``mode="pre"``  (default) — check user's query
        - ``mode="post"`` — check the AI-generated answer
        """
        if not self.providers.moderation:
            raise ValueError(
                "Flow step 'moderation' requires 'moderationConfig' in tenant config"
            )

        mode = step.mode or "pre"
        step_name = f"moderation:{mode}"

        if mode == "pre":
            # Check the user's input query
            result = await self.providers.moderation.check(ctx.query)
            if result.is_flagged:
                raise ContentFlaggedError(result)
            ctx.moderation_result = result
        elif mode == "post":
            # Check the AI-generated answer
            if ctx.llm_response is None:
                raise ValueError("Moderation 'post' requires a prior LLM response")
            result = await self.providers.moderation.check(ctx.llm_response)
            if result.is_flagged:
                # Replace the answer with a safe message instead of raising
                ctx.llm_response = (
                    "The generated response was flagged by content moderation "
                    "and has been removed."
                )
            ctx.metadata["post_moderation"] = {
                "is_flagged": result.is_flagged,
            }
        else:
            raise ValueError(f"Unknown moderation mode: {mode!r}")

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                step_name,
                {"is_flagged": result.is_flagged, "mode": mode},
            )
        return ctx

    # ── LLM (unified dispatcher) ─────────────────────────────────────

    async def _run_llm(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Unified LLM handler — ``step.mode`` selects the agent factory.

        Supported modes:
        - ``refine_question`` — rewrite / refine the query
        - ``intent``          — classify intent
        - ``answer``          — generate RAG answer (default)

        ``step.settings`` (if provided) overrides the model's base config
        for this call only (temperature, maxTokens, topP, …).
        """
        mode = step.mode or "answer"

        match mode:
            case "refine_question":
                return await self._llm_refine_question(ctx, step)
            case "intent":
                return await self._llm_intent(ctx, step)
            case "answer":
                return await self._llm_answer(ctx, step)
            case _:
                raise ValueError(f"Unknown llm mode: {mode!r}")

    async def _llm_refine_question(
        self, ctx: FlowContext, step: FlowStep
    ) -> FlowContext:
        model_name = step.model or "fast"
        agent = self._get_agent("refine_question", model_name)
        settings = _build_step_settings(step)
        async with agent.run_stream(ctx.query, model_settings=settings) as stream:
            result = await stream.get_output()
        ctx.refined_query = result.refined_query
        ctx.metadata["keywords"] = result.keywords
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "llm:refine_question",
                {
                    "refined_query": result.refined_query,
                    "keywords": result.keywords,
                    "model": model_name,
                },
            )
        return ctx

    async def _llm_intent(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        model_name = step.model or "intent"
        agent = self._get_agent("intent", model_name)
        effective_query = ctx.refined_query or ctx.query
        settings = _build_step_settings(step)
        async with agent.run_stream(effective_query, model_settings=settings) as stream:
            result = await stream.get_output()
        ctx.intent = result
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "llm:intent",
                {
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "sub_intents": result.sub_intents,
                    "model": model_name,
                },
            )
        return ctx

    async def _llm_answer(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        model_name = step.model or "pro"
        agent = self._get_agent("answer", model_name)

        context_docs = ctx.ranked_documents or ctx.documents
        context_text = "\n\n---\n\n".join(
            f"[Document {d.id}]\n{d.content}" for d in context_docs
        )
        effective_query = ctx.refined_query or ctx.query
        prompt = (
            f"Reference Documents:\n{context_text}\n\nUser Question: {effective_query}"
        )

        # All AI calls use streaming — emit per-token events
        settings = _build_step_settings(step)
        async with agent.run_stream(prompt, model_settings=settings) as stream:
            chunks: list[str] = []
            async for chunk in stream.stream_text():
                chunks.append(chunk)
                if ctx.emitter:
                    await ctx.emitter.emit_token(chunk)
            ctx.llm_response = "".join(chunks)

        if ctx.emitter:
            await ctx.emitter.emit_step_completed("llm:answer", {"model": model_name})
        return ctx

    # ── Retriever ─────────────────────────────────────────────────────

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

    # ── Ranking ────────────────────────────────────────────────────────

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

    # ── Groundedness ──────────────────────────────────────────────────

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

    # ── Analysis (observability) ──────────────────────────────────────

    async def _run_analysis(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Aggregate pipeline execution data for observability.

        Collects timing, token usage, and step summaries.  Results are
        stored in ``ctx.metadata["analysis"]`` for downstream consumers
        (BigQuery, logging, dashboards, …).
        """
        pipeline_start = ctx.metadata.get("pipeline_start")
        elapsed = time.time() - pipeline_start if pipeline_start else None

        analysis = {
            "pipeline_duration_seconds": round(elapsed, 3) if elapsed else None,
            "session_id": ctx.session_id,
            "query": ctx.query,
            "refined_query": ctx.refined_query,
            "answer_length": len(ctx.llm_response) if ctx.llm_response else 0,
            "documents_retrieved": len(ctx.documents),
            "documents_ranked": len(ctx.ranked_documents),
            "is_grounded": (
                ctx.groundedness_result.is_grounded if ctx.groundedness_result else None
            ),
            "token_usage": ctx.metadata.get("coordinator_usage"),
        }

        ctx.metadata["analysis"] = analysis

        if ctx.emitter:
            await ctx.emitter.emit_step_completed("analysis", analysis)

        return ctx

    # ── Memory (future) ───────────────────────────────────────────────

    async def _run_memory(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Persist session / long-term memory.

        Currently a no-op placeholder.  Future: integrate with
        ``BaseSessionStore`` or ``BaseLongTermMemory``.
        """
        if ctx.emitter:
            await ctx.emitter.emit_step_completed("memory", {"mode": step.mode})
        return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mapping from config-style keys (camelCase) to pydantic-ai ModelSettings keys
_SETTINGS_KEY_MAP: dict[str, str] = {
    "temperature": "temperature",
    "maxTokens": "max_tokens",
    "max_tokens": "max_tokens",
    "topP": "top_p",
    "top_p": "top_p",
}


def _build_step_settings(step: FlowStep) -> dict[str, Any] | None:
    """Convert ``FlowStep.settings`` to pydantic-ai ``ModelSettings``.

    - Accepts both camelCase (``maxTokens``) and snake_case (``max_tokens``)
    - Returns ``None`` when no overrides are present (agent defaults apply)
    - Unknown keys are passed through as-is (provider-specific settings)
    """
    if not step.settings:
        return None

    result: dict[str, Any] = {}
    for key, value in step.settings.items():
        mapped_key = _SETTINGS_KEY_MAP.get(key, key)
        result[mapped_key] = value

    return result
