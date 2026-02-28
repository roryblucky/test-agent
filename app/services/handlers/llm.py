"""Handler for LLM orchestration step."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent

from app.agents.intent_recognition import create_intent_agent
from app.agents.rag_answer import create_rag_answer_agent
from app.agents.refine_question import create_refine_agent
from app.config.models import FlowStep
from app.core.model_registry import ModelRegistry
from app.core.telemetry import trace_span
from app.services.flow_context import FlowContext

# Mapping from config-style keys (camelCase) to pydantic-ai ModelSettings keys
_SETTINGS_KEY_MAP: dict[str, str] = {
    "temperature": "temperature",
    "maxTokens": "max_tokens",
    "max_tokens": "max_tokens",
    "topP": "top_p",
    "top_p": "top_p",
}


def _build_step_settings(step: FlowStep) -> dict[str, Any] | None:
    """Convert ``FlowStep.settings`` to pydantic-ai ``ModelSettings``."""
    if not step.settings:
        return None

    result: dict[str, Any] = {}
    for key, value in step.settings.items():
        mapped_key = _SETTINGS_KEY_MAP.get(key, key)
        result[mapped_key] = value

    return result


class LLMHandler:
    """Handles LLM interactions via unified dispatcher."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        # Agent cache: keyed by (mode, model_name) — pydantic-ai Agent
        # is stateless & thread-safe, safe to reuse across requests.
        self._agent_cache: dict[tuple[str, str], Agent] = {}

    def warmup(self, steps: list[FlowStep]) -> None:
        """Pre-warm cache for agents declared in config steps."""
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
        for step in steps:
            if step.type == "llm":  # String check or enum
                mode = step.mode or "answer"
                model_name = step.model or _DEFAULT_MODELS.get(mode, "pro")
                factory = _AGENT_FACTORIES.get(mode)
                if factory and (mode, model_name) not in self._agent_cache:
                    self._agent_cache[(mode, model_name)] = factory(
                        self.registry, model_name
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

    @trace_span("llm_unified")
    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        """Run LLM step."""
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
        async with agent.run_stream(
            ctx.query,
            model_settings=settings,
            message_history=ctx.message_history or None,
        ) as stream:
            result = await stream.get_output()
            ctx.add_usage(stream.usage())

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
        async with agent.run_stream(
            effective_query,
            model_settings=settings,
            message_history=ctx.message_history or None,
        ) as stream:
            result = await stream.get_output()
            ctx.add_usage(stream.usage())

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

    @trace_span("llm_answer")
    async def _llm_answer(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        model_name = step.model or "pro"
        agent = self._get_agent("answer", model_name)

        context_docs = ctx.ranked_documents or ctx.documents
        context_text = "\n\n---\n\n".join(
            f"[Document {d.id}]\n{d.content}" for d in context_docs
        )
        effective_query = ctx.refined_query or ctx.query

        from app.agents.rag_answer import RAGAgentDeps

        deps = RAGAgentDeps(system_prompt=f"Reference Documents:\n{context_text}")

        # All AI calls use streaming — emit per-token events
        settings = _build_step_settings(step)
        async with agent.run_stream_e(
            effective_query,
            deps=deps,
            model_settings=settings,
            message_history=ctx.message_history or None,
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
            await ctx.emitter.emit_step_completed("llm:answer", {"model": model_name})
        return ctx
