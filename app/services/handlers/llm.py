"""Handler for LLM orchestration step.

Integrates with :class:`LayeredPromptBuilder` so that all LLM modes
(refine_question, intent, answer, compliance_review) respect the
tenant/domain contract layers — not just the Agent step.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    UserPromptPart,
)

from app.agents.intent_recognition import (
    DEFAULT_INSTRUCTIONS as _INTENT_INSTRUCTIONS,
)
from app.agents.intent_recognition import (
    DEFAULT_INTENT_CATALOG,
    create_intent_agent,
)
from app.agents.query_understanding import (
    DEFAULT_INSTRUCTIONS as _QUERY_UNDERSTANDING_INSTRUCTIONS,
)
from app.agents.query_understanding import (
    create_query_understanding_agent,
)
from app.agents.rag_answer import create_rag_answer_agent
from app.agents.refine_question import (
    DEFAULT_INSTRUCTIONS as _REFINE_INSTRUCTIONS,
)
from app.agents.refine_question import (
    create_refine_agent,
)
from app.config.models import FlowStep, TenantConfig
from app.core.model_registry import ModelRegistry
from app.core.telemetry import trace_span
from app.models.domain import RefinedQuestion
from app.models.workflow import (
    AggregatedEvidenceBundle,
    IntentCatalogItem,
    QueryUnderstandingClarification,
    QueryUnderstandingClarificationQuestion,
    QueryUnderstandingOutput,
    ResolvedQuery,
)
from app.prompts.builder import LayeredPromptBuilder
from app.services.flow_context import FlowContext

# Mapping from config-style keys (camelCase) to pydantic-ai ModelSettings keys
_SETTINGS_KEY_MAP: dict[str, str] = {
    "temperature": "temperature",
    "maxTokens": "max_tokens",
    "max_tokens": "max_tokens",
    "topP": "top_p",
    "top_p": "top_p",
}
_NON_MODEL_SETTINGS_KEYS = {"intentCatalog", "intent_catalog"}

_COMPLIANCE_BLOCKED_RESPONSE = (
    "The draft answer could not be released because it did not pass compliance review."
)
_HISTORY_MAX_TURNS = 10
_HISTORY_MAX_CHARS = 12_000
_CONVERSATION_REFERENCE_INTENTS = {
    "summarize_history",
    "revise_previous",
    "continue_previous",
    "compare_previous",
}


def _build_step_settings(step: FlowStep) -> dict[str, Any] | None:
    """Convert ``FlowStep.settings`` to pydantic-ai ``ModelSettings``."""
    if not step.settings:
        return None

    result: dict[str, Any] = {}
    for key, value in step.settings.items():
        if key in _NON_MODEL_SETTINGS_KEYS:
            continue
        mapped_key = _SETTINGS_KEY_MAP.get(key, key)
        result[mapped_key] = value

    return result


class LLMHandler:
    """Handles LLM interactions via unified dispatcher.

    Accepts an optional :class:`TenantConfig` so that all LLM modes
    can include tenant/domain contract layers in their system prompts
    via :class:`LayeredPromptBuilder`.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        tenant_config: TenantConfig | None = None,
    ):
        self.registry = registry
        self.tenant_config = tenant_config
        # Agent cache: keyed by (mode, model_name) — pydantic-ai Agent
        # is stateless & thread-safe, safe to reuse across requests.
        self._agent_cache: dict[tuple[str, str], Agent] = {}

    # Map modes to their canonical default instructions.  Imported from the
    # agent factory modules so there is a single source of truth.
    _MODE_IDENTITY: dict[str, str] = {
        "refine_question": _REFINE_INSTRUCTIONS,
        "intent": _INTENT_INSTRUCTIONS,
        "query_understanding": _QUERY_UNDERSTANDING_INSTRUCTIONS,
    }

    def _build_layered_instructions(self, mode: str) -> str:
        """Build a **static** layered prompt for API-level prompt caching.

        Every mode gets a static ``instructions`` string at Agent build
        time.  This prefix is identical across requests for the same
        tenant, allowing API providers (Anthropic, OpenAI) to cache it.

        Per-request answer context (documents, evidence) is sent as the
        runtime user prompt and then sanitized from persisted history.

        For ``refine_question``, ``intent``, and ``compliance_review``,
        the mode-specific role becomes the **identity layer**.
        For ``answer``, the default enterprise assistant identity is used.
        """
        identity = self._MODE_IDENTITY.get(mode)  # None → default identity
        return LayeredPromptBuilder.build_from_config(
            tenant_config=self.tenant_config,
            identity=identity,
        )

    def warmup(self, steps: list[FlowStep]) -> None:
        """Pre-warm cache for agents declared in config steps."""
        _AGENT_FACTORIES: dict[str, Callable] = {
            "refine_question": create_refine_agent,
            "intent": create_intent_agent,
            "query_understanding": create_query_understanding_agent,
            "answer": create_rag_answer_agent,
        }
        _DEFAULT_MODELS: dict[str, str] = {
            "refine_question": "fast",
            "intent": "intent",
            "query_understanding": "intent",
            "answer": "pro",
        }
        for step in steps:
            if step.type == "llm":  # String check or enum
                mode = step.mode or "answer"
                model_name = step.model or _DEFAULT_MODELS.get(mode, "pro")
                factory = _AGENT_FACTORIES.get(mode)
                if factory and (mode, model_name) not in self._agent_cache:
                    # All modes now get static instructions at build time
                    # so API-level prompt caching can cache the prefix.
                    self._agent_cache[(mode, model_name)] = factory(
                        self.registry, model_name,
                        instructions=self._build_layered_instructions(mode),
                    )

    def _get_agent(self, mode: str, model_name: str) -> Agent:
        """Get or create a cached Agent for (mode, model_name)."""
        key = (mode, model_name)
        if key not in self._agent_cache:
            factories: dict[str, Callable] = {
                "refine_question": create_refine_agent,
                "intent": create_intent_agent,
                "query_understanding": create_query_understanding_agent,
                "answer": create_rag_answer_agent,
            }
            factory = factories.get(mode)
            if factory is None:
                raise ValueError(f"No agent factory for LLM mode: {mode!r}")
            self._agent_cache[key] = factory(
                self.registry, model_name,
                instructions=self._build_layered_instructions(mode),
            )
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
            case "query_understanding":
                return await self._llm_query_understanding(ctx, step)
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
        runtime_prompt = self._build_query_resolver_runtime_prompt(ctx)
        async with agent.run_stream(
            runtime_prompt,
            model_settings=settings,
        ) as stream:
            result = await stream.get_output()
            ctx.add_usage(stream.usage())

        resolved_query = self._coerce_resolved_query(ctx, result)
        self._store_resolved_query(ctx, resolved_query)
        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "llm:refine_question",
                {
                    "refined_query": resolved_query.standalone_query,
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

    async def _llm_query_understanding(
        self, ctx: FlowContext, step: FlowStep
    ) -> FlowContext:
        model_name = step.model or "intent"
        agent = self._get_agent("query_understanding", model_name)
        settings = _build_step_settings(step)
        runtime_prompt = self._build_query_understanding_runtime_prompt(ctx, step)
        async with agent.run_stream(
            runtime_prompt,
            model_settings=settings,
        ) as stream:
            result = await stream.get_output()
            ctx.add_usage(stream.usage())

        output = self._coerce_query_understanding_output(ctx, result)
        self._store_resolved_query(ctx, output.resolved_query)
        ctx.intent = output.intent
        clarification = output.clarification

        if clarification is not None:
            self._apply_query_understanding_clarification(ctx, clarification)

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "llm:query_understanding",
                {
                    "refined_query": output.resolved_query.standalone_query,
                    "needs_clarification": clarification is not None,
                    "intent": output.intent.intent,
                    "confidence": output.intent.confidence,
                    "sub_intents": output.intent.sub_intents,
                    "model": model_name,
                },
            )
        return ctx

    @trace_span("llm_answer")
    async def _llm_answer(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        model_name = step.model or "pro"
        agent = self._get_agent("answer", model_name)

        if ctx.aggregated_evidence and not ctx.aggregated_evidence.synthesis_allowed:
            ctx.llm_response = (
                ctx.aggregated_evidence.synthesis_block_reason
                or "Unable to synthesize an answer from the available evidence."
            )
            if ctx.emitter:
                await ctx.emitter.emit_step_completed(
                    "llm:answer",
                    {"model": model_name, "blocked": True},
                )
            return ctx

        context_text = self._build_answer_context(ctx)
        runtime_prompt = self._build_answer_runtime_prompt(ctx, context_text)

        from app.agents.rag_answer import RAGAgentDeps

        deps = RAGAgentDeps()

        settings = _build_step_settings(step)
        async with agent.run_stream(
            runtime_prompt,
            deps=deps,
            model_settings=settings,
        ) as stream:
            chunks: list[str] = []
            async for chunk in stream.stream_text():
                # Check for stop signal
                if ctx.emitter and ctx.emitter.is_cancelled:
                    break
                chunks.append(chunk)
                if ctx.emitter:
                    await ctx.emitter.emit_token(chunk)
            ctx.llm_response = "".join(chunks)
            ctx.new_messages = _sanitize_answer_new_messages(
                stream.new_messages(),
                visible_user_query=ctx.query,
            )
            ctx.add_usage(stream.usage())

        # Signal stop if cancelled mid-stream
        if ctx.emitter and ctx.emitter.is_cancelled:
            await ctx.emitter.emit_stopped(ctx.llm_response)
            from app.services.events import GenerationCancelledError

            raise GenerationCancelledError("Stopped during llm:answer")

        # Run citation extractor service
        from app.services.citation_extractor import build_citations

        citations, usage = await build_citations(
            answer=ctx.llm_response or "",
            evidence_items=(
                ctx.aggregated_evidence.selected_evidence
                if ctx.aggregated_evidence else []
            ),
            documents=ctx.ranked_documents or ctx.documents,
            registry=self.registry,
        )
        ctx.metadata["citations"] = [
            citation.model_dump(mode="json")
            for citation in citations
        ]
        ctx.add_usage(usage)

        if ctx.emitter and citations:
            await ctx.emitter.emit_citations(
                [citation.model_dump(mode="json") for citation in citations]
            )

        if ctx.emitter:
            await ctx.emitter.emit_step_completed(
                "llm:answer",
                {"model": model_name, "buffered": False},
            )
        return ctx

    @staticmethod
    def _build_answer_context(ctx: FlowContext) -> str:
        if ctx.aggregated_evidence is not None:
            return _format_aggregated_evidence(ctx.aggregated_evidence)

        context_docs = ctx.ranked_documents or ctx.documents
        query_lines = [f"User Query: {ctx.query}"]
        standalone_query = ctx.refined_query or (
            ctx.resolved_query.standalone_query if ctx.resolved_query else None
        )
        if standalone_query and standalone_query != ctx.query:
            query_lines.append(f"Standalone Query: {standalone_query}")

        document_parts = []
        for idx, d in enumerate(context_docs, start=1):
            doc_str = f"[Document [{idx}] | id={d.id}]\nCite as: [{idx}]"
            if getattr(d, "section_title", None):
                doc_str += f"\nSection: {d.section_title}"
            doc_str += f"\n{d.content}"
            document_parts.append(doc_str)
        document_text = "\n\n---\n\n".join(document_parts)
        if document_text:
            return "\n".join(query_lines) + "\n\nReference Documents:\n" + document_text
        return "\n".join(query_lines)

    @staticmethod
    def _build_answer_runtime_prompt(ctx: FlowContext, reference_data: str) -> str:
        """Build the per-run answer input sent as a user prompt."""
        standalone_query = ctx.refined_query or (
            ctx.resolved_query.standalone_query if ctx.resolved_query else None
        )
        conversation_reference = _conversation_reference_for_answer(ctx)
        conversation_block = ""
        if conversation_reference:
            conversation_block = (
                "\n\n"
                "<conversation_reference>\n"
                f"{conversation_reference}\n"
                "</conversation_reference>\n"
            )
        return (
            "<runtime_answer_input>\n"
            "<runtime_rules>\n"
            "- ALWAYS cite retrieved factual claims using inline markers [n].\n"
            "- Use only citation indexes shown in evidence tags.\n"
            "- Cite as [1], [2], or [1][3] for multi-source claims.\n"
            "- Do NOT invent citation numbers.\n"
            "- Do NOT cite unsupported claims.\n"
            "- Use conversation_reference only when it is provided.\n"
            "- conversation_reference is sanitized prior conversation approved for this run.\n"
            "- Do not infer from prior conversation unless it appears in conversation_reference.\n"
            "</runtime_rules>\n\n"
            "<original_user_query>\n"
            f"{ctx.query}\n"
            "</original_user_query>\n\n"
            "<standalone_query>\n"
            f"{standalone_query or ctx.query}\n"
            "</standalone_query>\n\n"
            f"{conversation_block}"
            "<reference_data>\n"
            f"{reference_data}\n"
            "</reference_data>\n"
            "</runtime_answer_input>"
        )

    @staticmethod
    def _build_query_resolver_runtime_prompt(ctx: FlowContext) -> str:
        recent_history = _sanitize_visible_message_history(ctx.message_history)
        rolling_summary = str(ctx.metadata.get("conversation_summary") or "")
        return (
            "<query_resolver_input>\n"
            "<latest_user_query>\n"
            f"{ctx.query}\n"
            "</latest_user_query>\n\n"
            "<recent_chat_history>\n"
            f"{recent_history}\n"
            "</recent_chat_history>\n\n"
            "<rolling_summary>\n"
            f"{rolling_summary}\n"
            "</rolling_summary>\n"
            "</query_resolver_input>"
        )

    @staticmethod
    def _build_query_understanding_runtime_prompt(
        ctx: FlowContext,
        step: FlowStep,
    ) -> str:
        recent_history = _sanitize_visible_message_history(ctx.message_history)
        rolling_summary = str(ctx.metadata.get("conversation_summary") or "")
        intent_catalog = _intent_catalog_for_step(step)
        intent_catalog_json = _dump_runtime_json(
            [item.model_dump(mode="json") for item in intent_catalog]
        )
        return (
            "<query_understanding_input>\n"
            "<latest_user_query>\n"
            f"{ctx.query}\n"
            "</latest_user_query>\n\n"
            "<recent_chat_history>\n"
            f"{recent_history}\n"
            "</recent_chat_history>\n\n"
            "<rolling_summary>\n"
            f"{rolling_summary}\n"
            "</rolling_summary>\n\n"
            "<intent_catalog>\n"
            f"{intent_catalog_json}\n"
            "</intent_catalog>\n"
            "</query_understanding_input>"
        )

    @staticmethod
    def _coerce_resolved_query(ctx: FlowContext, output: Any) -> ResolvedQuery:
        if isinstance(output, ResolvedQuery):
            return output.model_copy(update={"original_query": ctx.query})
        if isinstance(output, RefinedQuestion):
            return ResolvedQuery(
                original_query=ctx.query,
                standalone_query=output.refined_query,
            )
        if isinstance(output, dict) and "refined_query" in output:
            return ResolvedQuery(
                original_query=ctx.query,
                standalone_query=str(output["refined_query"]),
            )
        resolved = ResolvedQuery.model_validate(output)
        return resolved.model_copy(update={"original_query": ctx.query})

    @staticmethod
    def _coerce_query_understanding_output(
        ctx: FlowContext,
        output: Any,
    ) -> QueryUnderstandingOutput:
        if isinstance(output, QueryUnderstandingOutput):
            understanding = output
        else:
            understanding = QueryUnderstandingOutput.model_validate(output)
        resolved_query = understanding.resolved_query.model_copy(
            update={"original_query": ctx.query}
        )
        return understanding.model_copy(update={"resolved_query": resolved_query})

    @staticmethod
    def _store_resolved_query(
        ctx: FlowContext,
        resolved_query: ResolvedQuery,
    ) -> None:
        ctx.refined_query = resolved_query.standalone_query
        ctx.resolved_query = resolved_query

    @staticmethod
    def _apply_query_understanding_clarification(
        ctx: FlowContext,
        clarification: QueryUnderstandingClarification,
    ) -> None:
        questions = [
            question for question in clarification.questions if question.question.strip()
        ]
        if not questions:
            if clarification.scope == "query_resolution":
                fallback_question = "Could you clarify what you mean?"
            else:
                fallback_question = (
                    "Could you clarify what you want this workflow to do?"
                )
            questions = [
                QueryUnderstandingClarificationQuestion(
                    question=fallback_question,
                )
            ]

        response = questions[0].question
        ctx.llm_response = response
        ctx.clarification_request = {
            "response": response,
            "quick_questions": [
                {"question": question.question, "options": question.options}
                for question in questions
            ],
        }
        ctx.metadata["stop_flow"] = True
        if clarification.scope == "query_resolution":
            stop_reason = "query_understanding_needs_query_clarification"
        else:
            stop_reason = "query_understanding_needs_intent_clarification"
        ctx.metadata["stop_reason"] = stop_reason

def _intent_catalog_for_step(step: FlowStep) -> list[IntentCatalogItem]:
    """Normalize optional per-step intent catalog settings."""
    settings = step.settings or {}
    raw_catalog = settings.get("intentCatalog")
    if raw_catalog is None:
        raw_catalog = settings.get("intent_catalog")
    if raw_catalog is None:
        return list(DEFAULT_INTENT_CATALOG)

    if isinstance(raw_catalog, dict):
        if "intent" in raw_catalog or "name" in raw_catalog:
            raw_items: Any = [raw_catalog]
        elif "items" in raw_catalog:
            raw_items = raw_catalog["items"]
        elif "intents" in raw_catalog:
            raw_items = raw_catalog["intents"]
        else:
            raw_items = list(raw_catalog.values())
    else:
        raw_items = raw_catalog

    if not isinstance(raw_items, list):
        raise ValueError("step.settings.intentCatalog must be a list of intent items")

    return [
        item
        if isinstance(item, IntentCatalogItem)
        else IntentCatalogItem.model_validate(item)
        for item in raw_items
    ]


def _conversation_reference_for_answer(ctx: FlowContext) -> str:
    """Return sanitized prior conversation only for conversation intents."""
    if ctx.intent is None:
        return ""
    if ctx.intent.intent not in _CONVERSATION_REFERENCE_INTENTS:
        return ""
    return _sanitize_visible_message_history(ctx.message_history)


def _dump_runtime_json(value: Any) -> str:
    """Serialize runtime context for prompt embedding."""
    return json.dumps(value, ensure_ascii=False, default=str)


def _format_aggregated_evidence(bundle: AggregatedEvidenceBundle) -> str:
    """Format aggregated evidence for answer synthesis."""
    lines = [
        "Aggregated Evidence Bundle",
        f"User Query: {bundle.user_query}",
        f"Standalone Query: {bundle.standalone_query}",
    ]
    if bundle.intent:
        lines.append(f"Intent: {bundle.intent}")
    if bundle.active_skills:
        lines.append(f"Active Skills: {', '.join(bundle.active_skills)}")
    if bundle.missing_tasks:
        lines.append(f"Missing Tasks: {', '.join(bundle.missing_tasks)}")
    if bundle.partial_tasks:
        lines.append(f"Partial Tasks: {', '.join(bundle.partial_tasks)}")
    if bundle.stale_tasks:
        lines.append(f"Stale Tasks: {', '.join(bundle.stale_tasks)}")
    if bundle.failed_tasks:
        lines.append(f"Failed Tasks: {', '.join(bundle.failed_tasks)}")
    if bundle.conflicting_evidence:
        lines.append(f"Conflicts: {', '.join(bundle.conflicting_evidence)}")

    lines.append("\nEvidence:")
    for item in bundle.selected_evidence:
        if isinstance(item.citation_index, int):
            header = f"[Evidence [{item.citation_index}] | source={item.source}"
            header += f" | relevance={item.relevance}"
            if item.score is not None:
                header += f" | score={item.score}"
            header += "]"
            parts = [header, f"Cite as: [{item.citation_index}]"]
        else:
            header = f"[Evidence {item.evidence_id} | source={item.source}"
            header += f" | relevance={item.relevance}"
            if item.score is not None:
                header += f" | score={item.score}"
            header += "]"
            parts = [header]

        if item.title:
            parts.append(f"Title: {item.title}")
        if item.url:
            parts.append(f"URL: {item.url}")
        if item.published_at:
            parts.append(f"Published At: {item.published_at.isoformat()}")
        parts.append(item.content)
        lines.append("\n".join(parts))

    return "\n\n---\n\n".join(lines)


def _sanitize_answer_new_messages(
    messages: list[ModelMessage],
    *,
    visible_user_query: str,
) -> list[ModelMessage]:
    """Replace the runtime answer prompt with the frontend-visible query."""
    sanitized: list[ModelMessage] = []
    replaced_user_prompt = False

    for message in messages:
        if isinstance(message, ModelRequest) and not replaced_user_prompt:
            parts = []
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not replaced_user_prompt:
                    parts.append(replace(part, content=visible_user_query))
                    replaced_user_prompt = True
                else:
                    parts.append(part)
            sanitized.append(replace(message, parts=parts))
        else:
            sanitized.append(message)

    return sanitized


def _sanitize_visible_message_history(
    messages: list[ModelMessage],
    *,
    max_turns: int = _HISTORY_MAX_TURNS,
    max_chars: int = _HISTORY_MAX_CHARS,
) -> str:
    """Render user-visible chat history for the query resolver.

    This deliberately ignores system instructions, tool returns, raw
    runtime prompts, and message metadata.
    """
    turns: list[tuple[str, str]] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    text = _visible_user_prompt_text(part)
                    if text:
                        turns.append(("user", text))
        elif isinstance(message, ModelResponse):
            text_parts = [
                part.content.strip()
                for part in message.parts
                if isinstance(part, TextPart) and part.content.strip()
            ]
            if text_parts:
                turns.append(("assistant", "\n".join(text_parts)))

    if not turns:
        return ""

    selected_turns = turns[-max_turns:]
    rendered = "\n\n".join(
        f"[{role}]\n{text}" for role, text in selected_turns
    )
    if len(rendered) <= max_chars:
        return rendered
    return "[truncated]\n" + rendered[-max_chars:]


def _visible_user_prompt_text(part: UserPromptPart) -> str:
    content = part.content
    if isinstance(content, str):
        text = content.strip()
    else:
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, TextContent):
                chunks.append(item.content)
        text = "\n".join(chunk.strip() for chunk in chunks if chunk.strip())

    if not text:
        return ""

    runtime_original = _extract_xml_text(text, "original_user_query")
    if runtime_original:
        return runtime_original
    if "<runtime_answer_input" in text or "<reference_data>" in text:
        return ""
    return text


def _extract_xml_text(text: str, tag: str) -> str | None:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_index = text.find(start_tag)
    if start_index < 0:
        return None
    content_start = start_index + len(start_tag)
    end_index = text.find(end_tag, content_start)
    if end_index < 0:
        return None
    extracted = text[content_start:end_index].strip()
    return extracted or None
