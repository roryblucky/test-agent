"""Config-driven step execution engine with conditional routing.

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

Conditional routing
-------------------
Steps may declare ``routing`` rules that are evaluated after execution.
This enables intent-based early exit, conditional branching, and
skip-ahead without breaking existing linear pipelines.

Actions:

- ``continue`` — proceed to next step (default)
- ``abort``    — stop the pipeline, optionally set a canned response
- ``goto``     — jump to a named step (``step.name``)
- ``skip_to``  — skip forward to a step by its ``type:mode`` label

Every step emits ``step_start`` and ``step_completed`` SSE events with
result payloads. Low-risk LLM answer steps emit ``token`` events, while
compliance-review flows buffer answer text until approval.
On any step failure the pipeline **terminates immediately** (raises).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.config.models import (
    FlowStep,
    FlowStepType,
    StepRoutingAction,
    StepRoutingRule,
    TenantConfig,
)
from app.services.events import EventEmitter, GenerationCancelledError
from app.services.flow_context import FlowContext
from app.services.handlers.base import StepHandler

logger = logging.getLogger(__name__)


class FlowEngine:
    """Executes a pipeline defined by ``flowConfig.steps`` with optional routing.

    Delegates actual work to injected :class:`StepHandler` instances.

    Each step emits ``step_start`` / ``step_completed`` events via the
    :class:`EventEmitter` on the :class:`FlowContext`.

    Usage::

        engine = FlowEngine(tenant_config, handlers)
        emitter = EventEmitter()
        ctx = await engine.execute("What is RAG?", emitter=emitter)
    """

    def __init__(
        self,
        tenant_config: TenantConfig,
        handlers: dict[FlowStepType, StepHandler],
    ) -> None:
        self.steps = tenant_config.flow_config.steps
        self.handlers = handlers
        self.streaming_policy = (
            "approved_answer_only" if self._has_compliance_review_step() else "token"
        )

    async def execute(
        self,
        query: str,
        emitter: EventEmitter | None = None,
        session_id: str | None = None,
        message_history: list | None = None,
    ) -> FlowContext:
        """Run the pipeline end-to-end.

        Raises on first error (fail-fast).
        Stops early if the emitter receives a cancellation signal.
        Evaluates per-step routing rules after each step completes.
        """
        ctx = FlowContext(
            query=query,
            emitter=emitter,
            session_id=session_id,
            message_history=message_history or [],
        )
        ctx.metadata["pipeline_start"] = time.time()
        ctx.metadata["streaming_policy"] = self.streaming_policy

        try:
            step_index = 0
            while step_index < len(self.steps):
                step = self.steps[step_index]

                # Check for stop signal between steps
                if ctx.emitter and ctx.emitter.is_cancelled:
                    ctx.metadata["stopped"] = True
                    break

                handler = self.handlers.get(step.type)
                if handler is None:
                    raise ValueError(f"Unknown flow step type: {step.type}")

                step_name = step.step_label

                # Track for audit trail
                ctx.metadata.setdefault("steps_executed", []).append(step_name)

                # Emit step_start
                if ctx.emitter:
                    await ctx.emitter.emit_step_start(step_name)

                ctx = await handler.handle(ctx, step)

                # --- Conditional routing ---
                if step.routing:
                    routing_result = self._evaluate_routing(
                        ctx, step.routing, step_name
                    )
                    if routing_result is not None:
                        action, rule = routing_result
                        if action == StepRoutingAction.ABORT:
                            ctx.llm_response = self._resolve_routing_response(ctx, rule)
                            ctx.metadata["routed_abort"] = step_name
                            ctx.metadata["abort_reason"] = (
                                f"Routing rule matched: "
                                f"{rule.match_field}={rule.match_value!r}"
                            )
                            logger.info(
                                "Flow aborted at step %r by routing rule "
                                "(field=%s, value=%r)",
                                step_name,
                                rule.match_field,
                                rule.match_value,
                            )
                            break

                        if action == StepRoutingAction.GOTO:
                            target_index = self._find_step_by_name(rule.target_step)
                            logger.info(
                                "Routing: goto step %r (index %d) from %r",
                                rule.target_step,
                                target_index,
                                step_name,
                            )
                            step_index = target_index
                            continue

                        if action == StepRoutingAction.SKIP_TO:
                            target_index = self._find_step_by_label(rule.target_step)
                            logger.info(
                                "Routing: skip_to step %r (index %d) from %r",
                                rule.target_step,
                                target_index,
                                step_name,
                            )
                            step_index = target_index
                            continue

                        # StepRoutingAction.CONTINUE — fall through

                step_index += 1

        except GenerationCancelledError:
            ctx.metadata["stopped"] = True

        return ctx

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _evaluate_routing(
        self,
        ctx: FlowContext,
        rules: list[StepRoutingRule],
        step_name: str,
    ) -> tuple[StepRoutingAction, StepRoutingRule] | None:
        """Evaluate routing rules; return the first match or ``None``."""
        for rule in rules:
            actual = self._resolve_field(ctx, rule.match_field)
            if self._values_match(actual, rule.match_value):
                logger.debug(
                    "Routing rule matched at step %r: %s == %r",
                    step_name,
                    rule.match_field,
                    rule.match_value,
                )
                return rule.action, rule
        return None

    @staticmethod
    def _resolve_field(ctx: FlowContext, dotted_path: str) -> Any:
        """Resolve a dotted field path on the FlowContext.

        Examples::

            _resolve_field(ctx, "intent.intent")  -> ctx.intent.intent
            _resolve_field(ctx, "refined_query")  -> ctx.refined_query
            _resolve_field(ctx, "metadata.key")   -> ctx.metadata["key"]

        Returns ``None`` for any missing attribute or key.
        """
        current: Any = ctx
        for part in dotted_path.split("."):
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
        return current

    @staticmethod
    def _values_match(actual: Any, expected: Any) -> bool:
        """Flexible comparison: supports equality, list membership, and None."""
        if actual is None:
            return expected is None
        if isinstance(expected, list):
            return actual in expected
        return actual == expected

    @staticmethod
    def _resolve_routing_response(
        ctx: FlowContext, rule: StepRoutingRule
    ) -> str | None:
        """Get the response text for an abort rule."""
        if rule.response_from_field:
            value = FlowEngine._resolve_field(ctx, rule.response_from_field)
            if value is not None:
                return str(value)
        return rule.response

    def _find_step_by_name(self, name: str | None) -> int:
        """Find a step index by its ``name`` field."""
        if name is None:
            raise ValueError("Routing rule with goto action requires targetStep")
        for i, step in enumerate(self.steps):
            if step.name == name:
                return i
        raise ValueError(f"Routing target step name {name!r} not found in flow steps")

    def _find_step_by_label(self, label: str | None) -> int:
        """Find a step index by its ``type:mode`` label."""
        if label is None:
            raise ValueError("Routing rule with skip_to action requires targetStep")
        for i, step in enumerate(self.steps):
            if step.step_label == label:
                return i
        raise ValueError(f"Routing target step label {label!r} not found in flow steps")

    def _has_compliance_review_step(self) -> bool:
        return any(
            step.type == FlowStepType.LLM and step.mode == "compliance_review"
            for step in self.steps
        )
