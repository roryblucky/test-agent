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

from app.config.models import FlowStepType, TenantConfig
from app.services.events import EventEmitter
from app.services.flow_context import FlowContext
from app.services.handlers.base import StepHandler


class FlowEngine:
    """Executes a linear pipeline defined by ``flowConfig.steps``.

    Refactored to use the Strategy pattern.  Delegates actual work to
    injected :class:`StepHandler` instances.

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
            handler = self.handlers.get(step.type)
            if handler is None:
                raise ValueError(f"Unknown flow step type: {step.type}")

            # Build a human-readable step name for SSE events
            step_name = (
                f"{step.type.value}:{step.mode}" if step.mode else step.type.value
            )

            # Emit step_start
            if ctx.emitter:
                await ctx.emitter.emit_step_start(step_name)

            ctx = await handler.handle(ctx, step)

        return ctx
