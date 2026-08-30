"""Tests for FlowEngine conditional routing.

Verifies that routing rules on steps correctly abort, skip, or goto
based on FlowContext field values after step execution.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.config.models import (
    FlowConfig,
    FlowStep,
    FlowStepType,
    LLMConfig,
    ModelConfig,
    StepRoutingAction,
    StepRoutingRule,
    TenantConfig,
)
from app.services.flow_context import FlowContext
from app.services.flow_engine import FlowEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tenant(steps: list[FlowStep]) -> TenantConfig:
    """Build a minimal TenantConfig with the given steps."""
    return TenantConfig(
        kms_app_name="test",
        application_id="test-app",
        ad_groups=["test"],
        llm_config=LLMConfig(
            models={"fast": ModelConfig(provider="azure", model_name="gpt-4o-mini")}
        ),
        flow_config=FlowConfig(steps=steps),
    )


type Mutator = Callable[[FlowContext, FlowStep], None]


class RecordingHandler:
    """Stub handler that records calls and optionally mutates context."""

    def __init__(self, name: str, mutator: Mutator | None = None) -> None:
        self.intent = name
        self.calls: list[str] = []
        self._mutator = mutator

    async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
        label = step.step_label
        self.calls.append(label)
        if self._mutator:
            self._mutator(ctx, step)
        return ctx


def _intent_mutator(intent_value: str) -> Mutator:
    """Return a mutator that sets ctx.intent to a mock IntentResult."""

    def _mutate(ctx: FlowContext, step: FlowStep) -> None:
        from app.models.workflow import IntentResult

        del step
        ctx.intent = IntentResult(intent=intent_value, confidence=0.95)

    return _mutate


def _clarification_mutator() -> Mutator:
    """Return a mutator that sets a generic clarification request."""

    def _mutate(ctx: FlowContext, step: FlowStep) -> None:
        del step
        ctx.metadata["needs_clarification"] = True
        ctx.clarification_request = {
            "response": "Which product are you asking about?",
            "quick_questions": [],
        }

    return _mutate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlowRoutingAbort:
    """Test the abort routing action."""

    @pytest.mark.asyncio
    async def test_abort_on_intent_match(self):
        """Pipeline aborts when intent matches out_of_scope."""
        llm_handler = RecordingHandler("llm", mutator=_intent_mutator("out_of_scope"))

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        match_field="intent.intent",
                        match_value="out_of_scope",
                        action=StepRoutingAction.ABORT,
                        response="This question is out of scope.",
                    ),
                ],
            ),
            FlowStep(type=FlowStepType.LLM, mode="answer"),
        ]

        engine = FlowEngine(
            _make_tenant(steps),
            {FlowStepType.LLM: llm_handler},
        )
        ctx = await engine.execute("random question")

        assert ctx.llm_response == "This question is out of scope."
        assert ctx.metadata.get("routed_abort") == "llm:intent"
        # answer step should NOT have been called
        assert "llm:answer" not in llm_handler.calls

    @pytest.mark.asyncio
    async def test_abort_with_response_from_field(self):
        """Pipeline abort can pull response from a FlowContext field."""
        llm_handler = RecordingHandler("llm", mutator=_clarification_mutator())

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        match_field="metadata.needs_clarification",
                        match_value=True,
                        action=StepRoutingAction.ABORT,
                        response_from_field="clarification_request.response",
                    ),
                ],
            ),
            FlowStep(type=FlowStepType.LLM, mode="answer"),
        ]

        engine = FlowEngine(
            _make_tenant(steps),
            {FlowStepType.LLM: llm_handler},
        )
        ctx = await engine.execute("something")

        assert ctx.llm_response == "Which product are you asking about?"

    @pytest.mark.asyncio
    async def test_no_abort_when_no_match(self):
        """Pipeline continues when no routing rule matches."""
        llm_handler = RecordingHandler(
            "llm", mutator=_intent_mutator("knowledge_query")
        )

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        match_field="intent.intent",
                        match_value="out_of_scope",
                        action=StepRoutingAction.ABORT,
                        response="Out of scope.",
                    ),
                ],
            ),
            FlowStep(type=FlowStepType.LLM, mode="answer"),
        ]

        engine = FlowEngine(
            _make_tenant(steps),
            {FlowStepType.LLM: llm_handler},
        )
        ctx = await engine.execute("valid question")

        # Both steps should execute
        assert llm_handler.calls == ["llm:intent", "llm:answer"]
        assert ctx.llm_response is None  # RecordingHandler doesn't set it


class TestFlowRoutingSkipTo:
    """Test the skip_to routing action."""

    @pytest.mark.asyncio
    async def test_skip_to_step_label(self):
        """Pipeline skips forward to a step identified by type:mode label."""
        handler = RecordingHandler("llm", mutator=_intent_mutator("simple_query"))

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        match_field="intent.intent",
                        match_value="simple_query",
                        action=StepRoutingAction.SKIP_TO,
                        target_step="llm:answer",
                    ),
                ],
            ),
            FlowStep(type=FlowStepType.LLM, mode="refine_question"),
            FlowStep(type=FlowStepType.LLM, mode="answer"),
        ]

        engine = FlowEngine(
            _make_tenant(steps),
            {FlowStepType.LLM: handler},
        )
        await engine.execute("simple question")

        # refine_question should be skipped
        assert handler.calls == ["llm:intent", "llm:answer"]


class TestFlowRoutingGoto:
    """Test the goto routing action."""

    @pytest.mark.asyncio
    async def test_goto_named_step(self):
        """Pipeline jumps to a named step."""
        handler = RecordingHandler("llm", mutator=_intent_mutator("fast_answer"))
        analysis_handler = RecordingHandler("analysis")

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        match_field="intent.intent",
                        match_value="fast_answer",
                        action=StepRoutingAction.GOTO,
                        target_step="final_answer",
                    ),
                ],
            ),
            FlowStep(type=FlowStepType.LLM, mode="refine_question"),
            FlowStep(
                type=FlowStepType.LLM,
                mode="answer",
                name="final_answer",
            ),
            FlowStep(type=FlowStepType.ANALYSIS),
        ]

        engine = FlowEngine(
            _make_tenant(steps),
            {
                FlowStepType.LLM: handler,
                FlowStepType.ANALYSIS: analysis_handler,
            },
        )
        await engine.execute("fast question")

        # refine_question skipped, answer and analysis both execute
        assert handler.calls == ["llm:intent", "llm:answer"]
        assert analysis_handler.calls == ["analysis"]


class TestFlowRoutingListMatch:
    """Test matching against a list of values."""

    @pytest.mark.asyncio
    async def test_match_value_in_list(self):
        """Routing rule matches when actual value is in a list of expected."""
        handler = RecordingHandler("llm", mutator=_intent_mutator("chitchat"))

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        match_field="intent.intent",
                        match_value=["out_of_scope", "chitchat"],
                        action=StepRoutingAction.ABORT,
                        response="I can only answer knowledge queries.",
                    ),
                ],
            ),
            FlowStep(type=FlowStepType.LLM, mode="answer"),
        ]

        engine = FlowEngine(
            _make_tenant(steps),
            {FlowStepType.LLM: handler},
        )
        ctx = await engine.execute("hello!")

        assert ctx.llm_response == "I can only answer knowledge queries."


class TestFlowRoutingNoRouting:
    """Verify backward compatibility — no routing = linear execution."""

    @pytest.mark.asyncio
    async def test_linear_execution_without_routing(self):
        """Steps without routing execute sequentially as before."""
        handler = RecordingHandler("llm")
        analysis_handler = RecordingHandler("analysis")

        steps = [
            FlowStep(type=FlowStepType.LLM, mode="refine_question"),
            FlowStep(type=FlowStepType.LLM, mode="intent"),
            FlowStep(type=FlowStepType.LLM, mode="answer"),
            FlowStep(type=FlowStepType.ANALYSIS),
        ]

        engine = FlowEngine(
            _make_tenant(steps),
            {
                FlowStepType.LLM: handler,
                FlowStepType.ANALYSIS: analysis_handler,
            },
        )
        await engine.execute("any question")

        assert handler.calls == [
            "llm:refine_question",
            "llm:intent",
            "llm:answer",
        ]
        assert analysis_handler.calls == ["analysis"]


class TestResolveField:
    """Test the _resolve_field utility."""

    def test_simple_field(self):
        ctx = FlowContext(query="test")
        ctx.refined_query = "refined"
        assert FlowEngine.resolve_field(ctx, "refined_query") == "refined"

    def test_nested_field(self):
        from app.models.workflow import IntentResult

        ctx = FlowContext(query="test")
        ctx.intent = IntentResult(intent="knowledge_query", confidence=0.9)
        assert FlowEngine.resolve_field(ctx, "intent.intent") == "knowledge_query"
        assert FlowEngine.resolve_field(ctx, "intent.confidence") == 0.9

    def test_metadata_dict_field(self):
        ctx = FlowContext(query="test")
        ctx.metadata["custom_key"] = "custom_value"
        assert FlowEngine.resolve_field(ctx, "metadata.custom_key") == "custom_value"

    def test_missing_field_returns_none(self):
        ctx = FlowContext(query="test")
        assert FlowEngine.resolve_field(ctx, "intent.intent") is None
        assert FlowEngine.resolve_field(ctx, "nonexistent") is None

    def test_deeply_nested_none(self):
        ctx = FlowContext(query="test")
        assert FlowEngine.resolve_field(ctx, "intent.intent.deep") is None
