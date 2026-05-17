"""Tests for FlowEngine conditional routing.

Verifies that routing rules on steps correctly abort, skip, or goto
based on FlowContext field values after step execution.
"""

from __future__ import annotations

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
        kmsAppName="test",
        applicationId="test-app",
        adGroups=["test"],
        llmConfig=LLMConfig(
            models={"fast": ModelConfig(provider="azure", modelName="gpt-4o-mini")}
        ),
        flowConfig=FlowConfig(steps=steps),
    )


class RecordingHandler:
    """Stub handler that records calls and optionally mutates context."""

    def __init__(self, name: str, mutator=None):
        self.name = name
        self.calls: list[str] = []
        self._mutator = mutator

    async def handle(self, ctx: FlowContext, step) -> FlowContext:
        label = step.step_label
        self.calls.append(label)
        if self._mutator:
            self._mutator(ctx, step)
        return ctx


def _intent_mutator(intent_value: str):
    """Return a mutator that sets ctx.intent to a mock IntentResult."""

    def _mutate(ctx: FlowContext, step):
        from app.models.workflow import IntentResult

        ctx.intent = IntentResult(intent=intent_value, confidence=0.95)

    return _mutate


def _clarification_mutator():
    """Return a mutator that sets needs_clarification on intent."""

    def _mutate(ctx: FlowContext, step):
        from app.models.workflow import IntentResult

        ctx.intent = IntentResult(
            intent="clarification_needed",
            confidence=0.90,
            needs_clarification=True,
            clarification_question="Which product are you asking about?",
        )

    return _mutate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlowRoutingAbort:
    """Test the abort routing action."""

    @pytest.mark.asyncio
    async def test_abort_on_intent_match(self):
        """Pipeline aborts when intent matches out_of_scope."""
        llm_handler = RecordingHandler(
            "llm", mutator=_intent_mutator("out_of_scope")
        )
        answer_handler = RecordingHandler("answer_llm")

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        matchField="intent.intent",
                        matchValue="out_of_scope",
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
                        matchField="intent.needs_clarification",
                        matchValue=True,
                        action=StepRoutingAction.ABORT,
                        responseFromField="intent.clarification_question",
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
                        matchField="intent.intent",
                        matchValue="out_of_scope",
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
        handler = RecordingHandler(
            "llm", mutator=_intent_mutator("simple_query")
        )

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        matchField="intent.intent",
                        matchValue="simple_query",
                        action=StepRoutingAction.SKIP_TO,
                        targetStep="llm:answer",
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
        ctx = await engine.execute("simple question")

        # refine_question should be skipped
        assert handler.calls == ["llm:intent", "llm:answer"]


class TestFlowRoutingGoto:
    """Test the goto routing action."""

    @pytest.mark.asyncio
    async def test_goto_named_step(self):
        """Pipeline jumps to a named step."""
        handler = RecordingHandler(
            "llm", mutator=_intent_mutator("fast_answer")
        )
        analysis_handler = RecordingHandler("analysis")

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        matchField="intent.intent",
                        matchValue="fast_answer",
                        action=StepRoutingAction.GOTO,
                        targetStep="final_answer",
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
        ctx = await engine.execute("fast question")

        # refine_question skipped, answer and analysis both execute
        assert handler.calls == ["llm:intent", "llm:answer"]
        assert analysis_handler.calls == ["analysis"]


class TestFlowRoutingListMatch:
    """Test matching against a list of values."""

    @pytest.mark.asyncio
    async def test_match_value_in_list(self):
        """Routing rule matches when actual value is in a list of expected."""
        handler = RecordingHandler(
            "llm", mutator=_intent_mutator("chitchat")
        )

        steps = [
            FlowStep(
                type=FlowStepType.LLM,
                mode="intent",
                routing=[
                    StepRoutingRule(
                        matchField="intent.intent",
                        matchValue=["out_of_scope", "chitchat"],
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
        ctx = await engine.execute("any question")

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
        assert FlowEngine._resolve_field(ctx, "refined_query") == "refined"

    def test_nested_field(self):
        from app.models.workflow import IntentResult

        ctx = FlowContext(query="test")
        ctx.intent = IntentResult(intent="knowledge_query", confidence=0.9)
        assert FlowEngine._resolve_field(ctx, "intent.intent") == "knowledge_query"
        assert FlowEngine._resolve_field(ctx, "intent.confidence") == 0.9

    def test_metadata_dict_field(self):
        ctx = FlowContext(query="test")
        ctx.metadata["custom_key"] = "custom_value"
        assert FlowEngine._resolve_field(ctx, "metadata.custom_key") == "custom_value"

    def test_missing_field_returns_none(self):
        ctx = FlowContext(query="test")
        assert FlowEngine._resolve_field(ctx, "intent.intent") is None
        assert FlowEngine._resolve_field(ctx, "nonexistent") is None

    def test_deeply_nested_none(self):
        ctx = FlowContext(query="test")
        assert FlowEngine._resolve_field(ctx, "intent.intent.deep") is None
