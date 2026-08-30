"""Unit tests for query resolver runtime prompt handling."""

from typing import Any, Self, cast
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage

from app.config.models import (
    FlowConfig,
    FlowStep,
    FlowStepType,
    LLMConfig,
    TenantConfig,
)
from app.models.domain import RefinedQuestion
from app.models.workflow import ResolvedQuery
from app.services.flow_context import FlowContext
from app.services.flow_engine import FlowEngine
from app.services.handlers.llm import LLMHandler


class FakeResolverStream:
    """Async context manager for query resolver tests."""

    def __init__(self, output: Any) -> None:
        self._output = output

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get_output(self) -> Any:
        return self._output

    def usage(self) -> RunUsage:
        return RunUsage()


class FakeResolverAgent:
    """Fake resolver agent that captures runtime prompt and kwargs."""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.last_prompt: str | None = None
        self.last_message_history: Any | None = None
        self.run_count = 0

    def run_stream(self, prompt: str, **kwargs: Any) -> FakeResolverStream:
        self.run_count += 1
        self.last_prompt = prompt
        self.last_message_history = kwargs.get("message_history")
        return FakeResolverStream(self.output)


def _handler(fake_agent: FakeResolverAgent) -> LLMHandler:
    handler = LLMHandler(MagicMock())
    handler.set_agent_override(
        "refine_question", "fast", cast(Agent[Any, Any], fake_agent)
    )
    return handler


@pytest.mark.asyncio
async def test_query_resolver_uses_sanitized_history_runtime_prompt() -> None:
    """Resolver sees visible history text, not raw pydantic-ai history."""
    fake_agent = FakeResolverAgent(
        RefinedQuestion(
            refined_query="What is the latest view on Microsoft?",
            keywords=["Microsoft"],
        )
    )
    handler = _handler(fake_agent)
    ctx = FlowContext(query="那微软呢？")
    ctx.message_history = [
        ModelRequest(parts=[UserPromptPart("苹果最近怎么看？")]),
        ModelResponse(parts=[TextPart("苹果需要基于最新证据判断。")]),
    ]

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="refine_question", model="fast"),
    )

    assert result.refined_query == "What is the latest view on Microsoft?"
    assert result.resolved_query == ResolvedQuery(
        original_query="那微软呢？",
        standalone_query="What is the latest view on Microsoft?",
    )
    assert fake_agent.last_message_history is None
    assert fake_agent.last_prompt is not None
    assert "<query_resolver_input>" in fake_agent.last_prompt
    assert (
        "<latest_user_query>\n那微软呢？\n</latest_user_query>"
        in fake_agent.last_prompt
    )
    assert "[user]\n苹果最近怎么看？" in fake_agent.last_prompt
    assert "[assistant]\n苹果需要基于最新证据判断。" in fake_agent.last_prompt


@pytest.mark.asyncio
async def test_query_resolver_history_sanitizer_drops_internal_payloads() -> None:
    """Runtime answer prompts and system parts are not exposed as history."""
    runtime_answer_prompt = """
<runtime_answer_input>
<original_user_query>
What is alpha?
</original_user_query>
<reference_data>
SECRET EVIDENCE PAYLOAD
</reference_data>
</runtime_answer_input>
""".strip()
    fake_agent = FakeResolverAgent(
        RefinedQuestion(refined_query="What is beta?", keywords=["beta"])
    )
    handler = _handler(fake_agent)
    ctx = FlowContext(query="那 beta 呢？")
    ctx.message_history = [
        ModelRequest(
            parts=[
                SystemPromptPart("SECRET SYSTEM PROMPT"),
                UserPromptPart(runtime_answer_prompt),
            ],
            instructions="SECRET INSTRUCTIONS",
        ),
        ModelResponse(parts=[TextPart("Visible assistant answer.")]),
    ]

    await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="refine_question", model="fast"),
    )

    assert fake_agent.last_prompt is not None
    assert "What is alpha?" in fake_agent.last_prompt
    assert "Visible assistant answer." in fake_agent.last_prompt
    assert "SECRET EVIDENCE PAYLOAD" not in fake_agent.last_prompt
    assert "SECRET SYSTEM PROMPT" not in fake_agent.last_prompt
    assert "SECRET INSTRUCTIONS" not in fake_agent.last_prompt


@pytest.mark.asyncio
async def test_flow_engine_honors_stop_flow_after_step() -> None:
    """FlowEngine does not run later steps after a handler sets stop_flow."""

    class StopHandler:
        async def handle(self, ctx: FlowContext, step: FlowStep, /) -> FlowContext:
            ctx.metadata["stop_flow"] = True
            ctx.metadata["stop_reason"] = "test_stop"
            ctx.llm_response = "stop"
            return ctx

    class ShouldNotRunHandler:
        async def handle(self, ctx: FlowContext, step: FlowStep, /) -> FlowContext:
            raise AssertionError("downstream step should not run")

    tenant = TenantConfig(
        kms_app_name="Stop Flow App",
        application_id="stop-flow-app",
        ad_groups=[],
        llm_config=LLMConfig(models={}),
        flow_config=FlowConfig(
            steps=[
                FlowStep(type=FlowStepType.LLM, mode="refine_question"),
                FlowStep(type=FlowStepType.LLM, mode="answer"),
            ]
        ),
    )

    ctx = await FlowEngine(
        tenant,
        handlers={
            FlowStepType.LLM: StopHandler(),
            FlowStepType.RETRIEVER: ShouldNotRunHandler(),
        },
    ).execute("ambiguous query")

    assert ctx.llm_response == "stop"
    assert ctx.metadata["steps_executed"] == ["llm:refine_question"]
