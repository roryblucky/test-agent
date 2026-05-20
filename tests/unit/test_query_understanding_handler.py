"""Unit tests for combined query understanding handling."""

from typing import Any, Self
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage

from app.agents.query_understanding import (
    DEFAULT_INSTRUCTIONS as QUERY_UNDERSTANDING_INSTRUCTIONS,
)
from app.api.schemas import QueryResponse
from app.config.models import FlowStep, FlowStepType
from app.models.domain import RefinedQuestion
from app.models.workflow import (
    IntentResult,
    QueryUnderstandingClarification,
    QueryUnderstandingClarificationQuestion,
    QueryUnderstandingOutput,
    ResolvedQuery,
)
from app.services.flow_context import FlowContext
from app.services.handlers.llm import LLMHandler


class FakeUnderstandingStream:
    """Async context manager for query understanding tests."""

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


class FakeUnderstandingAgent:
    """Fake query-understanding agent that captures prompt and kwargs."""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.last_prompt: str | None = None
        self.last_message_history: Any | None = None
        self.last_model_settings: Any | None = None
        self.run_count = 0

    def run_stream(self, prompt: str, **kwargs: Any) -> FakeUnderstandingStream:
        self.run_count += 1
        self.last_prompt = prompt
        self.last_message_history = kwargs.get("message_history")
        self.last_model_settings = kwargs.get("model_settings")
        return FakeUnderstandingStream(self.output)


def _handler(fake_agent: FakeUnderstandingAgent) -> LLMHandler:
    handler = LLMHandler(MagicMock())
    handler._agent_cache[("query_understanding", "intent")] = fake_agent
    return handler


@pytest.mark.asyncio
async def test_query_understanding_sets_resolved_query_and_intent() -> None:
    """One LLM call writes resolver and intent outputs into FlowContext."""
    fake_agent = FakeUnderstandingAgent(
        QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query="ignored",
                standalone_query="What is the latest view on Microsoft?",
            ),
            intent=IntentResult(
                intent="market_outlook",
                confidence=0.91,
            ),
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
        FlowStep(
            type=FlowStepType.LLM,
            mode="query_understanding",
            model="intent",
            settings={
                "temperature": 0,
                "intentCatalog": [
                    {
                        "intent": "market_outlook",
                        "description": "Market outlook questions.",
                    }
                ],
            },
        ),
    )

    assert result.resolved_query == ResolvedQuery(
        original_query="那微软呢？",
        standalone_query="What is the latest view on Microsoft?",
    )
    assert result.refined_query == "What is the latest view on Microsoft?"
    assert result.intent == IntentResult(
        intent="market_outlook",
        confidence=0.91,
    )
    assert fake_agent.last_message_history is None
    assert fake_agent.last_model_settings == {"temperature": 0}
    assert fake_agent.last_prompt is not None
    assert "<query_understanding_input>" in fake_agent.last_prompt
    assert "<latest_user_query>\n那微软呢？\n</latest_user_query>" in fake_agent.last_prompt
    assert "[user]\n苹果最近怎么看？" in fake_agent.last_prompt
    assert "[assistant]\n苹果需要基于最新证据判断。" in fake_agent.last_prompt
    assert '"intent": "market_outlook"' in fake_agent.last_prompt


def test_query_understanding_instructions_define_language_and_catalog_rules() -> None:
    """The combined agent prompt explains language, names, and catalog usage."""
    assert "<language_rules>" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "zh-Hans" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "zh-Hant" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "yue-Hant" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "<proper_noun_rules>" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "<platform_common_intents>" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "summarize_history" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "<intent_catalog_rules>" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "runtime <intent_catalog>" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "It may be empty" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "Do not invent intents outside these two sources" in (
        QUERY_UNDERSTANDING_INSTRUCTIONS
    )
    assert "QueryUnderstandingOutput.clarification" in QUERY_UNDERSTANDING_INSTRUCTIONS
    assert "ResolvedQuery and IntentResult do not contain user clarification fields" in (
        QUERY_UNDERSTANDING_INSTRUCTIONS
    )
    assert "question and option strings" in QUERY_UNDERSTANDING_INSTRUCTIONS


@pytest.mark.asyncio
async def test_query_understanding_default_runtime_catalog_is_empty() -> None:
    """Platform common intents are static instructions, not runtime catalog items."""
    fake_agent = FakeUnderstandingAgent(
        QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query="ignored",
                standalone_query="Summarize our previous conversation.",
            ),
            intent=IntentResult(intent="summarize_history", confidence=0.87),
        )
    )
    handler = _handler(fake_agent)
    ctx = FlowContext(query="总结一下我们刚才聊了什么")

    await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="query_understanding", model="intent"),
    )

    assert fake_agent.last_prompt is not None
    assert "<intent_catalog>\n[]\n</intent_catalog>" in fake_agent.last_prompt


@pytest.mark.asyncio
async def test_query_understanding_sanitizes_history_payloads() -> None:
    """Runtime answer payloads and system parts are not exposed as history."""
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
    fake_agent = FakeUnderstandingAgent(
        QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query="ignored",
                standalone_query="What is beta?",
            ),
            intent=IntentResult(intent="knowledge_query", confidence=0.9),
        )
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
        FlowStep(type=FlowStepType.LLM, mode="query_understanding", model="intent"),
    )

    assert fake_agent.last_prompt is not None
    assert "What is alpha?" in fake_agent.last_prompt
    assert "Visible assistant answer." in fake_agent.last_prompt
    assert "SECRET EVIDENCE PAYLOAD" not in fake_agent.last_prompt
    assert "SECRET SYSTEM PROMPT" not in fake_agent.last_prompt
    assert "SECRET INSTRUCTIONS" not in fake_agent.last_prompt


@pytest.mark.asyncio
async def test_query_understanding_query_clarification_stops_flow() -> None:
    """Top-level query-resolution clarification stops downstream flow."""
    fake_agent = FakeUnderstandingAgent(
        QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query="ignored",
                standalone_query="Clarify the referenced item.",
            ),
            intent=IntentResult(intent="market_outlook", confidence=0.55),
            clarification=QueryUnderstandingClarification(
                scope="query_resolution",
                questions=[
                    QueryUnderstandingClarificationQuestion(
                        question="你说的“这个”是指哪个资产？",
                        options=["Microsoft", "Apple"],
                    )
                ],
                reason="The referenced item is ambiguous.",
            ),
        )
    )
    handler = _handler(fake_agent)
    ctx = FlowContext(query="那这个呢？")

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="query_understanding", model="intent"),
    )

    assert result.metadata["stop_flow"] is True
    assert (
        result.metadata["stop_reason"]
        == "query_understanding_needs_query_clarification"
    )
    assert result.llm_response == "你说的“这个”是指哪个资产？"
    response = QueryResponse.from_flow_context(result)
    assert response.clarification is not None
    assert response.clarification.response == result.llm_response
    assert response.clarification.quick_questions is not None
    assert response.clarification.quick_questions[0].options == ["Microsoft", "Apple"]


@pytest.mark.asyncio
async def test_query_understanding_intent_selection_clarification_stops_flow() -> None:
    """Top-level intent-selection clarification stops downstream flow."""
    fake_agent = FakeUnderstandingAgent(
        QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query="ignored",
                standalone_query="Analyze Microsoft.",
            ),
            intent=IntentResult(
                intent="market_outlook",
                confidence=0.52,
            ),
            clarification=QueryUnderstandingClarification(
                scope="intent_selection",
                questions=[
                    QueryUnderstandingClarificationQuestion(
                        question="你想看市场观点、目标价，还是新闻摘要？",
                        options=["市场观点", "目标价", "新闻摘要"],
                    )
                ],
            ),
        )
    )
    handler = _handler(fake_agent)
    ctx = FlowContext(query="分析微软")

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="query_understanding", model="intent"),
    )

    assert result.metadata["stop_flow"] is True
    assert (
        result.metadata["stop_reason"]
        == "query_understanding_needs_intent_clarification"
    )
    assert result.llm_response == "你想看市场观点、目标价，还是新闻摘要？"
    response = QueryResponse.from_flow_context(result)
    assert response.clarification is not None
    assert response.clarification.response == result.llm_response
    assert response.clarification.quick_questions is not None
    assert response.clarification.quick_questions[0].options == [
        "市场观点",
        "目标价",
        "新闻摘要",
    ]


@pytest.mark.asyncio
async def test_refine_question_compatibility_still_accepts_legacy_output() -> None:
    """The legacy refine_question mode still accepts RefinedQuestion output."""
    fake_agent = FakeUnderstandingAgent(
        RefinedQuestion(refined_query="What is alpha?", keywords=["alpha"])
    )
    handler = LLMHandler(MagicMock())
    handler._agent_cache[("refine_question", "fast")] = fake_agent
    ctx = FlowContext(query="alpha?")

    result = await handler.handle(
        ctx,
        FlowStep(type=FlowStepType.LLM, mode="refine_question", model="fast"),
    )

    assert result.resolved_query == ResolvedQuery(
        original_query="alpha?",
        standalone_query="What is alpha?",
    )
