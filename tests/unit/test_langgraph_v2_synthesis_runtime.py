"""Bounded Synthesis actor coverage."""

from datetime import date
from typing import Any, cast

import pydantic_ai.models as models
import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import ContentFilterError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from app.agents.synthesis import (
    SYNTHESIS_INSTRUCTIONS,
    SYNTHESIS_MAX_TOKENS,
    SYNTHESIS_OUTPUT_RETRIES,
    SYNTHESIS_TIMEOUT_SECONDS,
    PydanticAISynthesisActor,
    create_synthesis_agent,
)
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_evidence import (
    FinancialResearchReport,
    PreparedCalculation,
    PreparedEvidence,
    PreparedSynthesis,
    synthesize_report,
)


class _Registry:
    def __init__(self, model: models.Model | None = None) -> None:
        self.kwargs: dict[str, Any] | None = None
        self.model = model if model is not None else TestModel()

    def create_agent(self, model_name: str, **kwargs: Any) -> object:
        assert model_name == "synthesis"
        self.kwargs = kwargs
        return Agent(self.model, **kwargs)


def _prepared() -> PreparedSynthesis:
    return PreparedSynthesis(
        standalone_query="Apple",
        intent="market",
        evidence=(
            PreparedEvidence(
                id="evidence-1",
                source="filing",
                source_url="https://example.test",
                title="Filing",
                excerpt="Apple grew.",
            ),
        ),
    )


def _synthesis_agent(
    model: models.Model,
) -> Agent[PreparedSynthesis, FinancialResearchReport]:
    return create_synthesis_agent(
        cast(ModelRegistry, _Registry(model)), model_name="synthesis"
    )


def test_synthesis_factory_instructs_evidence_markers() -> None:
    registry = _Registry()

    agent = create_synthesis_agent(
        cast(ModelRegistry, registry), model_name="synthesis"
    )

    assert registry.kwargs is not None
    assert registry.kwargs["instructions"] == SYNTHESIS_INSTRUCTIONS
    assert registry.kwargs["tools"] == ()
    assert registry.kwargs["tool_retries"] == 0
    assert registry.kwargs["deps_type"] is PreparedSynthesis
    assert registry.kwargs["output_retries"] == SYNTHESIS_OUTPUT_RETRIES
    assert registry.kwargs["end_strategy"] == "early"
    assert isinstance(agent, Agent)


def test_synthesis_actor_construction_does_not_configure_agent() -> None:
    agent = cast(Agent[PreparedSynthesis, FinancialResearchReport], object())

    first = PydanticAISynthesisActor(agent)
    second = PydanticAISynthesisActor(agent)

    assert first.agent is agent
    assert second.agent is agent


@pytest.mark.asyncio
async def test_synthesis_returns_marked_report_with_fixed_request_limits() -> None:
    actor = PydanticAISynthesisActor(
        _synthesis_agent(
            TestModel(
                call_tools=[],
                custom_output_args={"markdown_report": "Apple grew. [[E:1]]"},
            )
        )
    )
    result = await actor.synthesize(
        PreparedSynthesis(
            standalone_query="Apple",
            intent="market",
            evidence=(
                PreparedEvidence(
                    id="evidence-1",
                    source="filing",
                    source_url="https://example.test",
                    title="Filing",
                    excerpt="Apple grew.",
                ),
            ),
        )
    )
    assert result.markdown_report == "Apple grew. [[E:1]]"
    assert SYNTHESIS_TIMEOUT_SECONDS == 120
    assert SYNTHESIS_MAX_TOKENS == 4000


@pytest.mark.asyncio
async def test_synthesis_function_model_captures_one_toolless_overrideable_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    calls: list[str] = []

    def report_response(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        calls.append("default")
        assert len(messages) == 1
        assert info.function_tools == []
        assert info.model_settings == {
            "max_tokens": SYNTHESIS_MAX_TOKENS,
            "timeout": SYNTHESIS_TIMEOUT_SECONDS,
        }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"markdown_report": "Apple grew. [[E:1]]"},
                )
            ]
        )

    def override_response(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        calls.append("override")
        assert len(messages) == 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"markdown_report": "Apple grew. [[E:1]]"},
                )
            ]
        )

    agent = _synthesis_agent(FunctionModel(report_response))
    actor = PydanticAISynthesisActor(agent)
    prepared = PreparedSynthesis(
        standalone_query="Apple",
        intent="market",
        evidence=(
            PreparedEvidence(
                id="evidence-1",
                source="filing",
                source_url="https://example.test",
                title="Filing",
                excerpt="Apple grew.",
            ),
        ),
    )

    with capture_run_messages() as messages:
        assert await actor.synthesize(prepared) == FinancialResearchReport(
            markdown_report="Apple grew. [[E:1]]"
        )
    with agent.override(model=FunctionModel(override_response)):
        assert await actor.synthesize(prepared) == FinancialResearchReport(
            markdown_report="Apple grew. [[E:1]]"
        )

    assert calls == ["default", "override"]
    assert len(messages) == 3
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].usage.requests == 1


@pytest.mark.asyncio
async def test_synthesis_marker_retry_stays_in_one_native_run() -> None:
    calls = 0
    retry_parts: list[RetryPromptPart] = []

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        assert len(messages) == (1 if calls == 1 else 3)
        assert info.function_tools == []
        assert info.model_settings == {
            "max_tokens": SYNTHESIS_MAX_TOKENS,
            "timeout": SYNTHESIS_TIMEOUT_SECONDS,
        }
        retry_parts.extend(
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        )
        markdown = (
            "Apple grew. [[E:1]] [[E:1]]"
            if calls == 1
            else "Apple grew. [[E:1]]"
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"markdown_report": markdown},
                )
            ]
        )

    prepared = PreparedSynthesis(
        standalone_query="Apple",
        intent="market",
        evidence=(
            PreparedEvidence(
                id="evidence-1",
                source="filing",
                source_url="https://example.test",
                title="Filing",
                excerpt="Apple grew.",
            ),
        ),
    )
    actor = PydanticAISynthesisActor(_synthesis_agent(FunctionModel(response)))

    with capture_run_messages() as messages:
        published = await synthesize_report(actor, prepared)

    assert published.answer == "Apple grew. [[E:1]]"
    assert calls == 2
    assert len(retry_parts) == 1
    assert retry_parts[0].content == "Evidence marker is duplicated"
    assert len(messages) == 5
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].usage.requests == 1


@pytest.mark.asyncio
async def test_schema_invalid_synthesis_output_uses_one_native_retry() -> None:
    calls = 0
    retry_parts: list[RetryPromptPart] = []

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        assert len(messages) == (1 if calls == 1 else 3)
        assert info.function_tools == []
        retry_parts.extend(
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        )
        args = (
            {"unexpected": "invalid output"}
            if calls == 1
            else {"markdown_report": "Apple grew. [[E:1]]"}
        )
        return ModelResponse(
            parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)]
        )

    prepared = PreparedSynthesis(
        standalone_query="Apple",
        intent="market",
        evidence=(
            PreparedEvidence(
                id="evidence-1",
                source="filing",
                source_url="https://example.test",
                title="Filing",
                excerpt="Apple grew.",
            ),
        ),
    )
    actor = PydanticAISynthesisActor(_synthesis_agent(FunctionModel(response)))

    published = await synthesize_report(actor, prepared)

    assert published.answer == "Apple grew. [[E:1]]"
    assert calls == 2
    assert len(retry_parts) == 1
    assert "Field required" in str(retry_parts[0].content)


@pytest.mark.asyncio
async def test_synthesis_second_marker_rejection_fails_without_publication() -> None:
    calls = 0

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        del messages
        calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"markdown_report": "Apple grew."},
                )
            ]
        )

    actor = PydanticAISynthesisActor(_synthesis_agent(FunctionModel(response)))
    prepared = _prepared()

    with pytest.raises(UnexpectedModelBehavior, match="maximum output retries"):
        await synthesize_report(actor, prepared)

    assert calls == 2


@pytest.mark.asyncio
async def test_synthesis_propagates_non_output_model_failures() -> None:
    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        raise ContentFilterError("blocked")

    actor = PydanticAISynthesisActor(_synthesis_agent(FunctionModel(response)))
    prepared = _prepared()

    with pytest.raises(ContentFilterError, match="blocked"):
        await actor.synthesize(prepared)


@pytest.mark.asyncio
async def test_synthesis_projection_invariant_fails_before_model_request() -> None:
    calls = 0

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        del messages, info
        calls += 1
        raise AssertionError("model must not be called")

    actor = PydanticAISynthesisActor(_synthesis_agent(FunctionModel(response)))
    prepared = _prepared().model_copy(
        update={
            "calculations": (
                PreparedCalculation(
                    alias="C:1",
                    method="period_return",
                    unit="percent",
                    currency="USD",
                    period_start=date(2026, 1, 1),
                    period_end=date(2026, 1, 2),
                    as_of_date=date(2026, 1, 2),
                    assumptions=("close",),
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="projection is invalid"):
        await actor.synthesize(prepared)

    assert calls == 0
