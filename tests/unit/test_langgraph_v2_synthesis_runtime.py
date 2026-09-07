"""Bounded Synthesis actor coverage."""

import json
from typing import cast

import pydantic_ai.models as models
import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from app.agents.synthesis import (
    SYNTHESIS_INSTRUCTIONS,
    SYNTHESIS_MAX_TOKENS,
    SYNTHESIS_TIMEOUT_SECONDS,
    PydanticAISynthesisActor,
    create_synthesis_agent,
)
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_evidence import (
    FinancialResearchReport,
    PreparedEvidence,
    PreparedSynthesis,
    synthesize_report,
)


class _Registry:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create_agent(self, model_name: str, **kwargs: object) -> object:
        assert model_name == "synthesis"
        self.kwargs = kwargs
        return object()


def test_synthesis_factory_instructs_evidence_markers() -> None:
    registry = _Registry()

    create_synthesis_agent(cast(ModelRegistry, registry), model_name="synthesis")

    assert registry.kwargs is not None
    assert registry.kwargs["instructions"] == SYNTHESIS_INSTRUCTIONS
    assert registry.kwargs["tools"] == ()


@pytest.mark.asyncio
async def test_synthesis_returns_marked_report_with_fixed_request_limits() -> None:
    actor = PydanticAISynthesisActor(
        Agent(
            TestModel(
                call_tools=[],
                custom_output_args={"markdown_report": "Apple grew. [[E:1]]"},
            ),
            output_type=FinancialResearchReport,
            tools=(),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
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
        assert info.model_settings == {"max_tokens": SYNTHESIS_MAX_TOKENS}
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

    agent = Agent(
        FunctionModel(report_response),
        output_type=FinancialResearchReport,
        tools=(),
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )
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
async def test_synthesis_repair_uses_two_toolless_single_request_traces() -> None:
    prompts: list[str] = []

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert len(messages) == 1
        assert info.function_tools == []
        assert info.model_settings == {"max_tokens": SYNTHESIS_MAX_TOKENS}
        assert isinstance(messages[0], ModelRequest)
        assert isinstance(messages[0].parts[0], UserPromptPart)
        assert isinstance(messages[0].parts[0].content, str)
        prompts.append(messages[0].parts[0].content)
        markdown = (
            "Apple grew. [[E:1]] [[E:1]]"
            if len(prompts) == 1
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
    actor = PydanticAISynthesisActor(
        Agent(
            FunctionModel(response),
            output_type=FinancialResearchReport,
            tools=(),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )

    with capture_run_messages() as messages:
        published = await synthesize_report(actor, prepared)

    first_prompt = json.loads(prompts[0])
    repair_prompt = json.loads(prompts[1])
    assert published.answer == "Apple grew. [[E:1]]"
    assert first_prompt == prepared.model_dump(mode="json")
    assert repair_prompt == {
        "prepared": first_prompt,
        "validation_errors": ["Evidence marker is duplicated"],
    }
    assert len(messages) == 3
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].usage.requests == 1


@pytest.mark.asyncio
async def test_schema_invalid_synthesis_output_uses_the_one_repair() -> None:
    calls = 0
    repair_prompts: list[dict[str, object]] = []

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        assert len(messages) == 1
        assert info.function_tools == []
        if calls == 2:
            assert isinstance(messages[0], ModelRequest)
            assert isinstance(messages[0].parts[0], UserPromptPart)
            assert isinstance(messages[0].parts[0].content, str)
            repair_prompts.append(json.loads(messages[0].parts[0].content))
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
    actor = PydanticAISynthesisActor(
        Agent(
            FunctionModel(response),
            output_type=FinancialResearchReport,
            tools=(),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )

    published = await synthesize_report(actor, prepared)

    assert published.answer == "Apple grew. [[E:1]]"
    assert calls == 2
    assert repair_prompts[0]["validation_errors"] == ["Synthesis report is invalid"]
