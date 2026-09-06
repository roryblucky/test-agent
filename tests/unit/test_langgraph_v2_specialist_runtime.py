"""No-Tool Specialist PydanticAI actor coverage."""

from datetime import date
from typing import cast

import pydantic_ai.models as models
import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from app.agents.specialist import (
    SPECIALIST_MAX_TOKENS,
    SPECIALIST_TIMEOUT_SECONDS,
    PydanticAISpecialistActor,
    create_bound_specialist_actor,
    create_specialist_agent,
)
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import SpecialistFindingDraft, SpecialistTaskInput
from app.langgraph_v2.agent_evidence import EvidenceEnvelope, bind_evidence_tool


@pytest.fixture(autouse=True)
def disable_real_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep all Specialist actor coverage local to TestModel/FunctionModel."""
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


class _Registry:
    def __init__(self) -> None:
        self.name: str | None = None
        self.kwargs: dict[str, object] | None = None

    def create_agent(self, name: str, **kwargs: object) -> object:
        self.name = name
        self.kwargs = kwargs
        return object()


def test_specialist_factory_disables_tools_and_builtin_retries() -> None:
    registry = _Registry()

    create_specialist_agent(cast(ModelRegistry, registry), model_name="specialist")

    assert registry.name == "specialist"
    assert registry.kwargs is not None
    assert registry.kwargs == {
        "output_type": SpecialistFindingDraft,
        "instructions": registry.kwargs["instructions"],
        "tools": (),
        "retries": 0,
        "tool_retries": 0,
        "output_retries": 0,
        "end_strategy": "early",
    }


def test_bound_specialist_factory_uses_exact_frozen_tools() -> None:
    registry = _Registry()

    async def read_evidence(source: str, query: str) -> object:
        del source, query
        return object()

    actor = create_bound_specialist_actor(
        cast(ModelRegistry, registry),
        model_name="specialist",
        tools=(read_evidence,),
        returned_evidence=[],
    )

    assert isinstance(actor, PydanticAISpecialistActor)
    assert registry.kwargs is not None
    assert registry.kwargs["tools"] == (read_evidence,)


@pytest.mark.asyncio
async def test_no_tool_specialist_returns_one_structured_finding() -> None:
    model = TestModel(
        call_tools=[],
        custom_output_args={"summary": "No-tool finding", "evidence_ids": []},
    )
    agent = Agent(
        model,
        output_type=SpecialistFindingDraft,
        tools=(),
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )
    actor = PydanticAISpecialistActor(agent)

    with capture_run_messages() as messages:
        finding = await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess market outlook.")
        )

    assert finding.finding == SpecialistFindingDraft(summary="No-tool finding")
    assert len(messages) == 3
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].usage.requests == 1
    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.function_tools == []
    assert SPECIALIST_TIMEOUT_SECONDS == 60
    assert SPECIALIST_MAX_TOKENS == 2000


@pytest.mark.asyncio
async def test_specialist_function_model_has_one_request_and_structured_trace() -> None:
    captures: list[tuple[list[ModelMessage], AgentInfo]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captures.append((messages, info))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"summary": "No-tool finding", "evidence_ids": []},
                )
            ]
        )

    agent = Agent(
        FunctionModel(model),
        output_type=SpecialistFindingDraft,
        tools=(),
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )
    actor = PydanticAISpecialistActor(agent)
    with capture_run_messages() as messages:
        finding = await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess market outlook.")
        )

    assert finding.finding == SpecialistFindingDraft(summary="No-tool finding")
    assert len(captures) == 1
    assert captures[0][1].function_tools == []
    assert captures[0][1].model_settings == {"max_tokens": SPECIALIST_MAX_TOKENS}
    assert len(messages) == 3
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].usage.requests == 1

    with capture_run_messages() as result_messages:
        result = await agent.run("Assess market outlook.")
    assert result.output == finding.finding
    assert len(result.new_messages()) == 3
    assert result.usage().requests == 1
    assert len(result_messages) == 3


@pytest.mark.asyncio
async def test_specialist_accepts_tool_metadata_only_after_terminal_finding() -> None:
    returned_evidence: list[EvidenceEnvelope] = []
    calls = 0

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        assert (source, query) == ("filing", "Apple revenue")
        return EvidenceEnvelope(
            id="evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            source="filing",
            source_url="https://example.test/filing",
            title="Annual filing",
            body="BODY-SENTINEL",
            excerpt="Apple revenue grew.",
            as_of_date=date(2026, 9, 6),
        )

    tool = bind_evidence_tool(
        provider,
        allowed_sources=frozenset({"filing"}),
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
        returned_evidence=returned_evidence,
    )

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert [tool.name for tool in info.function_tools] == ["read_evidence"]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_evidence",
                        args={"source": "filing", "query": "Apple revenue"},
                    )
                ]
            )
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert len(tool_returns) == 1
        assert tool_returns[0].content == {
            "evidence_id": "evidence-1",
            "excerpt": "Apple revenue grew.",
        }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "summary": "Apple revenue grew.",
                        "evidence_ids": ["evidence-1"],
                    },
                )
            ]
        )

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(tool,),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        ),
        returned_evidence=returned_evidence,
    )

    attempt = await actor.run(
        SpecialistTaskInput(task_id="task-1", objective="Assess Apple revenue.")
    )

    assert calls == 2
    assert attempt.finding.evidence_ids == ("evidence-1",)
    assert attempt.evidence[0].id == "evidence-1"
    assert returned_evidence == []


@pytest.mark.asyncio
async def test_specialist_discards_tool_metadata_when_model_fails() -> None:
    returned_evidence: list[EvidenceEnvelope] = []

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        return EvidenceEnvelope(
            id="evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            source=source,
            source_url="https://example.test/filing",
            title="Annual filing",
            body="BODY-SENTINEL",
            excerpt=query,
            as_of_date=date(2026, 9, 6),
        )

    tool = bind_evidence_tool(
        provider,
        allowed_sources=frozenset({"filing"}),
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
        returned_evidence=returned_evidence,
    )
    calls = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_evidence",
                        args={"source": "filing", "query": "excerpt"},
                    )
                ]
            )
        raise RuntimeError("model failed")

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(tool,),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        ),
        returned_evidence=returned_evidence,
    )

    with pytest.raises(Exception):
        await actor.run(SpecialistTaskInput(task_id="task-1", objective="Assess."))

    assert actor.returned_evidence == []
