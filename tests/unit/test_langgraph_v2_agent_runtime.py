"""Coordinator actor construction coverage."""

import json
from typing import Any, cast

import pydantic_ai.models as models
import pytest
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from app.agents.coordinator import (
    COORDINATOR_MAX_TOKENS,
    COORDINATOR_OUTPUT_RETRIES,
    COORDINATOR_TIMEOUT_SECONDS,
    PydanticAICoordinatorActor,
    create_coordinator_agent,
)
from app.config.models import (
    AgentResearchConfig,
    AgentResearchIntentConfig,
    AgentResearchSpecialistConfig,
    FlowConfig,
    LLMConfig,
    TenantConfig,
)
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import (
    SpecialistRegistration,
    SpecialistRegistry,
)
from app.langgraph_v2.agent_coordination import (
    AcceptedCoordinationDispatch,
    CoordinationRound,
    CoordinatorDecision,
    CoordinatorDecisionExhausted,
    CoordinatorInput,
    Finish,
    decide_coordination_round,
)
from app.langgraph_v2.agent_runtime import build_agent_runtime
from app.langgraph_v2.agent_scope import SpecialistDescriptor
from app.langgraph_v2.authorization import TrustedRequestContext


class _Registry:
    def __init__(self, model: models.Model | None = None) -> None:
        self.name: str | None = None
        self.kwargs: dict[str, Any] | None = None
        self.model = model if model is not None else TestModel()

    def create_agent(self, name: str, **kwargs: Any) -> object:
        self.name = name
        self.kwargs = kwargs
        return Agent(self.model, **kwargs)


def _coordinator_agent(
    model: models.Model,
) -> Agent[CoordinatorInput, CoordinatorDecision]:
    return create_coordinator_agent(
        cast(ModelRegistry, _Registry(model)), model_name="coordinator"
    )


def test_coordinator_factory_enables_only_one_output_retry() -> None:
    registry = _Registry()

    agent = create_coordinator_agent(
        cast(ModelRegistry, registry), model_name="coordinator"
    )

    assert registry.name == "coordinator"
    assert registry.kwargs is not None
    assert registry.kwargs == {
        "deps_type": CoordinatorInput,
        "output_type": CoordinatorDecision,
        "instructions": registry.kwargs["instructions"],
        "tools": (),
        "tool_retries": 0,
        "output_retries": COORDINATOR_OUTPUT_RETRIES,
        "end_strategy": "early",
    }
    assert isinstance(agent, Agent)


def test_coordinator_actor_construction_does_not_configure_agent() -> None:
    agent = cast(Agent[CoordinatorInput, CoordinatorDecision], object())

    first = PydanticAICoordinatorActor(agent)
    second = PydanticAICoordinatorActor(agent)

    assert first.agent is agent
    assert second.agent is agent


@pytest.mark.asyncio
async def test_coordinator_runs_real_pydantic_actor_with_fixed_limits() -> None:
    actor = PydanticAICoordinatorActor(_coordinator_agent(TestModel()))

    decision = await actor.decide(
        CoordinatorInput(
            standalone_query="Apple outlook",
            intent="market_outlook",
            specialist_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
            remaining_task_slots=32,
            dispatch_allowed=True,
        )
    )

    assert decision == Finish(kind="finish")
    assert COORDINATOR_TIMEOUT_SECONDS == 60
    assert COORDINATOR_MAX_TOKENS == 1500


@pytest.mark.asyncio
async def test_coordinator_function_model_captures_one_toolless_overrideable_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    calls: list[str] = []

    def finish_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append("default")
        assert len(messages) == 1
        assert info.function_tools == []
        assert info.model_settings == {
            "max_tokens": COORDINATOR_MAX_TOKENS,
            "timeout": COORDINATOR_TIMEOUT_SECONDS,
        }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name, args={"kind": "finish"}
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
                    tool_name=info.output_tools[0].name, args={"kind": "finish"}
                )
            ]
        )

    agent = _coordinator_agent(FunctionModel(finish_response))
    actor = PydanticAICoordinatorActor(agent)
    input = CoordinatorInput(
        standalone_query="Apple outlook",
        intent="market_outlook",
        specialist_descriptors=(),
        remaining_task_slots=32,
        dispatch_allowed=True,
    )

    with capture_run_messages() as messages:
        assert await actor.decide(input) == Finish(kind="finish")
    with agent.override(model=FunctionModel(override_response)):
        assert await actor.decide(input) == Finish(kind="finish")

    assert calls == ["default", "override"]
    assert len(messages) == 3
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].usage.requests == 1


@pytest.mark.asyncio
async def test_coordinator_uses_builtin_retry_with_schema_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    calls = 0
    retry_parts: list[RetryPromptPart] = []

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        prompt = next(
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        assert isinstance(prompt, str)
        assert json.loads(prompt) == {"input": input.model_dump(mode="json")}
        retry_parts.extend(
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        )
        args: dict[str, str] = {"kind": "invalid" if calls == 1 else "finish"}
        return ModelResponse(
            parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)]
        )

    input = CoordinatorInput(
        standalone_query="Apple outlook",
        intent="market_outlook",
        specialist_descriptors=(),
        remaining_task_slots=32,
        dispatch_allowed=True,
    )
    actor = PydanticAICoordinatorActor(
        _coordinator_agent(FunctionModel(response))
    )

    decision = await decide_coordination_round(
        actor,
        input,
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        registry=SpecialistRegistry(registrations=(), tenant_eligible_ids=frozenset()),
        scope_descriptors=(),
    )

    assert isinstance(decision, CoordinationRound)
    assert decision.kind == "finish"
    assert calls == 2
    assert len(retry_parts) == 1
    assert "literal_error" in str(retry_parts[0].content)


@pytest.mark.asyncio
async def test_coordinator_uses_builtin_retry_with_actionable_policy_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    calls = 0
    retry_parts: list[RetryPromptPart] = []

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        retry_parts.extend(
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        )
        context_task_ids = ["missing"] if calls == 1 else []
        dispatch_tool = next(
            tool for tool in info.output_tools if tool.name.endswith("DispatchBatch")
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=dispatch_tool.name,
                    args={
                        "kind": "dispatch",
                        "tasks": [
                            {
                                "specialist_id": "market-data",
                                "objective": "Assess the market.",
                                "context_task_ids": context_task_ids,
                            }
                        ],
                    },
                )
            ]
        )

    input = CoordinatorInput(
        standalone_query="Apple outlook",
        intent="market_outlook",
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
        remaining_task_slots=32,
        dispatch_allowed=True,
    )
    actor = PydanticAICoordinatorActor(
        _coordinator_agent(FunctionModel(response))
    )

    decision = await decide_coordination_round(
        actor,
        input,
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        registry=SpecialistRegistry(
            registrations=(SpecialistRegistration(id="market-data"),),
            tenant_eligible_ids=frozenset({"market-data"}),
        ),
        scope_descriptors=input.specialist_descriptors,
    )

    assert isinstance(decision, AcceptedCoordinationDispatch)
    assert calls == 2
    assert len(retry_parts) == 1
    assert retry_parts[0].content == (
        "Task context is not an accepted prior success"
    )


@pytest.mark.asyncio
async def test_coordinator_returns_typed_exhaustion_after_builtin_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    calls = 0

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        del messages
        calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"kind": "invalid"},
                )
            ]
        )

    input = CoordinatorInput(
        standalone_query="Apple outlook",
        intent="market_outlook",
        specialist_descriptors=(),
        remaining_task_slots=32,
        dispatch_allowed=True,
    )
    actor = PydanticAICoordinatorActor(
        _coordinator_agent(FunctionModel(response))
    )

    result = await actor.decide(input)

    assert result == CoordinatorDecisionExhausted(
        reason="Coordinator decision is invalid"
    )
    assert calls == 2


@pytest.mark.asyncio
async def test_coordinator_does_not_reclassify_non_output_model_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        raise ContentFilterError(
            f"Exceeded maximum output retries ({COORDINATOR_OUTPUT_RETRIES})"
        )

    input = CoordinatorInput(
        standalone_query="Apple outlook",
        intent="market_outlook",
        specialist_descriptors=(),
        remaining_task_slots=32,
        dispatch_allowed=True,
    )
    actor = PydanticAICoordinatorActor(
        _coordinator_agent(FunctionModel(response))
    )

    with pytest.raises(ContentFilterError, match="maximum output retries"):
        await actor.decide(input)


def test_finish_requires_its_discriminator() -> None:
    with pytest.raises(ValidationError):
        Finish.model_validate({})

    assert Finish.model_validate({"kind": "finish"}) == Finish(kind="finish")


def test_runtime_resolves_intent_policy_from_trusted_tenant_config() -> None:
    class _TenantManager:
        def get_tenant_config(self, tenant_id: str) -> TenantConfig:
            assert tenant_id == "tenant-a"
            return TenantConfig(
                kms_app_name="Agent Tenant",
                application_id="tenant-a",
                ad_groups=[],
                llm_config=LLMConfig(models={}),
                flow_config=FlowConfig(),
                agent_research_config=AgentResearchConfig(
                    intents=[
                        AgentResearchIntentConfig(
                            intent="market_outlook",
                            description="Assess market conditions.",
                            allowed_skill_names=["filing-analysis"],
                            specialist_descriptors=[
                                AgentResearchSpecialistConfig(
                                    id="market-data", description="Market data"
                                )
                            ],
                        )
                    ]
                ),
            )

    app = FastAPI()
    app.state.tenant_manager = _TenantManager()
    runtime = build_agent_runtime(
        app,
        request_context=TrustedRequestContext(
            tenant_id="tenant-a", subject_id="subject-a"
        ),
        checkpointer=InMemorySaver(),
        query_understanding_actor=cast(Any, object()),
        coordinator_actor=cast(Any, object()),
    )

    assert runtime.intent_policies["market_outlook"].specialist_descriptors == (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    assert runtime.intent_policies["market_outlook"].allowed_skill_names == frozenset(
        {"filing-analysis"}
    )
