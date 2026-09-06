"""Coordinator actor construction coverage."""

from typing import Any, cast

import pytest
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from app.agents.coordinator import (
    COORDINATOR_MAX_TOKENS,
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
from app.langgraph_v2.agent_graph import CoordinatorInput, Finish
from app.langgraph_v2.agent_runtime import build_agent_runtime
from app.langgraph_v2.agent_scope import SpecialistDescriptor
from app.langgraph_v2.authorization import TrustedRequestContext


class _Registry:
    def __init__(self) -> None:
        self.name: str | None = None
        self.kwargs: dict[str, Any] | None = None

    def create_agent(self, name: str, **kwargs: Any) -> object:
        self.name = name
        self.kwargs = kwargs
        return object()


def test_coordinator_factory_disables_tools_and_builtin_retries() -> None:
    registry = _Registry()

    create_coordinator_agent(cast(ModelRegistry, registry), model_name="coordinator")

    assert registry.name == "coordinator"
    assert registry.kwargs is not None
    assert registry.kwargs == {
        "output_type": Finish,
        "instructions": registry.kwargs["instructions"],
        "tools": (),
        "retries": 0,
        "tool_retries": 0,
        "output_retries": 0,
        "end_strategy": "early",
    }


@pytest.mark.asyncio
async def test_coordinator_runs_real_pydantic_actor_with_fixed_limits() -> None:
    actor = PydanticAICoordinatorActor(
        Agent(
            TestModel(),
            output_type=Finish,
            tools=(),
            retries=0,
            tool_retries=0,
            output_retries=0,
        )
    )

    decision = await actor.decide(
        CoordinatorInput(
            standalone_query="Apple outlook",
            intent="market_outlook",
            specialist_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
        )
    )

    assert decision == Finish(kind="finish")
    assert COORDINATOR_TIMEOUT_SECONDS == 60
    assert COORDINATOR_MAX_TOKENS == 1500


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
        request_context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
        checkpointer=InMemorySaver(),
        query_understanding_actor=cast(Any, object()),
        coordinator_actor=cast(Any, object()),
    )

    assert runtime.intent_policies["market_outlook"].specialist_descriptors == (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
