"""Tenant-scoped construction for the clarification-first Agent runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic_ai import Agent

from app.agents.coordinator import PydanticAICoordinatorActor, create_coordinator_agent
from app.agents.query_understanding import (
    DEFAULT_INSTRUCTIONS,
    create_query_understanding_agent,
)
from app.config.models import AgentResearchConfig, LangGraphRuntimeMode
from app.langgraph_v2.agent_batch import SpecialistRegistry
from app.langgraph_v2.agent_graph import (
    CoordinatorActor,
    CoordinatorInput,
    Finish,
    QueryUnderstandingActor,
    build_agent_graph,
)
from app.langgraph_v2.agent_scope import AgentIntentPolicy, SpecialistDescriptor
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import (
    AgentCheckpointStateAdapter,
    CheckpointStateAdapter,
)
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.conversation_context import (
    ConversationExchange,
    to_model_message_history,
)
from app.langgraph_v2.pre_moderation import MockModerationProvider, ModerationProvider
from app.langgraph_v2.stream import RequestOwnedGraph
from app.models.workflow import QueryUnderstandingOutput


@dataclass(frozen=True)
class PydanticAIQueryUnderstandingActor:
    """PydanticAI boundary that alone receives prior Conversation pairs."""

    agent: Agent[None, QueryUnderstandingOutput]

    async def understand(
        self,
        query: str,
        history: Sequence[ConversationExchange],
    ) -> QueryUnderstandingOutput:
        """Run the actor with only projected complete Conversation pairs."""
        result = await self.agent.run(
            query,
            message_history=to_model_message_history(history),
        )
        return result.output


@dataclass(frozen=True)
class AgentGraphRuntimeAdapter:
    """Agent-mode implementation of the shared Query runtime interface."""

    checkpointer: BaseCheckpointSaver[Any]
    query_understanding_actor: QueryUnderstandingActor
    coordinator_actor: CoordinatorActor
    specialist_registry: SpecialistRegistry
    intent_policies: Mapping[str, AgentIntentPolicy]
    moderation_provider: ModerationProvider

    @property
    def runtime_mode(self) -> LangGraphRuntimeMode:
        """Identify this adapter as the Agent runtime."""
        return LangGraphRuntimeMode.AGENT

    @property
    def checkpoint_state_adapter(self) -> CheckpointStateAdapter:
        """Return the validator for Agent-owned checkpoint state."""
        return AgentCheckpointStateAdapter()

    def build_graph(self, *, request_id: str) -> RequestOwnedGraph:
        """Build the request-owned Agent graph."""
        del request_id
        return build_agent_graph(
            self.checkpointer,
            query_understanding_actor=self.query_understanding_actor,
            coordinator_actor=self.coordinator_actor,
            specialist_registry=self.specialist_registry,
            intent_policies=self.intent_policies,
            moderation_provider=self.moderation_provider,
            checkpoint_state_adapter=AgentCheckpointStateAdapter(),
        )

    def initial_state_fields(
        self,
        *,
        payload: V2QueryRequest,
    ) -> Mapping[str, Any]:
        """Return no fields beyond the shared Query state."""
        del payload
        return {}


def build_agent_runtime(
    app: FastAPI,
    *,
    request_context: TrustedRequestContext,
    checkpointer: BaseCheckpointSaver[Any],
    query_understanding_actor: QueryUnderstandingActor | None = None,
    coordinator_actor: CoordinatorActor | None = None,
    specialist_registry: SpecialistRegistry | None = None,
    intent_policies: Mapping[str, AgentIntentPolicy] | None = None,
    moderation_provider: ModerationProvider | None = None,
) -> AgentGraphRuntimeAdapter:
    """Build one Agent runtime from trusted Tenant configuration and dependencies."""
    policies = (
        intent_policies
        if intent_policies is not None
        else _resolve_intent_policies(app, request_context.tenant_id)
    )
    actor = query_understanding_actor or _resolve_query_understanding_actor(
        app, request_context.tenant_id, policies
    )
    coordinator = coordinator_actor or _resolve_coordinator_actor(
        app, request_context.tenant_id
    )
    specialists = specialist_registry or SpecialistRegistry(
        registrations=(), tenant_eligible_ids=frozenset()
    )
    moderation = moderation_provider or getattr(
        app.state, "langgraph_v2_moderation_provider", None
    )
    return AgentGraphRuntimeAdapter(
        checkpointer=checkpointer,
        query_understanding_actor=actor,
        coordinator_actor=coordinator,
        specialist_registry=specialists,
        intent_policies=policies,
        moderation_provider=moderation or MockModerationProvider(),
    )


def _resolve_query_understanding_actor(
    app: FastAPI,
    tenant_id: str,
    intent_policies: Mapping[str, AgentIntentPolicy],
) -> QueryUnderstandingActor:
    configured = getattr(app.state, "langgraph_v2_query_understanding_actor", None)
    if configured is not None:
        return configured
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        raise RuntimeError("Agent Query Understanding actor is not configured")
    config = _agent_research_config(app, tenant_id)
    if config is None:
        raise RuntimeError("Agent Research config is not configured")
    intent_catalog = "\n".join(
        f"- {policy.intent}: {policy.description}"
        for policy in intent_policies.values()
    )
    return PydanticAIQueryUnderstandingActor(
        create_query_understanding_agent(
            manager.get_model_registry(tenant_id),
            model_name=config.query_understanding_model,
            instructions=f"{DEFAULT_INSTRUCTIONS}\n<intent_catalog>\n{intent_catalog}\n</intent_catalog>",
        )
    )


def _resolve_coordinator_actor(app: FastAPI, tenant_id: str) -> CoordinatorActor:
    configured = getattr(app.state, "langgraph_v2_coordinator_actor", None)
    if configured is not None:
        return configured
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        return _UnavailableCoordinatorActor()
    config = _agent_research_config(app, tenant_id)
    if config is None:
        return _UnavailableCoordinatorActor()
    return PydanticAICoordinatorActor(
        create_coordinator_agent(
            manager.get_model_registry(tenant_id), model_name=config.coordinator_model
        )
    )


def _resolve_intent_policies(
    app: FastAPI,
    tenant_id: str,
) -> Mapping[str, AgentIntentPolicy]:
    config = _agent_research_config(app, tenant_id)
    if config is None:
        return {}
    return {
        policy.intent: AgentIntentPolicy(
            intent=policy.intent,
            description=policy.description,
            specialist_descriptors=tuple(
                SpecialistDescriptor(
                    id=descriptor.id,
                    description=descriptor.description,
                )
                for descriptor in policy.specialist_descriptors
            ),
        )
        for policy in config.intents
    }


def _agent_research_config(
    app: FastAPI,
    tenant_id: str,
) -> AgentResearchConfig | None:
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_tenant_config"):
        return None
    return manager.get_tenant_config(tenant_id).agent_research_config


@dataclass(frozen=True)
class _UnavailableCoordinatorActor:
    """Delay missing Coordinator configuration until research reaches it."""

    async def decide(self, input: CoordinatorInput) -> Finish:
        del input
        raise RuntimeError("Agent Coordinator actor is not configured")
