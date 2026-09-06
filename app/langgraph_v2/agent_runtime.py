"""Tenant-scoped construction for the clarification-first Agent runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic_ai import Agent

from app.agents.query_understanding import create_query_understanding_agent
from app.config.models import LangGraphRuntimeMode
from app.langgraph_v2.agent_graph import (
    QueryUnderstandingActor,
    build_agent_graph,
)
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
        """Build the request-owned clarification graph."""
        del request_id
        return build_agent_graph(
            self.checkpointer,
            query_understanding_actor=self.query_understanding_actor,
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
    moderation_provider: ModerationProvider | None = None,
) -> AgentGraphRuntimeAdapter:
    """Build one Agent runtime from trusted Tenant configuration and dependencies."""
    actor = query_understanding_actor or _resolve_query_understanding_actor(
        app, request_context.tenant_id
    )
    moderation = moderation_provider or getattr(
        app.state, "langgraph_v2_moderation_provider", None
    )
    return AgentGraphRuntimeAdapter(
        checkpointer=checkpointer,
        query_understanding_actor=actor,
        moderation_provider=moderation or MockModerationProvider(),
    )


def _resolve_query_understanding_actor(
    app: FastAPI,
    tenant_id: str,
) -> QueryUnderstandingActor:
    configured = getattr(app.state, "langgraph_v2_query_understanding_actor", None)
    if configured is not None:
        return configured
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        raise RuntimeError("Agent Query Understanding actor is not configured")
    return PydanticAIQueryUnderstandingActor(
        create_query_understanding_agent(manager.get_model_registry(tenant_id))
    )
