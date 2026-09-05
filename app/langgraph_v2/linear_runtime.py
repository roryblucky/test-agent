"""Tenant-scoped construction for the v2 Linear Graph runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config.models import LangGraphRuntimeMode
from app.langgraph_v2.answer import AnswerActor, build_answer_actor
from app.langgraph_v2.checkpointing import (
    CheckpointStateAdapter,
    LinearCheckpointStateAdapter,
)
from app.langgraph_v2.contracts import V2QueryRequest
from app.langgraph_v2.graph import LinearGraph, build_linear_graph
from app.langgraph_v2.groundedness import (
    GroundednessActor,
    UnavailableGroundednessActor,
    build_groundedness_actor,
)
from app.langgraph_v2.output_assessments import (
    LoggingOutputAssessmentAudit,
    OutputAssessmentAudit,
)
from app.langgraph_v2.pre_moderation import ModerationProvider
from app.langgraph_v2.provider_adapters import (
    MissingModeration,
    MissingRanker,
    MissingRetriever,
    V2ProviderBundle,
    adapt_tenant_providers,
)
from app.langgraph_v2.question_refinement import (
    QuestionRefinementActor,
    build_question_refinement_actor,
)
from app.langgraph_v2.reranking import Ranker
from app.langgraph_v2.retrieval import Retriever
from app.langgraph_v2.stream import RequestOwnedGraph


@dataclass(frozen=True)
class LinearGraphOverrides:
    """Optional Linear implementations used instead of Tenant defaults."""

    refinement_actor: QuestionRefinementActor | None
    retriever: Retriever | None
    ranker: Ranker | None
    moderation_provider: ModerationProvider | None
    answer_actor: AnswerActor | None
    groundedness_actor: GroundednessActor | None
    history_token_budget: int
    output_assessment_audit: OutputAssessmentAudit | None


@dataclass(frozen=True)
class _LinearGraphDependencies:
    """Resolved dependencies for one request-owned Linear Graph."""

    checkpointer: BaseCheckpointSaver[Any]
    tenant_id: str
    history_token_budget: int
    output_assessment_audit: OutputAssessmentAudit
    refinement_actor: QuestionRefinementActor | None
    retriever: Retriever | None
    ranker: Ranker | None
    moderation_provider: ModerationProvider | None
    answer_actor: AnswerActor | None
    groundedness_actor: GroundednessActor | None


@dataclass(frozen=True)
class LinearGraphRuntimeAdapter:
    """Linear Graph implementation of the shared Query runtime interface."""

    dependencies: _LinearGraphDependencies
    graph_override: RequestOwnedGraph | None = None

    @property
    def runtime_mode(self) -> LangGraphRuntimeMode:
        """Identify this adapter as the Linear runtime."""
        return LangGraphRuntimeMode.LINEAR

    @property
    def checkpoint_state_adapter(self) -> CheckpointStateAdapter:
        """Return the validator for every persisted Linear state channel."""
        return LinearCheckpointStateAdapter()

    def build_graph(self, *, request_id: str) -> RequestOwnedGraph:
        """Build the request-owned Linear Graph for one request."""
        return self.graph_override or _build_graph(
            self.dependencies,
            request_id=request_id,
        )

    def initial_state_fields(
        self,
        *,
        payload: V2QueryRequest,
    ) -> Mapping[str, Any]:
        """Return no fields beyond the shared Query state."""
        del payload
        return {}


def build_linear_runtime(
    app: FastAPI,
    *,
    tenant_id: str,
    checkpointer: BaseCheckpointSaver[Any],
    overrides: LinearGraphOverrides,
    graph_override: RequestOwnedGraph | None,
) -> LinearGraphRuntimeAdapter:
    """Resolve Tenant dependencies and construct one Linear runtime adapter."""
    provider_bundle = _resolve_provider_bundle(app, tenant_id)
    retriever, ranker, moderation = _resolve_phase_providers(
        app,
        overrides=overrides,
        provider_bundle=provider_bundle,
    )
    dependencies = _LinearGraphDependencies(
        checkpointer=checkpointer,
        tenant_id=tenant_id,
        history_token_budget=overrides.history_token_budget,
        output_assessment_audit=_resolve_output_assessment_audit(
            app, overrides.output_assessment_audit
        ),
        refinement_actor=_resolve_refinement_actor(
            app, tenant_id, overrides.refinement_actor
        ),
        retriever=retriever,
        ranker=ranker,
        moderation_provider=moderation,
        answer_actor=_resolve_answer_actor(app, tenant_id, overrides.answer_actor),
        groundedness_actor=_resolve_groundedness_actor_safely(
            app,
            tenant_id,
            overrides.groundedness_actor,
            provider_bundle,
        ),
    )
    return LinearGraphRuntimeAdapter(
        dependencies=dependencies,
        graph_override=graph_override,
    )


def _resolve_refinement_actor(
    app: FastAPI,
    tenant_id: str,
    injected: QuestionRefinementActor | None,
) -> QuestionRefinementActor | None:
    if injected is not None:
        return injected
    configured = getattr(app.state, "langgraph_v2_refinement_actor", None)
    if configured is not None:
        return configured
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        return None
    return build_question_refinement_actor(manager.get_model_registry(tenant_id))


def _resolve_provider_bundle(app: FastAPI, tenant_id: str) -> V2ProviderBundle | None:
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_providers"):
        return None
    ranker_top_n: int | None = None
    if hasattr(manager, "get_tenant_config"):
        tenant_config = manager.get_tenant_config(tenant_id)
        if tenant_config.ranking_config is not None:
            ranker_top_n = tenant_config.ranking_config.top_n
    return adapt_tenant_providers(
        manager.get_providers(tenant_id),
        ranker_top_n=ranker_top_n,
    )


def _resolve_answer_actor(
    app: FastAPI,
    tenant_id: str,
    injected: AnswerActor | None,
) -> AnswerActor | None:
    if injected is not None:
        return injected
    configured = getattr(app.state, "langgraph_v2_answer_actor", None)
    if configured is not None:
        return configured
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        return None
    return build_answer_actor(manager.get_model_registry(tenant_id))


def _resolve_groundedness_actor(
    app: FastAPI,
    tenant_id: str,
    injected: GroundednessActor | None,
    provider_bundle: V2ProviderBundle | None,
) -> GroundednessActor | None:
    if injected is not None:
        return injected
    configured = getattr(app.state, "langgraph_v2_groundedness_actor", None)
    if configured is not None:
        return configured
    if provider_bundle is not None and provider_bundle.groundedness is not None:
        return provider_bundle.groundedness
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        return None
    return build_groundedness_actor(manager.get_model_registry(tenant_id))


def _resolve_groundedness_actor_safely(
    app: FastAPI,
    tenant_id: str,
    injected: GroundednessActor | None,
    provider_bundle: V2ProviderBundle | None,
) -> GroundednessActor | None:
    try:
        return _resolve_groundedness_actor(app, tenant_id, injected, provider_bundle)
    except Exception as exc:
        return UnavailableGroundednessActor(exc)


def _resolve_output_assessment_audit(
    app: FastAPI,
    injected: OutputAssessmentAudit | None,
) -> OutputAssessmentAudit:
    if injected is not None:
        return injected
    configured = getattr(app.state, "langgraph_v2_output_assessment_audit", None)
    return configured or LoggingOutputAssessmentAudit()


def _resolve_phase_providers(
    app: FastAPI,
    *,
    overrides: LinearGraphOverrides,
    provider_bundle: V2ProviderBundle | None,
) -> tuple[Retriever | None, Ranker | None, ModerationProvider | None]:
    retriever = overrides.retriever or getattr(
        app.state, "langgraph_v2_retriever", None
    )
    ranker = overrides.ranker or getattr(app.state, "langgraph_v2_ranker", None)
    moderation = overrides.moderation_provider or getattr(
        app.state, "langgraph_v2_moderation_provider", None
    )
    if provider_bundle is not None:
        retriever = retriever or provider_bundle.retriever or MissingRetriever()
        ranker = ranker or provider_bundle.ranker or MissingRanker()
        moderation = moderation or provider_bundle.moderation or MissingModeration()
    return retriever, ranker, moderation


def _build_graph(
    dependencies: _LinearGraphDependencies,
    *,
    request_id: str,
) -> LinearGraph:
    return build_linear_graph(
        dependencies.checkpointer,
        tenant_id=dependencies.tenant_id,
        current_request_id=request_id,
        history_token_budget=dependencies.history_token_budget,
        output_assessment_audit=dependencies.output_assessment_audit,
        refinement_actor=dependencies.refinement_actor,
        retriever=dependencies.retriever,
        ranker=dependencies.ranker,
        moderation_provider=dependencies.moderation_provider,
        answer_actor=dependencies.answer_actor,
        groundedness_actor=dependencies.groundedness_actor,
        checkpoint_state_adapter=LinearCheckpointStateAdapter(),
    )
