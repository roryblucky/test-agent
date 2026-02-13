"""Tenant manager — initialises and caches providers per tenant.

Reads the list of :class:`TenantConfig` objects, creates
:class:`ModelRegistry` and non-LLM provider instances for each,
and exposes a ``get_flow_engine`` / ``get_orchestrator`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.orchestrator import AgentOrchestrator
from app.config.models import TenantConfig
from app.core.http_client_pool import HttpClientPool
from app.core.model_registry import ModelRegistry
from app.providers.base import (
    BaseGroundednessProvider,
    BaseModerationProvider,
    BaseRankerProvider,
    BaseRetrieverProvider,
)
from app.providers.factory import ProviderFactory
from app.services.exceptions import TenantNotFoundError
from app.services.flow_engine import FlowEngine


@dataclass
class TenantProviders:
    """Holds all instantiated non-LLM providers for a tenant.

    All fields are optional — a tenant only needs the providers
    that its ``flowConfig`` steps actually reference.
    """

    retriever: BaseRetrieverProvider | None = None
    ranker: BaseRankerProvider | None = None
    moderation: BaseModerationProvider | None = None
    groundedness: BaseGroundednessProvider | None = None


class TenantManager:
    """Manages per-tenant configuration, models, and providers.

    Initialised once at application startup and used throughout the
    application lifetime (stored on ``app.state``).
    """

    def __init__(
        self,
        configs: list[TenantConfig],
        http_pool: HttpClientPool,
    ) -> None:
        self._tenants: dict[str, TenantConfig] = {}
        self._registries: dict[str, ModelRegistry] = {}
        self._providers: dict[str, TenantProviders] = {}

        for cfg in configs:
            self._tenants[cfg.application_id] = cfg
            cloud_configs = self._collect_cloud_configs(cfg)

            # Import concrete providers so @register_provider decorators fire
            self._ensure_providers_imported()

            self._registries[cfg.application_id] = ModelRegistry(
                cfg.llm_config, cloud_configs, http_pool
            )
            self._providers[cfg.application_id] = self._init_providers(
                cfg, cloud_configs, http_pool
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_flow_engine(self, app_id: str) -> FlowEngine | AgentOrchestrator:
        """Get the appropriate flow executor for a tenant.

        Returns a :class:`FlowEngine` for ``"simple"`` mode configs,
        or an :class:`AgentOrchestrator` for ``"agent"`` mode.
        """
        cfg = self._resolve_tenant(app_id)
        registry = self._registries[app_id]
        providers = self._providers[app_id]

        if cfg.flow_config.mode == "agent":
            graph_name = cfg.flow_config.agent_graph or "rag_with_intent_branching"
            return AgentOrchestrator(registry, providers, graph_name=graph_name)
        return FlowEngine(cfg, registry, providers)

    def get_tenant_config(self, app_id: str) -> TenantConfig:
        return self._resolve_tenant(app_id)

    def validate_ad_group(self, app_id: str, user_groups: list[str]) -> bool:
        """Check whether *user_groups* overlap with the tenant's ``adGroups``."""
        cfg = self._resolve_tenant(app_id)
        tenant_groups = {g.strip().lower() for g in cfg.ad_groups}
        user_groups_normalised = {g.strip().lower() for g in user_groups}
        return bool(tenant_groups & user_groups_normalised)

    @property
    def tenant_ids(self) -> list[str]:
        return list(self._tenants)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_tenant(self, app_id: str) -> TenantConfig:
        if app_id not in self._tenants:
            raise TenantNotFoundError(app_id)
        return self._tenants[app_id]

    @staticmethod
    def _collect_cloud_configs(cfg: TenantConfig) -> dict[str, Any]:
        """Gather cloud-provider configs into a flat dict keyed by provider name."""
        configs: dict[str, Any] = {}
        if cfg.azure_config:
            configs["azure"] = cfg.azure_config
        if cfg.gcp_config:
            configs["gcp"] = cfg.gcp_config
        # Future: cfg.ali_config, cfg.aws_config, etc.
        return configs

    @staticmethod
    def _init_providers(
        cfg: TenantConfig,
        cloud_configs: dict[str, Any],
        http_pool: HttpClientPool,
    ) -> TenantProviders:
        """Create non-LLM provider instances using the ProviderFactory.

        Only creates providers for which the tenant has a config section.
        Unconfigured providers remain ``None``.
        """
        providers = TenantProviders()

        if cfg.retriever_config:
            providers.retriever = ProviderFactory.create(
                "retriever",
                cfg.retriever_config.provider,
                cfg.retriever_config,
                cloud_configs.get(cfg.retriever_config.provider),
                http_pool,
            )
        if cfg.ranking_config:
            providers.ranker = ProviderFactory.create(
                "ranker",
                cfg.ranking_config.provider,
                cfg.ranking_config,
                cloud_configs.get(cfg.ranking_config.provider),
                http_pool,
            )
        if cfg.moderation_config:
            providers.moderation = ProviderFactory.create(
                "moderation",
                cfg.moderation_config.provider,
                cfg.moderation_config,
                cloud_configs.get(cfg.moderation_config.provider),
                http_pool,
            )
        if cfg.groundedness_config:
            providers.groundedness = ProviderFactory.create(
                "groundedness",
                cfg.groundedness_config.provider,
                cfg.groundedness_config,
                cloud_configs.get(cfg.groundedness_config.provider),
                http_pool,
            )

        return providers

    @staticmethod
    def _ensure_providers_imported() -> None:
        """Import concrete providers to trigger ``@register_provider``."""
        import app.providers.groundedness.gcp_groundedness  # noqa: F401
        import app.providers.moderation.azure_content_safety  # noqa: F401
        import app.providers.ranker.gcp_ranker  # noqa: F401
        import app.providers.retriever.gcp_vertex_search  # noqa: F401
