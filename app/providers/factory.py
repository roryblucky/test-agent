"""Provider factory with decorator-based registry.

New provider implementations are registered with
``@register_provider("component", "provider_name")`` and automatically
discovered when the factory creates instances.

Example::

    @register_provider("retriever", "gcp")
    class GCPVertexSearchRetriever(BaseRetrieverProvider):
        ...

    provider = ProviderFactory.create("retriever", "gcp", config, cloud_cfg, pool)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.core.http_client_pool import HttpClientPool

# Global registry: (component, provider) → implementation class
_REGISTRY: dict[tuple[str, str], type] = {}


def register_provider(component: str, provider: str):
    """Class decorator that registers a provider implementation.

    Args:
        component: Component type, e.g. ``"retriever"``, ``"ranker"``.
        provider: Provider name, e.g.  ``"gcp"``, ``"azure"``.
    """

    def wrapper(cls: type) -> type:
        _REGISTRY[(component, provider)] = cls
        return cls

    return wrapper


class ProviderFactory:
    """Creates provider instances from config using the registry."""

    @staticmethod
    def create(
        component: str,
        provider: str,
        config: BaseModel,
        cloud_config: BaseModel | None,
        http_pool: HttpClientPool,
    ) -> Any:
        """Instantiate a registered provider.

        Raises:
            ValueError: If no implementation is registered for the
                *(component, provider)* combination.
        """
        key = (component, provider)
        if key not in _REGISTRY:
            available = [k[1] for k in _REGISTRY if k[0] == component]
            raise ValueError(
                f"No provider registered for ({component}, {provider}). Available {component} providers: {available}"
            )
        cls = _REGISTRY[key]
        return cls(config=config, cloud_config=cloud_config, http_pool=http_pool)

    @staticmethod
    def available(component: str | None = None) -> list[tuple[str, str]]:
        """List registered (component, provider) pairs."""
        if component:
            return [k for k in _REGISTRY if k[0] == component]
        open("")
        return list(_REGISTRY)
