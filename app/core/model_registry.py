"""Model registry — thin wrapper around pydantic-ai Model creation.

Creates and holds pydantic-ai :class:`Model` instances keyed by purpose name
(e.g. ``"fast"``, ``"pro"``, ``"intent"``).  Does **not** wrap
:class:`pydantic_ai.Agent` — callers create their own Agents with the
appropriate ``system_prompt``, ``output_type``, and ``tools``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from app.config.models import LLMConfig, ModelConfig
from app.core.http_client_pool import HttpClientPool


@dataclass(frozen=True)
class RegisteredModel:
    """A named model with its pydantic-ai ``Model`` instance and defaults."""

    name: str
    model: Model
    settings: ModelSettings


class ModelRegistry:
    """Manages named pydantic-ai ``Model`` objects built from tenant config.

    Usage::

        registry = ModelRegistry(llm_config, cloud_configs, http_pool)

        # Option A: Get raw model + settings
        reg = registry.get_model("fast")
        agent = Agent(reg.model, model_settings=reg.settings, ...)

        # Option B: Convenience factory
        agent = registry.create_agent("pro",
            instructions="Answer based on documents...",
            output_type=str,
        )
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        cloud_configs: dict[str, Any],
        http_pool: HttpClientPool,
    ) -> None:
        self._models: dict[str, RegisteredModel] = {}
        for name, model_cfg in llm_config.models.items():
            self._models[name] = self._create_model(
                name, model_cfg, cloud_configs, http_pool
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model(self, name: str) -> RegisteredModel:
        """Return a registered model by purpose name.

        Raises:
            KeyError: If *name* is not found.
        """
        if name not in self._models:
            available = ", ".join(sorted(self._models))
            raise KeyError(f"Model '{name}' not found. Available: [{available}]")
        return self._models[name]

    def create_agent(self, model_name: str, **agent_kwargs: Any) -> Agent:
        """Create a :class:`pydantic_ai.Agent` pre-configured with a named model.

        All extra *agent_kwargs* (``instructions``, ``output_type``,
        ``tools``, etc.) are forwarded directly to the Agent constructor.
        """
        registered = self.get_model(model_name)
        return Agent(
            registered.model,
            model_settings=registered.settings,
            **agent_kwargs,
        )

    @property
    def available_models(self) -> list[str]:
        """List all available model names."""
        return sorted(self._models)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _create_model(
        name: str,
        cfg: ModelConfig,
        cloud_configs: dict[str, Any],
        http_pool: HttpClientPool,
    ) -> RegisteredModel:
        match cfg.provider:
            case "azure":
                model = _build_azure_model(cfg, cloud_configs, http_pool)
                settings = _build_azure_settings(cfg)
            case "gcp":
                model = _build_gcp_model(cfg, cloud_configs, http_pool)
                settings = _build_gcp_settings(cfg)
            case _:
                raise ValueError(
                    f"Unknown LLM provider '{cfg.provider}' for model '{name}'. Supported: azure, gcp"
                )

        return RegisteredModel(name=name, model=model, settings=settings)


# -----------------------------------------------------------------------
# Provider-specific model builders
# -----------------------------------------------------------------------


def _build_azure_model(
    cfg: ModelConfig,
    cloud_configs: dict[str, Any],
    http_pool: HttpClientPool,
) -> Model:
    """Create a pydantic-ai ``OpenAIChatModel`` with ``AzureProvider``."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.azure import AzureProvider

    azure_cfg = cloud_configs.get("azure")
    if azure_cfg is None:
        raise ValueError("Azure LLM model requested but no azureConfig provided")

    provider = AzureProvider(
        azure_endpoint=azure_cfg.openai_endpoint,
        api_key=azure_cfg.client_secret,
        http_client=http_pool.get("azure"),
    )
    return OpenAIChatModel(cfg.model_name, provider=provider)


def _build_gcp_model(
    cfg: ModelConfig,
    cloud_configs: dict[str, Any],
    http_pool: HttpClientPool,
) -> Model:
    """Create a pydantic-ai ``GoogleModel``."""
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    gcp_cfg = cloud_configs.get("gcp")
    project_id = gcp_cfg.project_id if gcp_cfg else None

    provider = GoogleProvider(
        project=project_id,
        http_client=http_pool.get("gcp"),
    )
    return GoogleModel(cfg.model_name, provider=provider)


# -----------------------------------------------------------------------
# Provider-specific settings builders (with thinking/reasoning support)
# -----------------------------------------------------------------------


def _build_gcp_settings(cfg: ModelConfig) -> ModelSettings:
    """Build settings for a GCP (Gemini) model.

    If ``thinkingLevel`` or ``thinkingBudget`` is configured,
    produces a :class:`GoogleModelSettings` with ``google_thinking_config``.
    Otherwise, returns a plain :class:`ModelSettings`.
    """
    base: dict[str, Any] = {"max_tokens": cfg.max_tokens}
    if cfg.temperature is not None:
        base["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        base["top_p"] = cfg.top_p

    if cfg.thinking_level or cfg.thinking_budget:
        from google.genai.types import ThinkingConfigDict
        from pydantic_ai.models.google import GoogleModelSettings

        thinking_config: dict[str, Any] = {"include_thoughts": True}

        if cfg.thinking_level:
            thinking_config["thinking_level"] = cfg.thinking_level.upper()

        if cfg.thinking_budget:
            thinking_config["thinking_budget"] = cfg.thinking_budget

        return GoogleModelSettings(
            **base,
            google_thinking_config=ThinkingConfigDict(**thinking_config),
        )

    return ModelSettings(**base)


def _build_azure_settings(cfg: ModelConfig) -> ModelSettings:
    """Build settings for an Azure (OpenAI) model.

    If ``thinkingEffort`` is configured, produces an
    :class:`OpenAIChatModelSettings` with ``openai_reasoning_effort``.
    Otherwise, returns a plain :class:`ModelSettings`.

    Valid effort values: ``none``, ``minimal``, ``low``, ``medium``,
    ``high``, ``xhigh``.
    """
    base: dict[str, Any] = {"max_tokens": cfg.max_tokens}
    if cfg.temperature is not None:
        base["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        base["top_p"] = cfg.top_p

    if cfg.thinking_effort:
        from openai.types import ReasoningEffort
        from pydantic_ai.models.openai import OpenAIChatModelSettings

        return OpenAIChatModelSettings(
            **base,
            openai_reasoning_effort=cast(ReasoningEffort, cfg.thinking_effort.lower()),
        )

    return ModelSettings(**base)
