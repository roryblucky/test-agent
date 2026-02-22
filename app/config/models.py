"""Pydantic models for config.json tenant configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM Config
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    provider: str  # "azure", "gcp", extensible
    model_name: str = Field(alias="modelName")
    temperature: float | None = None
    top_p: float | None = Field(None, alias="topP")
    max_tokens: int = Field(8000, alias="maxTokens")
    # Provider-specific optional fields for thinking/reasoning control
    # GCP (Gemini): thinkingLevel maps to google_thinking_config.thinking_level
    thinking_level: str | None = Field(None, alias="thinkingLevel")
    # GCP (Gemini): thinkingBudget maps to google_thinking_config.thinking_budget
    thinking_budget: int | None = Field(None, alias="thinkingBudget")
    # Azure/OpenAI: thinkingEffort maps to openai_reasoning_effort
    thinking_effort: str | None = Field(None, alias="thinkingEffort")
    # Not all models support thinking — these fields are silently ignored
    # when the provider doesn't support them.

    model_config = {"populate_by_name": True}


class LLMConfig(BaseModel):
    """Named map of model configurations.

    Keys are purpose names like "fast", "pro", "intent".
    """

    models: dict[str, ModelConfig]

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Non-LLM Provider Configs
# ---------------------------------------------------------------------------


class RetrieverConfig(BaseModel):
    """Retriever component configuration."""

    provider: str  # "gcp", "azure", extensible
    top_k: int = Field(10, alias="topK")
    search_type: str = Field("semantic", alias="searchType")

    model_config = {"populate_by_name": True}


class RankingConfig(BaseModel):
    """Ranking component configuration."""

    provider: str
    top_n: int = Field(5, alias="topN")
    model: str = "semantic-ranker-512"

    model_config = {"populate_by_name": True}


class ModerationConfig(BaseModel):
    """Content moderation configuration."""

    provider: str
    categories: list[str] = Field(
        default_factory=lambda: ["hate", "violence", "self_harm", "sexual"]
    )
    threshold: str = "medium"

    model_config = {"populate_by_name": True}


class GroundednessConfig(BaseModel):
    """Groundedness checking configuration."""

    provider: str
    threshold: float = 0.7

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Flow Config
# ---------------------------------------------------------------------------


class FlowStepType(StrEnum):
    """Available flow step module types.

    Each type is a *module* — the ``mode`` field on :class:`FlowStep`
    selects the specific action within that module.
    """

    MODERATION = "moderation"
    LLM = "llm"
    RETRIEVER = "retriever"
    RANKING = "ranking"
    GROUNDEDNESS = "groundedness"
    ANALYSIS = "analysis"
    MEMORY = "memory"


class FlowStep(BaseModel):
    """A single step in the flow pipeline.

    - ``type``     — the module to execute (moderation, llm, retriever, …)
    - ``mode``     — action variant within that module
      (e.g. ``"pre"``/``"post"`` for moderation,
      ``"refine_question"``/``"intent"``/``"answer"`` for llm)
    - ``model``    — named model from ``llmConfig.models`` (llm steps only)
    - ``settings`` — per-step overrides for model parameters
      (temperature, maxTokens, topP, …).  Merged over the base
      ``ModelConfig`` defaults at runtime.
    """

    type: FlowStepType
    mode: str | None = None
    model: str | None = None
    settings: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}


class MCPServerConfig(BaseModel):
    """External MCP tool-server connection.

    Provide *either* ``url`` (HTTP/SSE) or ``command`` (stdio).
    """

    name: str
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None

    model_config = {"populate_by_name": True}


class UsageLimitConfig(BaseModel):
    """Token / call budget for agent orchestration.

    Maps directly to pydantic-ai's ``UsageLimits``.
    """

    request_limit: int = Field(50, alias="requestLimit")
    tool_calls_limit: int | None = Field(None, alias="toolCallsLimit")
    input_tokens_limit: int | None = Field(None, alias="inputTokensLimit")
    output_tokens_limit: int | None = Field(None, alias="outputTokensLimit")
    total_tokens_limit: int | None = Field(None, alias="totalTokensLimit")

    model_config = {"populate_by_name": True}


class FlowConfig(BaseModel):
    """Pipeline orchestration configuration.

    ``mode`` controls the execution strategy:
    - "simple": linear pipeline using ``steps``
    - "agent": complex multi-agent orchestration using ``agent_graph``
    """

    mode: str = "simple"
    steps: list[FlowStep] = Field(default_factory=list)
    agent_graph: str | None = Field(None, alias="agentGraph")
    usage_limits: UsageLimitConfig | None = Field(None, alias="usageLimits")
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list, alias="mcpServers")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Cloud Provider Configs
# ---------------------------------------------------------------------------


class AzureConfig(BaseModel):
    """Azure cloud configuration."""

    tenant_id: str = Field(alias="tenantId")
    client_id: str = Field(alias="clientId")
    client_secret: str = Field(alias="clientSecret")
    openai_endpoint: str = Field(alias="openAIEndpoint")
    content_safety_endpoint: str = Field(alias="contentSafetyEndpoint")
    ai_language_endpoint: str = Field(alias="aiLanguageEndpoint")
    proxy_host: str | None = Field(None, alias="proxyHost")
    proxy_port: int | None = Field(None, alias="proxyPort")
    no_proxy: str | None = Field(None, alias="noProxy")

    model_config = {"populate_by_name": True}


class GCPConfig(BaseModel):
    """GCP cloud configuration."""

    project_id: str = Field(alias="projectId")
    bucket_name: str | None = Field(None, alias="bucketName")
    datastore: str | None = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Tenant Config (top-level)
# ---------------------------------------------------------------------------


class TenantConfig(BaseModel):
    """Complete configuration for a single tenant / KMS application."""

    kms_app_name: str = Field(alias="kmsAppName")
    application_id: str = Field(alias="applicationId")
    ad_groups: list[str] = Field(alias="adGroups")

    llm_config: LLMConfig = Field(alias="llmConfig")
    retriever_config: RetrieverConfig | None = Field(None, alias="retrieverConfig")
    ranking_config: RankingConfig | None = Field(None, alias="rankingConfig")
    moderation_config: ModerationConfig | None = Field(None, alias="moderationConfig")
    groundedness_config: GroundednessConfig | None = Field(
        None, alias="groundednessConfig"
    )
    flow_config: FlowConfig = Field(alias="flowConfig")

    # Cloud configs — top-level, extensible (future: aliConfig, awsConfig, etc.)
    azure_config: AzureConfig | None = Field(None, alias="azureConfig")
    gcp_config: GCPConfig | None = Field(None, alias="gcpConfig")

    model_config = {"populate_by_name": True}
