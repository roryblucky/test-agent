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


class RetrieverSourceConfig(BaseModel):
    """Configuration for a single retriever source."""

    provider: str  # "gcp", "azure", extensible
    top_k: int = Field(10, alias="topK")
    search_type: str = Field("semantic", alias="searchType")

    model_config = {"populate_by_name": True}


class RetrieverConfig(BaseModel):
    """Retriever component configuration. Supports multiple concurrent sources."""

    sources: list[RetrieverSourceConfig] = Field(default_factory=list)

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
# Domain / Output Config
# ---------------------------------------------------------------------------


class DomainConfig(BaseModel):
    """Domain-level configuration for prompt layering and policy.

    Controls domain-specific behaviours such as prompt pack selection,
    locale, and whether model common knowledge is allowed.
    """

    name: str
    prompt_pack: str | None = Field(None, alias="promptPack")
    locale: str = "zh-CN"
    allow_model_common_knowledge: bool = Field(
        False, alias="allowModelCommonKnowledge"
    )

    model_config = {"populate_by_name": True}


class TenantOutputConfig(BaseModel):
    """Tenant-level output formatting and compliance wording."""

    default_format: str = Field("markdown", alias="defaultFormat")
    disclaimer: str | None = None
    forbidden_phrases: list[str] = Field(
        default_factory=list, alias="forbiddenPhrases"
    )
    contract: dict[str, Any] = Field(default_factory=dict)

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
    AGENT = "agent"
    AGGREGATION = "aggregation"


class ToolRuntimeConfig(BaseModel):
    """Tenant-level runtime policy for built-in tools."""

    enabled_tools: list[str] = Field(default_factory=list, alias="enabledTools")
    max_tool_calls: int = Field(8, alias="maxToolCalls")
    require_confirmation_for_high_risk: bool = Field(
        True, alias="requireConfirmationForHighRisk"
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Step Routing (conditional branching)
# ---------------------------------------------------------------------------


class StepRoutingAction(StrEnum):
    """What to do when a routing rule matches."""

    CONTINUE = "continue"  # proceed to next step (default)
    ABORT = "abort"        # stop the pipeline, optionally set a response
    GOTO = "goto"          # jump to a named step
    SKIP_TO = "skip_to"    # skip forward to a step type:mode


class StepRoutingRule(BaseModel):
    """A single routing rule evaluated after a step completes.

    Rules are evaluated against fields on ``FlowContext``.  The first
    matching rule wins; if none match, the pipeline continues normally.

    Examples::

        # Abort on out-of-scope intent
        {"match_field": "intent.intent", "match_value": "out_of_scope",
         "action": "abort", "response": "This question is out of scope."}

        # Skip to answer step when clarification is needed
        {"match_field": "intent.needs_clarification", "match_value": true,
         "action": "abort",
         "response_from_field": "intent.clarification_question"}
    """

    match_field: str = Field(alias="matchField")
    match_value: Any = Field(alias="matchValue")
    action: StepRoutingAction = StepRoutingAction.CONTINUE
    response: str | None = None
    response_from_field: str | None = Field(None, alias="responseFromField")
    target_step: str | None = Field(None, alias="targetStep")

    model_config = {"populate_by_name": True}


class AgentMCPServerConfig(BaseModel):
    """Agent specific MCP server configuration."""

    transports: str = "sse"
    url: str = ""
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    allowed_tools: list[str] | None = Field(None, alias="allowedTools")

    model_config = {"populate_by_name": True}


class AgentConfig(BaseModel):
    """Configuration for Pydantic AI config-driven orchestrator."""

    llm_type: str = Field("fast", alias="llmType")
    enable_todo: bool = Field(False, alias="enableTodo")
    mcp_servers: dict[str, AgentMCPServerConfig] = Field(
        default_factory=dict, alias="mcpServers"
    )
    skills: list[str] = Field(default_factory=list)
    built_in_tools: list[str] = Field(default_factory=list, alias="buildInTools")
    prompt_type: str | None = Field(None, alias="promptType")

    model_config = {"populate_by_name": True}


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
    - ``routing``  — optional conditional routing rules evaluated
      after step execution.  If omitted, the pipeline continues
      to the next step unconditionally.
    - ``name``     — optional step name used as a ``goto`` / ``skip_to``
      target by routing rules on other steps.
    """

    type: FlowStepType
    mode: str | None = None
    model: str | None = None
    settings: dict[str, Any] | None = None
    agent_config: AgentConfig | None = Field(None, alias="agentConfig")
    routing: list[StepRoutingRule] | None = None
    name: str | None = None

    model_config = {"populate_by_name": True}

    @property
    def step_label(self) -> str:
        """Human-readable step label: ``type:mode`` or just ``type``."""
        return f"{self.type.value}:{self.mode}" if self.mode else self.type.value


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
    """Pipeline orchestration configuration."""

    steps: list[FlowStep] = Field(default_factory=list)
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
# Rate Limiting Config
# ---------------------------------------------------------------------------


class EndpointRateLimitPolicy(BaseModel):
    """Rate limit policy for a specific endpoint type."""

    requests_per_minute: int = Field(60, alias="requestsPerMinute")
    requests_per_day: int = Field(10000, alias="requestsPerDay")
    concurrent_requests: int = Field(10, alias="concurrentRequests")

    model_config = {"populate_by_name": True}


class RateLimitConfig(BaseModel):
    """Per-tenant rate limiting configuration.

    Separate policies for ``/query`` (non-streaming) and
    ``/query/stream`` (SSE streaming) endpoints, plus a shared
    monthly token budget.
    """

    query_policy: EndpointRateLimitPolicy = Field(
        default_factory=EndpointRateLimitPolicy, alias="queryPolicy"
    )
    stream_policy: EndpointRateLimitPolicy = Field(
        default_factory=lambda: EndpointRateLimitPolicy(
            requests_per_minute=30,
            requests_per_day=5000,
            concurrent_requests=5,
        ),
        alias="streamPolicy",
    )
    tokens_per_month: int = Field(1_000_000, alias="tokensPerMonth")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Audit Config
# ---------------------------------------------------------------------------


class AuditConfig(BaseModel):
    """Audit logging configuration for compliance."""

    enabled: bool = True
    bigquery_dataset: str = Field("audit_logs", alias="bigqueryDataset")
    bigquery_table: str = Field("kms_audit", alias="bigqueryTable")
    retention_years: int = Field(7, alias="retentionYears")

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
    rate_limit_config: RateLimitConfig | None = Field(
        None, alias="rateLimitConfig"
    )
    audit_config: AuditConfig | None = Field(None, alias="auditConfig")
    tool_runtime_config: ToolRuntimeConfig | None = Field(
        None, alias="toolRuntimeConfig"
    )
    domain_config: DomainConfig | None = Field(None, alias="domainConfig")
    output_config: TenantOutputConfig | None = Field(None, alias="outputConfig")

    # Cloud configs — top-level, extensible (future: aliConfig, awsConfig, etc.)
    azure_config: AzureConfig | None = Field(None, alias="azureConfig")
    gcp_config: GCPConfig | None = Field(None, alias="gcpConfig")

    model_config = {"populate_by_name": True}
