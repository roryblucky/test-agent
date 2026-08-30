"""Pydantic models for config.json tenant configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import AliasChoices, BaseModel, Field

# ---------------------------------------------------------------------------
# LLM Config
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    provider: str  # "azure", "gcp", extensible
    model_name: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("model_name", "modelName"),
            serialization_alias="modelName",
        ),
    ]
    temperature: float | None = None
    top_p: Annotated[
        float | None,
        Field(
            validation_alias=AliasChoices("top_p", "topP"), serialization_alias="topP"
        ),
    ] = None
    max_tokens: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("max_tokens", "maxTokens"),
            serialization_alias="maxTokens",
        ),
    ] = 8000
    # Provider-specific optional fields for thinking/reasoning control
    # GCP (Gemini): thinkingLevel maps to google_thinking_config.thinking_level
    thinking_level: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("thinking_level", "thinkingLevel"),
            serialization_alias="thinkingLevel",
        ),
    ] = None
    # GCP (Gemini): thinkingBudget maps to google_thinking_config.thinking_budget
    thinking_budget: Annotated[
        int | None,
        Field(
            validation_alias=AliasChoices("thinking_budget", "thinkingBudget"),
            serialization_alias="thinkingBudget",
        ),
    ] = None
    # Azure/OpenAI: thinkingEffort maps to openai_reasoning_effort
    thinking_effort: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("thinking_effort", "thinkingEffort"),
            serialization_alias="thinkingEffort",
        ),
    ] = None
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
    top_k: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("top_k", "topK"), serialization_alias="topK"
        ),
    ] = 10
    search_type: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("search_type", "searchType"),
            serialization_alias="searchType",
        ),
    ] = "semantic"

    model_config = {"populate_by_name": True}


class RetrieverConfig(BaseModel):
    """Retriever component configuration. Supports multiple concurrent sources."""

    sources: list[RetrieverSourceConfig] = Field(
        default_factory=list[RetrieverSourceConfig]
    )

    model_config = {"populate_by_name": True}


class RankingConfig(BaseModel):
    """Ranking component configuration."""

    provider: str
    top_n: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("top_n", "topN"), serialization_alias="topN"
        ),
    ] = 5
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
    prompt_pack: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("prompt_pack", "promptPack"),
            serialization_alias="promptPack",
        ),
    ] = None
    locale: str = "zh-CN"
    allow_model_common_knowledge: Annotated[
        bool,
        Field(
            validation_alias=AliasChoices(
                "allow_model_common_knowledge", "allowModelCommonKnowledge"
            ),
            serialization_alias="allowModelCommonKnowledge",
        ),
    ] = False

    model_config = {"populate_by_name": True}


class TenantOutputConfig(BaseModel):
    """Tenant-level output formatting and compliance wording."""

    default_format: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("default_format", "defaultFormat"),
            serialization_alias="defaultFormat",
        ),
    ] = "markdown"
    disclaimer: str | None = None
    forbidden_phrases: Annotated[
        list[str],
        Field(
            validation_alias=AliasChoices("forbidden_phrases", "forbiddenPhrases"),
            serialization_alias="forbiddenPhrases",
        ),
    ] = Field(default_factory=list)
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

    enabled_tools: Annotated[
        list[str],
        Field(
            validation_alias=AliasChoices("enabled_tools", "enabledTools"),
            serialization_alias="enabledTools",
        ),
    ] = Field(default_factory=list)
    max_tool_calls: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("max_tool_calls", "maxToolCalls"),
            serialization_alias="maxToolCalls",
        ),
    ] = 8
    require_confirmation_for_high_risk: Annotated[
        bool,
        Field(
            validation_alias=AliasChoices(
                "require_confirmation_for_high_risk", "requireConfirmationForHighRisk"
            ),
            serialization_alias="requireConfirmationForHighRisk",
        ),
    ] = True

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Step Routing (conditional branching)
# ---------------------------------------------------------------------------


class StepRoutingAction(StrEnum):
    """What to do when a routing rule matches."""

    CONTINUE = "continue"  # proceed to next step (default)
    ABORT = "abort"  # stop the pipeline, optionally set a response
    GOTO = "goto"  # jump to a named step
    SKIP_TO = "skip_to"  # skip forward to a step type:mode


class StepRoutingRule(BaseModel):
    """A single routing rule evaluated after a step completes.

    Rules are evaluated against fields on ``FlowContext``.  The first
    matching rule wins; if none match, the pipeline continues normally.

    Examples::

        # Abort on out-of-scope intent
        {"match_field": "intent.intent", "match_value": "out_of_scope",
         "action": "abort", "response": "This question is out of scope."}

        # Abort when a handler has produced a clarification request
        {"match_field": "metadata.needs_clarification", "match_value": true,
         "action": "abort",
         "response_from_field": "clarification_request.response"}
    """

    match_field: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("match_field", "matchField"),
            serialization_alias="matchField",
        ),
    ]
    match_value: Annotated[
        Any,
        Field(
            validation_alias=AliasChoices("match_value", "matchValue"),
            serialization_alias="matchValue",
        ),
    ]
    action: StepRoutingAction = StepRoutingAction.CONTINUE
    response: str | None = None
    response_from_field: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("response_from_field", "responseFromField"),
            serialization_alias="responseFromField",
        ),
    ] = None
    target_step: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("target_step", "targetStep"),
            serialization_alias="targetStep",
        ),
    ] = None

    model_config = {"populate_by_name": True}


class AgentMCPServerConfig(BaseModel):
    """Agent specific MCP server configuration."""

    transports: str = "sse"
    url: str = ""
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    allowed_tools: Annotated[
        list[str] | None,
        Field(
            validation_alias=AliasChoices("allowed_tools", "allowedTools"),
            serialization_alias="allowedTools",
        ),
    ] = None

    model_config = {"populate_by_name": True}


class AgentConfig(BaseModel):
    """Configuration for Pydantic AI config-driven orchestrator."""

    llm_type: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("llm_type", "llmType"),
            serialization_alias="llmType",
        ),
    ] = "fast"
    enable_todo: Annotated[
        bool,
        Field(
            validation_alias=AliasChoices("enable_todo", "enableTodo"),
            serialization_alias="enableTodo",
        ),
    ] = False
    mcp_servers: Annotated[
        dict[str, AgentMCPServerConfig],
        Field(
            validation_alias=AliasChoices("mcp_servers", "mcpServers"),
            serialization_alias="mcpServers",
        ),
    ] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    built_in_tools: Annotated[
        list[str],
        Field(
            validation_alias=AliasChoices("built_in_tools", "buildInTools"),
            serialization_alias="buildInTools",
        ),
    ] = Field(default_factory=list)
    prompt_type: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("prompt_type", "promptType"),
            serialization_alias="promptType",
        ),
    ] = None

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
    agent_config: Annotated[
        AgentConfig | None,
        Field(
            validation_alias=AliasChoices("agent_config", "agentConfig"),
            serialization_alias="agentConfig",
        ),
    ] = None
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

    request_limit: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("request_limit", "requestLimit"),
            serialization_alias="requestLimit",
        ),
    ] = 50
    tool_calls_limit: Annotated[
        int | None,
        Field(
            validation_alias=AliasChoices("tool_calls_limit", "toolCallsLimit"),
            serialization_alias="toolCallsLimit",
        ),
    ] = None
    input_tokens_limit: Annotated[
        int | None,
        Field(
            validation_alias=AliasChoices("input_tokens_limit", "inputTokensLimit"),
            serialization_alias="inputTokensLimit",
        ),
    ] = None
    output_tokens_limit: Annotated[
        int | None,
        Field(
            validation_alias=AliasChoices("output_tokens_limit", "outputTokensLimit"),
            serialization_alias="outputTokensLimit",
        ),
    ] = None
    total_tokens_limit: Annotated[
        int | None,
        Field(
            validation_alias=AliasChoices("total_tokens_limit", "totalTokensLimit"),
            serialization_alias="totalTokensLimit",
        ),
    ] = None

    model_config = {"populate_by_name": True}


class FlowConfig(BaseModel):
    """Pipeline orchestration configuration."""

    steps: list[FlowStep] = Field(default_factory=list[FlowStep])
    usage_limits: Annotated[
        UsageLimitConfig | None,
        Field(
            validation_alias=AliasChoices("usage_limits", "usageLimits"),
            serialization_alias="usageLimits",
        ),
    ] = None
    mcp_servers: Annotated[
        list[MCPServerConfig],
        Field(
            validation_alias=AliasChoices("mcp_servers", "mcpServers"),
            serialization_alias="mcpServers",
        ),
    ] = Field(default_factory=list[MCPServerConfig])

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Cloud Provider Configs
# ---------------------------------------------------------------------------


class AzureConfig(BaseModel):
    """Azure cloud configuration."""

    tenant_id: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("tenant_id", "tenantId"),
            serialization_alias="tenantId",
        ),
    ]
    client_id: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("client_id", "clientId"),
            serialization_alias="clientId",
        ),
    ]
    client_secret: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("client_secret", "clientSecret"),
            serialization_alias="clientSecret",
        ),
    ]
    openai_endpoint: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("openai_endpoint", "openAIEndpoint"),
            serialization_alias="openAIEndpoint",
        ),
    ]
    content_safety_endpoint: Annotated[
        str,
        Field(
            validation_alias=AliasChoices(
                "content_safety_endpoint", "contentSafetyEndpoint"
            ),
            serialization_alias="contentSafetyEndpoint",
        ),
    ]
    ai_language_endpoint: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("ai_language_endpoint", "aiLanguageEndpoint"),
            serialization_alias="aiLanguageEndpoint",
        ),
    ]
    proxy_host: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("proxy_host", "proxyHost"),
            serialization_alias="proxyHost",
        ),
    ] = None
    proxy_port: Annotated[
        int | None,
        Field(
            validation_alias=AliasChoices("proxy_port", "proxyPort"),
            serialization_alias="proxyPort",
        ),
    ] = None
    no_proxy: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("no_proxy", "noProxy"),
            serialization_alias="noProxy",
        ),
    ] = None

    model_config = {"populate_by_name": True}


class GCPConfig(BaseModel):
    """GCP cloud configuration."""

    project_id: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("project_id", "projectId"),
            serialization_alias="projectId",
        ),
    ]
    bucket_name: Annotated[
        str | None,
        Field(
            validation_alias=AliasChoices("bucket_name", "bucketName"),
            serialization_alias="bucketName",
        ),
    ] = None
    datastore: str | None = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Rate Limiting Config
# ---------------------------------------------------------------------------


class EndpointRateLimitPolicy(BaseModel):
    """Rate limit policy for a specific endpoint type."""

    requests_per_minute: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("requests_per_minute", "requestsPerMinute"),
            serialization_alias="requestsPerMinute",
        ),
    ] = 60
    requests_per_day: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("requests_per_day", "requestsPerDay"),
            serialization_alias="requestsPerDay",
        ),
    ] = 10000
    concurrent_requests: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("concurrent_requests", "concurrentRequests"),
            serialization_alias="concurrentRequests",
        ),
    ] = 10

    model_config = {"populate_by_name": True}


class RateLimitConfig(BaseModel):
    """Per-tenant rate limiting configuration.

    Separate policies for ``/query`` (non-streaming) and
    ``/query/stream`` (SSE streaming) endpoints, plus a shared
    monthly token budget.
    """

    query_policy: Annotated[
        EndpointRateLimitPolicy,
        Field(
            validation_alias=AliasChoices("query_policy", "queryPolicy"),
            serialization_alias="queryPolicy",
        ),
    ] = Field(
        default_factory=lambda: EndpointRateLimitPolicy(
            requests_per_minute=60, requests_per_day=10000, concurrent_requests=10
        )
    )
    stream_policy: Annotated[
        EndpointRateLimitPolicy,
        Field(
            validation_alias=AliasChoices("stream_policy", "streamPolicy"),
            serialization_alias="streamPolicy",
        ),
    ] = Field(
        default_factory=lambda: EndpointRateLimitPolicy(
            requests_per_minute=30, requests_per_day=5000, concurrent_requests=5
        )
    )
    tokens_per_month: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("tokens_per_month", "tokensPerMonth"),
            serialization_alias="tokensPerMonth",
        ),
    ] = 1000000

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Audit Config
# ---------------------------------------------------------------------------


class AuditConfig(BaseModel):
    """Audit logging configuration for compliance."""

    enabled: bool = True
    bigquery_dataset: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("bigquery_dataset", "bigqueryDataset"),
            serialization_alias="bigqueryDataset",
        ),
    ] = "audit_logs"
    bigquery_table: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("bigquery_table", "bigqueryTable"),
            serialization_alias="bigqueryTable",
        ),
    ] = "kms_audit"
    retention_years: Annotated[
        int,
        Field(
            validation_alias=AliasChoices("retention_years", "retentionYears"),
            serialization_alias="retentionYears",
        ),
    ] = 7

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Tenant Config (top-level)
# ---------------------------------------------------------------------------


class TenantConfig(BaseModel):
    """Complete configuration for a single tenant / KMS application."""

    kms_app_name: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("kms_app_name", "kmsAppName"),
            serialization_alias="kmsAppName",
        ),
    ]
    application_id: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("application_id", "applicationId"),
            serialization_alias="applicationId",
        ),
    ]
    ad_groups: Annotated[
        list[str],
        Field(
            validation_alias=AliasChoices("ad_groups", "adGroups"),
            serialization_alias="adGroups",
        ),
    ]

    llm_config: Annotated[
        LLMConfig,
        Field(
            validation_alias=AliasChoices("llm_config", "llmConfig"),
            serialization_alias="llmConfig",
        ),
    ]
    retriever_config: Annotated[
        RetrieverConfig | None,
        Field(
            validation_alias=AliasChoices("retriever_config", "retrieverConfig"),
            serialization_alias="retrieverConfig",
        ),
    ] = None
    ranking_config: Annotated[
        RankingConfig | None,
        Field(
            validation_alias=AliasChoices("ranking_config", "rankingConfig"),
            serialization_alias="rankingConfig",
        ),
    ] = None
    moderation_config: Annotated[
        ModerationConfig | None,
        Field(
            validation_alias=AliasChoices("moderation_config", "moderationConfig"),
            serialization_alias="moderationConfig",
        ),
    ] = None
    groundedness_config: Annotated[
        GroundednessConfig | None,
        Field(
            validation_alias=AliasChoices("groundedness_config", "groundednessConfig"),
            serialization_alias="groundednessConfig",
        ),
    ] = None
    flow_config: Annotated[
        FlowConfig,
        Field(
            validation_alias=AliasChoices("flow_config", "flowConfig"),
            serialization_alias="flowConfig",
        ),
    ]
    rate_limit_config: Annotated[
        RateLimitConfig | None,
        Field(
            validation_alias=AliasChoices("rate_limit_config", "rateLimitConfig"),
            serialization_alias="rateLimitConfig",
        ),
    ] = None
    audit_config: Annotated[
        AuditConfig | None,
        Field(
            validation_alias=AliasChoices("audit_config", "auditConfig"),
            serialization_alias="auditConfig",
        ),
    ] = None
    tool_runtime_config: Annotated[
        ToolRuntimeConfig | None,
        Field(
            validation_alias=AliasChoices("tool_runtime_config", "toolRuntimeConfig"),
            serialization_alias="toolRuntimeConfig",
        ),
    ] = None
    domain_config: Annotated[
        DomainConfig | None,
        Field(
            validation_alias=AliasChoices("domain_config", "domainConfig"),
            serialization_alias="domainConfig",
        ),
    ] = None
    output_config: Annotated[
        TenantOutputConfig | None,
        Field(
            validation_alias=AliasChoices("output_config", "outputConfig"),
            serialization_alias="outputConfig",
        ),
    ] = None

    # Cloud configs — top-level, extensible (future: aliConfig, awsConfig, etc.)
    azure_config: Annotated[
        AzureConfig | None,
        Field(
            validation_alias=AliasChoices("azure_config", "azureConfig"),
            serialization_alias="azureConfig",
        ),
    ] = None
    gcp_config: Annotated[
        GCPConfig | None,
        Field(
            validation_alias=AliasChoices("gcp_config", "gcpConfig"),
            serialization_alias="gcpConfig",
        ),
    ] = None

    model_config = {"populate_by_name": True}
