# Agentic Workflow Platform 演进设计方案

本文档描述如何将当前 repo 演进为一个“配置驱动、多租户、可选 Agent 的企业 Agentic Workflow Platform”。

目标读者：

- Junior 工程师：能理解为什么这么设计，以及每一步改造要做什么。
- 其他 coding agent：能按本文的任务拆分逐步实现，不需要重新推导架构。
- 平台维护者：能判断一个新业务场景应该配置成普通 workflow、planner agent workflow，还是 open supervisor agent。

本文基于当前 repo 的实际结构设计，不要求把现有系统推翻重写。核心原则是：保留已有的 `FlowEngine`、config loading、tenant manager、model registry、provider/infra 抽象，在其上演进更通用的平台能力。

## 1. 设计结论

当前系统最有价值的抽象是：

- `FlowEngine`：按 `flowConfig.steps` 顺序执行 step。
- `config.json` + `app/config/loader.py`：配置驱动租户行为。
- `TenantManager`：按 tenant 组装模型、provider、flow engine。
- `ModelRegistry`：管理多模型和模型参数。
- provider/factory/http/audit/rate limit/telemetry：infra 层可复用。

平台的最终形态不是“所有业务都变成 Agent”，而是：

> 所有业务都通过 workflow 配置驱动；Agent 只是 workflow 中可选的一类 step。

因此平台必须同时支持：

1. 普通固定 workflow：例如现有 RAG 流程，不引入 agent。
2. Planner 型 agent workflow：例如财富洞察，agent 只负责规划和取证，最终答案由后续 LLM step 输出。
3. Supervisor 型 agent workflow：例如内部开发 open agent，一个 agent 加很多 tools 端到端完成任务。
4. 纯工具/规则 workflow：不需要 agent，也不需要复杂 LLM synthesis。

## 2. 和现有 repo 的映射

当前 config 中的 step 已经能表达很多平台能力。不要把 `pre_guard`、`query_resolver`、`intent_detector` 写死成新的顶层概念，而是映射到已有 step。

| 设计概念 | 当前 step 表达 | 说明 |
| --- | --- | --- |
| Pre Guard | `{"type": "moderation", "mode": "pre"}` | Azure moderation 已经承担输入安全检查职责。 |
| Query Resolver | `{"type": "llm", "mode": "refine_question"}` | 当前叫 refine question，后续 prompt/schema 可以升级成 resolver。 |
| Intent Detector | `{"type": "llm", "mode": "intent"}` | 当前已有 intent step，后续应租户化/领域化。 |
| Planner Agent | `{"type": "agent", "mode": "planner"}` | 新增 agent mode，负责 skill + tools + evidence。 |
| Supervisor Agent | `{"type": "agent", "mode": "supervisor"}` | 单 agent + tools 端到端处理。 |
| Retrieval | `{"type": "retriever"}` | 普通 RAG workflow 继续使用。 |
| Ranking | `{"type": "ranking"}` | 普通 RAG workflow 继续使用。 |
| Answer/Synthesis | `{"type": "llm", "mode": "answer"}` | 后续 answer prompt 应只使用 context/evidence。 |
| Post Safety | `{"type": "moderation", "mode": "post"}` | 输出内容安全检查。 |
| Groundedness | `{"type": "groundedness"}` | RAG/证据检查。 |
| Analysis | `{"type": "analysis"}` | 记录执行结果、latency、usage、tool calls。 |

## 3. `mode` 的使用原则

`mode` 不应该成为所有 step 的必填字段。它只适合“同一个 step type 下面确实有多个行为变体”的情况。

推荐保留 `mode` 的 step：

- `llm`
  - `refine_question`
  - `intent`
  - `answer`
  - 未来可加 `compliance_review`
- `moderation`
  - `pre`
  - `post`
- `agent`
  - `planner`
  - `supervisor`
  - 未来可加 `executor`、`router`

不建议强行加 `mode` 的 step：

- `retriever`
- `ranking`
- `groundedness`
- `analysis`
- 简单的 `aggregation`

判断标准：

> 如果一个 handler 内部需要按不同职责切换不同 prompt、output schema、tool policy 或写入 context 的方式，就适合使用 `mode`。否则不需要。

## 4. Agent Step 的角色设计

Agent 是平台中的一个可选 step。不同 use case 对 agent 的要求不同，因此 `agent` 应支持不同 mode。

### 4.1 `agent:planner`

适合高合规、高证据要求的场景，例如财富洞察。

职责：

- 根据上游 `refined_query` 和 `intent` 理解任务。
- 激活或读取相关 skills。
- 根据 skill 的 `allowed-tools`、`required-tools`、`tool-constraints` 组装 runtime tools。
- 调用 tools 获取 evidence。
- 将完整 tool result 写入 workflow execution context。
- 返回结构化 `PlannerOutput`。

不应该做：

- 不直接生成最终用户答案。
- 不使用模型常识补充 evidence。
- 不直接拼底层系统 raw filter。
- 不调用 tenant 未启用的 tools。

典型 flow：

```json
[
  {"type": "moderation", "mode": "pre"},
  {"type": "llm", "mode": "refine_question", "model": "fast"},
  {"type": "llm", "mode": "intent", "model": "intent"},
  {"type": "agent", "mode": "planner", "agentConfig": {"llmType": "fast"}},
  {"type": "aggregation"},
  {"type": "llm", "mode": "answer", "model": "pro"},
  {"type": "llm", "mode": "compliance_review", "model": "fast"},
  {"type": "moderation", "mode": "post"},
  {"type": "analysis"}
]
```

如果第一阶段暂不实现 `aggregation` 和 `compliance_review`，可以先用：

```json
[
  {"type": "moderation", "mode": "pre"},
  {"type": "llm", "mode": "refine_question", "model": "fast"},
  {"type": "llm", "mode": "intent", "model": "intent"},
  {"type": "agent", "mode": "planner", "agentConfig": {"llmType": "fast"}},
  {"type": "llm", "mode": "answer", "model": "pro"},
  {"type": "groundedness"},
  {"type": "moderation", "mode": "post"},
  {"type": "analysis"}
]
```

### 4.2 `agent:supervisor`

适合单 agent + 多 tools 的开放任务场景，例如内部开发 open agent、运维排障 agent、代码库分析 agent。

职责：

- 端到端理解用户任务。
- 选择并调用 tools。
- 根据 tool result 决定是否继续。
- 可以直接生成最终答案或结构化输出。

典型 flow：

```json
[
  {"type": "agent", "mode": "supervisor", "agentConfig": {"llmType": "pro"}},
  {"type": "analysis"}
]
```

如果需要一点输入安全：

```json
[
  {"type": "moderation", "mode": "pre"},
  {"type": "agent", "mode": "supervisor", "agentConfig": {"llmType": "pro"}},
  {"type": "analysis"}
]
```

### 4.3 Open Agent 的平台边界

内部开发使用的 open agent 可以去掉业务合规 safeguard，例如不配置 `moderation`、`groundedness`、`compliance_review`。

但仍建议保留 runtime safety：

- tenant 级 tool allowlist。
- tool risk level。
- destructive tool 需要 confirmation 或单独权限。
- audit log。
- max tool calls、max tokens、timeout。
- secret redaction。
- workspace/environment boundary。

一句话：

> Open agent 可以去掉业务合规 guardrail，但不应该绕过平台 runtime guardrail。

## 5. 多租户目标形态

平台要支持不同 tenant 通过配置组装不同业务场景。

### 5.1 现有普通 RAG tenant

不需要 agent。

```json
[
  {"type": "moderation", "mode": "pre"},
  {"type": "llm", "mode": "refine_question", "model": "fast"},
  {"type": "retriever"},
  {"type": "ranking"},
  {"type": "llm", "mode": "answer", "model": "pro"},
  {"type": "groundedness"},
  {"type": "moderation", "mode": "post"},
  {"type": "analysis"}
]
```

### 5.2 财富洞察 tenant

需要 planner agent + evidence 约束。

```json
[
  {"type": "moderation", "mode": "pre"},
  {"type": "llm", "mode": "refine_question", "model": "fast"},
  {"type": "llm", "mode": "intent", "model": "intent"},
  {"type": "agent", "mode": "planner", "agentConfig": {"llmType": "fast"}},
  {"type": "aggregation"},
  {"type": "llm", "mode": "answer", "model": "pro"},
  {"type": "llm", "mode": "compliance_review", "model": "fast"},
  {"type": "moderation", "mode": "post"},
  {"type": "analysis"}
]
```

### 5.3 内部开发 open agent tenant

单 agent + tools。

```json
[
  {
    "type": "agent",
    "mode": "supervisor",
    "agentConfig": {
      "llmType": "pro",
      "enableTodo": true,
      "buildInTools": [
        "search_documents",
        "rank_documents",
        "plan_and_reason"
      ]
    }
  },
  {"type": "analysis"}
]
```

### 5.4 纯工具/规则 tenant

不一定需要 agent，也不一定需要最终 LLM。

```json
[
  {"type": "moderation", "mode": "pre"},
  {"type": "llm", "mode": "intent", "model": "intent"},
  {"type": "tool", "name": "some_api.lookup"},
  {"type": "formatter"},
  {"type": "analysis"}
]
```

`tool` 和 `formatter` 当前还不存在，只是未来扩展方向。第一阶段不需要实现。

## 6. 平台分层设计

推荐把平台理解成四层。

### 6.1 Flow Layer

负责“按配置执行”。

包含：

- `FlowEngine`
- `FlowStep`
- `FlowContext` / `WorkflowExecutionContext`
- `EventEmitter`

原则：

- 不知道具体业务域。
- 不知道 wealth、fund、CIO 等业务词。
- 只按 `flowConfig.steps` 找 handler 并执行。

### 6.2 Node Layer

负责每类 step 的执行逻辑。

包含：

- `ModerationHandler`
- `LLMHandler`
- `AgentHandler`
- `RetrieverHandler`
- `RankingHandler`
- `GroundednessHandler`
- `AnalysisHandler`
- 未来：`AggregationHandler`

原则：

- handler 可以使用 `mode` 做变体分发。
- handler 不应该硬编码某个 tenant 的业务规则。
- handler 的 prompt/schema 应通过 registry 或 tenant/domain config 选择。

### 6.3 Capability Layer

负责可插拔能力。

包含：

- model registry
- prompt registry
- skill registry
- tools registry
- provider registry
- MCP registry

原则：

- capability 可以被不同 tenant 复用。
- capability 是否启用由 tenant config 决定。
- tool/skill/provider 必须有 metadata，不能只是函数列表。

### 6.4 Tenant Layer

负责业务场景组装。

包含：

- flow steps
- enabled models
- enabled tools
- enabled skills
- source policy
- output contract
- domain contract
- compliance wording

原则：

- 新业务场景优先通过配置接入。
- 平台 common prompt 不包含具体业务词。
- 业务规则放在 domain contract、tenant contract、skill instructions 里。

## 7. WorkflowExecutionContext 设计

当前 `FlowContext` 更像 RAG context。建议演进成更通用的 workflow execution context。

第一阶段可以直接扩展现有 `FlowContext`，不必立即改名。字段可以逐步增加。

推荐字段：

```python
@dataclass
class FlowContext:
    # request
    query: str
    session_id: str | None = None
    message_history: list[ModelMessage] = field(default_factory=list)
    new_messages: list[ModelMessage] = field(default_factory=list)

    # resolver / intent
    refined_query: str | None = None
    resolved_query: ResolvedQuery | None = None
    intent: IntentResult | None = None

    # classic RAG
    documents: list[Document] = field(default_factory=list)
    ranked_documents: list[Document] = field(default_factory=list)

    # agent / tools / evidence
    active_skills: list[str] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    evidence_store: dict[str, EvidenceItem] = field(default_factory=dict)
    planner_output: PlannerOutput | None = None
    aggregated_evidence: AggregatedEvidenceBundle | None = None

    # answer / safety
    llm_response: str | None = None
    draft_answer: Any | None = None
    compliance_review: ComplianceReviewResult | None = None
    moderation_result: ModerationResult | None = None
    groundedness_result: GroundednessResult | None = None
    clarification_request: Any | None = None

    # infra
    metadata: dict[str, Any] = field(default_factory=dict)
    emitter: EventEmitter | None = None
    total_usage: RunUsage = field(default_factory=RunUsage)
```

原则：

- context 是 workflow 内唯一事实源。
- tool 完整结果写入 `evidence_store`。
- planner 只能拿轻量 `ToolObservation` 做下一步判断。
- synthesis 优先读取 `aggregated_evidence`，而不是 planner 的自然语言总结。

## 8. 核心数据契约

这些 model 应放在类似 `app/models/workflow.py` 或 `app/models/contracts.py` 中。不要都塞进 `domain.py`，避免未来越来越乱。

### 8.1 `ResolvedQuery`

用于 `llm:refine_question` 的升级版输出。普通 RAG 可以只用 `refined_query`，复杂业务可以用完整结构。

```python
class ResolvedQuery(BaseModel):
    original_query: str
    standalone_query: str
    language: str = "zh-Hans"

    subject_text: str | None = None
    subject_type: str = "unknown"
    normalized_subject_name: str | None = None
    aliases: list[str] = Field(default_factory=list)

    time_range_text: str | None = None
    lookback_days: int | None = None

    ambiguity: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

注意：

- `subject_type` 不要在平台层写死 wealth 枚举。
- wealth 可以通过 domain schema 或 metadata 约束它的枚举。

### 8.2 `IntentResult`

当前已有简单 `IntentResult`。建议兼容扩展。

```python
class IntentResult(BaseModel):
    intent: str
    confidence: float
    sub_intents: list[str] = Field(default_factory=list)

    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

wealth tenant 可以定义自己的 intent 名称，例如：

- `company_investment_view`
- `currency_investment_view`
- `fund_compare`
- `market_outlook`
- `out_of_scope`
- `clarification_needed`

其他 tenant 可以使用完全不同的 intent。

### 8.3 `EvidenceItem`

所有 tool 产生的可引用证据都应该 normalize 成这个结构。

```python
class EvidenceItem(BaseModel):
    id: str
    source: str
    source_type: str | None = None
    title: str | None = None
    content: str
    url: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

原则：

- `id` 由平台生成，便于 audit 和引用。
- 不同工具返回的数据都要 normalize。
- 原始响应不要直接暴露给 LLM 或用户，必要时存 raw ref。

### 8.4 `ToolObservation`

tool 返回给 planner 的轻量观察结果。

```python
class ToolObservation(BaseModel):
    tool_name: str
    status: Literal["success", "empty", "partial", "error"]
    evidence_ids: list[str] = Field(default_factory=list)

    summary_for_planner: str | None = None
    entities_found: list[str] = Field(default_factory=list)
    data_freshness: Literal["fresh", "stale", "unknown"] = "unknown"
    relevance: Literal["high", "medium", "low", "none", "unknown"] = "unknown"

    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
```

不要让 tool 返回大段文档文本给 planner。完整证据放 context。

### 8.5 `ToolCallRecord`

用于审计。

```python
class ToolCallRecord(BaseModel):
    tool_name: str
    input_payload: dict[str, Any]
    compiled_filter: str | None = None
    output_evidence_ids: list[str] = Field(default_factory=list)
    status: str
    latency_ms: int | None = None
    error: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
```

### 8.6 `PlannerOutput`

`agent:planner` 的结构化输出。

```python
class PlannerOutput(BaseModel):
    active_skills: list[str] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    required_tools_missing: list[str] = Field(default_factory=list)

    evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    stale_evidence: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)

    can_synthesize: bool
    reason: str
```

### 8.7 `AggregatedEvidenceBundle`

给 synthesis 使用的唯一证据包。

```python
class AggregatedEvidenceBundle(BaseModel):
    user_query: str
    standalone_query: str
    tenant_id: str
    intent: str | None = None
    active_skills: list[str] = Field(default_factory=list)

    evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    stale_evidence: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    synthesis_allowed: bool
    synthesis_block_reason: str | None = None
```

### 8.8 `ComplianceReviewResult`

用于 `llm:compliance_review`。

```python
class ComplianceReviewResult(BaseModel):
    passed: bool
    reason: str | None = None
    violations: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
    safe_response: str | None = None
```

## 9. Tool Registry 设计

当前 `BuiltInToolRegistry` 是 `name -> function`。为了支持多租户和不同业务场景，需要增加 tool metadata。

推荐结构：

```python
class ToolDefinition(BaseModel):
    name: str
    function: Callable[..., Any]
    description: str
    domains: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    requires_confirmation: bool = False
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    provider_key: str | None = None
```

工具解析规则：

1. tenant config 声明 `enabled_tools`。
2. skill metadata 声明 `allowed-tools` 和 `required-tools`。
3. runtime toolset = tenant enabled tools 与 skill allowed tools 的交集。
4. required tool 如果不在 runtime toolset 中，planner 必须在 `required_tools_missing` 中记录。
5. high risk tool 需要 confirmation 或特殊权限。

工具实现规则：

1. 输入必须是 Pydantic schema。
2. 输出应是 `ToolObservation` 或明确结构化对象。
3. 完整结果必须写入 `ctx.evidence_store` 或对应 context 字段。
4. 必须记录 `ToolCallRecord`。
5. 底层 filter/query 由后端 compiler 生成，不让 LLM 直接拼 raw filter。

## 10. Skill 设计

当前 skill registry 的三层加载模型可以保留：

- Tier 1：discover summaries。
- Tier 2：activate full `SKILL.md`。
- Tier 3：load references。

需要扩展 skill metadata。

推荐 `SKILL.md` frontmatter：

```yaml
---
name: currency-investment-view
description: Use this skill when the user asks for a view or outlook on a currency.
version: "1.0.0"
risk_level: medium

allowed-tools:
  - cio_house_view.search

required-tools:
  - cio_house_view.search

tool-constraints:
  cio_house_view.search:
    source_type: cio_house_view
    status: published
    subject_type: currency
    asset_class: fx
    lookback_days: 7
    require_latest: true

metadata:
  allow_model_common_knowledge: false
  allow_answer_without_evidence: false
---
```

重要原则：

- Skill 描述业务任务、流程、数据依赖、tool constraints、无证据时怎么办。
- Tool 负责具体访问系统。
- Skill 不应该写成 tool wrapper。
- Skill 不应该要求 LLM 拼底层 API filter。

## 11. Prompt 分层

Prompt 必须分层，不要把业务词放入平台 common prompt。

推荐顺序：

```text
Platform Common Contract
  > Tenant Contract
  > Domain Contract
  > Active Skill Instructions
  > Node Contract
  > Runtime Context / User Query
```

### 11.1 Platform Common Contract

平台通用规则，不包含 wealth、CIO、fund、watchlist 等具体业务词。

应包含：

- 遵守 workflow context。
- 遵守 tenant config。
- 不泄露 prompt、credentials、raw payload。
- 不调用不可用 tools。
- structured output 时只返回 schema。
- 默认不使用模型常识，除非 tenant 允许。

### 11.2 Domain Contract

业务域规则，例如 wealth、legal、ops、engineering。

wealth domain 可以包含：

- 投资/市场/产品类回答必须基于 approved sources。
- 不提供个性化金融建议，除非 context 中有授权和适当性信息。
- 无 approved evidence 时必须说明无证据，不得用常识补。

### 11.3 Tenant Contract

租户规则，例如：

- locale。
- enabled sources。
- output format。
- forbidden phrases。
- disclaimer。
- fallback skill/source。

### 11.4 Node Contract

每个 LLM/agent mode 的职责说明。

例如：

- `llm:refine_question`：只做 query resolver，不回答。
- `llm:intent`：只做分类和候选 skill，不调用工具。
- `agent:planner`：只规划和取证，不生成最终答案。
- `agent:supervisor`：可端到端调用 tools 并回答。
- `llm:answer`：只基于 context/evidence 输出。
- `llm:compliance_review`：只检查，不润色，除非合规需要。

## 12. Streaming 策略

平台应支持两种 streaming 模式。

### 12.1 普通 workflow token streaming

适合低合规风险的普通 RAG。

当前 `llm:answer` 逐 token emit 可以继续保留。

### 12.2 高合规 buffered streaming

适合财富等场景。

推荐流程：

```text
progress events
  -> planner/tool observation events
  -> synthesis internally buffered
  -> compliance review
  -> approved answer_delta events
  -> done
```

在 post safety/compliance 通过前，不直接把最终答案 token 释放给用户。

需要在 tenant config 或 flow step settings 中声明：

```json
{
  "streamingPolicy": "approved_answer_only"
}
```

第一阶段也可以先不新增配置，通过是否存在 `llm:compliance_review` step 来决定是否 buffer。

## 13. 配置模型建议

现有 `TenantConfig` 可逐步扩展，不要一次性大改。

建议新增可选字段：

```python
class DomainConfig(BaseModel):
    name: str
    prompt_pack: str | None = Field(None, alias="promptPack")
    locale: str = "zh-CN"
    allow_model_common_knowledge: bool = Field(False, alias="allowModelCommonKnowledge")
```

```python
class ToolRuntimeConfig(BaseModel):
    enabled_tools: list[str] = Field(default_factory=list, alias="enabledTools")
    max_tool_calls: int = Field(8, alias="maxToolCalls")
    require_confirmation_for_high_risk: bool = Field(
        True, alias="requireConfirmationForHighRisk"
    )
```

```python
class TenantOutputConfig(BaseModel):
    default_format: str = Field("markdown", alias="defaultFormat")
    disclaimer: str | None = None
    forbidden_phrases: list[str] = Field(default_factory=list, alias="forbiddenPhrases")
    contract: dict[str, Any] = Field(default_factory=dict)
```

可以挂到 `TenantConfig`：

```python
class TenantConfig(BaseModel):
    ...
    domain_config: DomainConfig | None = Field(None, alias="domainConfig")
    tool_runtime_config: ToolRuntimeConfig | None = Field(
        None, alias="toolRuntimeConfig"
    )
    output_config: TenantOutputConfig | None = Field(None, alias="outputConfig")
```

兼容原则：

- 老 config 不填这些字段也能运行。
- 老 RAG flow 不需要 agent/tool runtime config。
- 新 tenant 可以逐步启用。

## 14. Handler 改造设计

### 14.1 `LLMHandler`

现状：

- `mode=refine_question/intent/answer` 已存在。
- prompt 硬编码在 `app/agents/refine_question.py`、`intent_recognition.py`、`rag_answer.py`。

改造目标：

- 保留 mode 分发。
- prompt/schema 从 prompt registry 或 tenant/domain config 选择。
- `refine_question` 向 `ResolvedQuery` 兼容演进。
- `intent` 向扩展版 `IntentResult` 演进。
- `answer` 优先读取 `aggregated_evidence`；没有时兼容读取 `ranked_documents/documents`。
- 新增可选 `compliance_review` mode。

### 14.2 `AgentHandler`

现状：

- 已经支持 skills capability、MCP、built-in tools。
- 当前更像一个通用 agent orchestration step。

改造目标：

- 根据 `step.mode or "supervisor"` 分发。
- `supervisor`：可直接写 `ctx.llm_response`。
- `planner`：输出 `PlannerOutput`，写 `ctx.planner_output`，不要直接写最终答案。
- runtime tools 必须经过 tenant allowlist 和 skill allowlist 过滤。
- tool result 必须进入 context。

伪代码：

```python
async def handle(self, ctx: FlowContext, step: FlowStep) -> FlowContext:
    mode = step.mode or "supervisor"
    match mode:
        case "planner":
            return await self._run_planner(ctx, step)
        case "supervisor":
            return await self._run_supervisor(ctx, step)
        case _:
            raise ValueError(f"Unknown agent mode: {mode}")
```

### 14.3 `AggregationHandler`

新增 handler。第一版可以很简单。

职责：

- 读取 `ctx.planner_output.evidence_ids`。
- 从 `ctx.evidence_store` 取 evidence。
- 去重。
- 检查 missing/stale/conflicts。
- 写 `ctx.aggregated_evidence`。

第一版不需要 LLM，尽量 deterministic。

### 14.4 `AnalysisHandler`

扩展现有 analysis。

新增记录：

- active skills。
- tool calls。
- evidence count。
- planner output。
- compliance review。
- streaming policy。

## 15. 实施路线图

### Phase 0：整理现状和保护兼容

目标：不改变行为，只补齐当前明显不一致。

任务：

1. 修复 `FlowContext` 注释与实际字段不一致的问题，明确 `intent` 仍然存在。
2. 检查 `app/skills/capability.py` 中 `logger` 是否缺失。
3. 更新或标记过期测试，例如引用不存在 coordinator 的测试。
4. 给现有 RAG flow 加 regression tests。

验收：

- 当前 config 下原有 flow 行为不变。
- 单元测试能覆盖 `moderation -> llm -> retriever -> ranking -> answer` 基本链路。

### Phase 1：扩展 workflow contracts ✅ DONE

目标：引入通用 contract，不接入复杂业务。

任务：

1. ✅ 新增 `app/models/workflow.py`。
2. ✅ 定义：
   - `ResolvedQuery`
   - 扩展版 `IntentResult`
   - `EvidenceItem`
   - `ToolObservation`
   - `ToolCallRecord`
   - `PlannerOutput`
   - `AggregatedEvidenceBundle`
   - `ComplianceReviewResult`
3. ✅ 扩展 `FlowContext` 字段。
4. ✅ 保持 `QueryResponse` 对老字段兼容。

验收：

- 老接口响应不破坏。
- 新字段为空时不会报错。

### Phase 1.5：条件路由与 Prompt 分层 ✅ DONE

目标：支持 step 级条件路由（intent 后短路/跳转），Prompt 5 层分层，以及 Domain/Tenant output 配置。

任务：

1. ✅ 新增 `StepRoutingRule`、`StepRoutingAction` config models。
2. ✅ 在 `FlowStep` 中增加 `routing` 和 `name` 字段。
3. ✅ 修改 `FlowEngine.execute()` 支持 `abort`、`goto`、`skip_to`、`continue` 路由动作。
4. ✅ 新增 `DomainConfig`、`TenantOutputConfig` 配置模型。
5. ✅ 在 `TenantConfig` 中新增 `domain_config`、`output_config` 可选字段。
6. ✅ 扩展 `LayeredPromptBuilder` 支持 5 层分层（Identity → Guardrails → Tenant Contract → Domain Contract → Context）。
7. ✅ 新增 `LayeredPromptBuilder.build_from_config()` 便捷方法。
8. ✅ 新增测试 `tests/unit/test_flow_routing.py`（12 个测试）。
9. ✅ 新增测试 `tests/unit/test_prompt_builder.py`（14 个测试）。

验收：

- intent 识别 `out_of_scope` 后可通过 routing 直接 abort 并返回 canned response。
- 不配置 routing 的老 flow 仍按线性执行。
- 老的 3 参数 `LayeredPromptBuilder.build()` 调用不受影响。
- 新 tenant 可以通过 `domainConfig` 和 `outputConfig` 注入 contract 层。

条件路由 config 示例：

```json
{
  "type": "llm", "mode": "intent",
  "routing": [
    {"matchField": "intent.intent", "matchValue": "out_of_scope",
     "action": "abort", "response": "This question is out of scope."},
    {"matchField": "metadata.needs_clarification", "matchValue": true,
     "action": "abort", "responseFromField": "clarification_request.response"},
    {"matchField": "intent.intent", "matchValue": "simple_query",
     "action": "skip_to", "targetStep": "llm:answer"}
  ]
}
```

### Phase 2：Tool Registry metadata 化 ✅ DONE

目标：从 function list 演进成 tool definition registry。

任务：

1. ✅ 定义 `ToolDefinition`。
2. ✅ 改造 `BuiltInToolRegistry` 支持 metadata。
3. ✅ 保留 `get_tools(application_id, allowed_tool_names)` 的兼容接口。
4. ✅ 新增 `resolve_tools(tenant_config, skill_defs, requested_names)`。
5. 让 tools 可以访问 execution context 并写入 evidence。

验收：

- 现有 built-in tools 仍可用。
- 新工具可以返回 `ToolObservation`。

### Phase 3：Agent mode 分发 ✅ DONE

目标：支持 `agent:planner` 和 `agent:supervisor`。

任务：

1. ✅ 修改 `AgentHandler.handle()`，按 `step.mode or "supervisor"` 分发。
2. ✅ `supervisor` 保持现有行为，输出到 `ctx.llm_response`。
3. ✅ 新增 `_run_planner()`：
   - output type 为 `PlannerOutput`。
   - system prompt 使用 planner node contract。
   - tools 使用 tenant/skill allowlist 过滤。
   - 写 `ctx.planner_output`。
4. ✅ agent step 的缓存 key 要包含 mode。

验收：

- 不填 mode 的老 agent config 仍按 supervisor 运行。
- `mode=planner` 不直接生成最终答案。

### Phase 4：Skill metadata 扩展 ✅ DONE

目标：支持 required tools 和 constraints。

任务：

1. ✅ 扩展 `SkillMetadata`：
   - `required_tools`
   - `tool_constraints`
   - `risk_level`
2. ✅ 更新 loader parser。
3. ✅ 更新 skill discovery/activation 返回。
4. 更新 planner prompt，让 planner 明确遵守 required tools。

验收：

- 老 skill 只有 `allowed-tools` 也能加载。
- 新 skill 可以声明 required tools 和 constraints。

### Phase 5：Aggregation step ✅ DONE

目标：planner 和 answer 分离。

任务：

1. ✅ 新增 `FlowStepType.AGGREGATION`。
2. ✅ 新增 `AggregationHandler`。
3. ✅ 在 `TenantManager._build_engine()` 注册。
4. ✅ `LLMHandler.answer` 优先读取 `ctx.aggregated_evidence`。

验收：

- 有 planner/evidence 时，answer 不直接读取 raw tool observation。
- 无 aggregation 时，老 RAG answer 仍可用。

### Phase 6：Compliance review mode

目标：支持高合规场景 answer release 前检查。

任务：

1. 在 `LLMHandler` 新增 `mode=compliance_review`。
2. 定义 `ComplianceReviewResult` output schema。
3. 读取 `ctx.llm_response` 和 `ctx.aggregated_evidence`。
4. 如果不通过，设置 `ctx.llm_response` 为 safe response 或 blocked response。

验收：

- review 不通过时，最终用户拿不到 draft answer。
- review 通过时，后续 `moderation:post` 继续执行。

### Phase 7：Streaming policy

目标：支持 progress first、approved answer later。

任务：

1. 扩展 `EventType`，可选新增：
   - `progress`
   - `tool_observation`
   - `answer_delta`
2. 对高合规 flow，`llm:answer` 内部 buffer，不直接 emit token。
3. review 通过后，由 renderer 或 answer handler 释放 `answer_delta`。
4. 保留普通 RAG token streaming。

验收：

- 普通 tenant 仍可 token streaming。
- 高合规 tenant 在 review 前不会发最终答案 token。

## 16. 财富 use case 如何落地

财富只是平台上的一个 tenant/domain，不应该污染平台 common 层。

### 16.1 Wealth domain contract

放在 prompt pack 或 domain config 中。

规则示例：

- 投资、市场、基金、产品、证券、指数、货币、宏观相关回答必须基于 approved enterprise sources。
- 不允许用模型常识回答财富洞察。
- 无 evidence 时必须说明没有找到 approved evidence。
- 不提供个性化投资建议，除非 context 中有适当性、组合、司法辖区和授权。
- 必须应用 tenant disclaimer。

### 16.2 Wealth tools

推荐第一批：

- `cio_house_view.search`
- `funds_data.search`
- `watchlist.search`
- `watchlist.get_target_price`

每个 tool：

- 输入 Pydantic schema。
- 后端 filter compiler 生成 raw filter。
- 返回 `ToolObservation`。
- 写 `EvidenceItem`。

### 16.3 Wealth skill catalog

推荐第一批：

- `company-investment-view`
- `currency-investment-view`
- `index-investment-view`
- `fund-investment-view`
- `fund-compare`
- `common-cio-house-view-answer`

### 16.4 Wealth flow

建议：

```json
[
  {"type": "moderation", "mode": "pre"},
  {"type": "llm", "mode": "refine_question", "model": "fast"},
  {"type": "llm", "mode": "intent", "model": "intent"},
  {"type": "agent", "mode": "planner", "agentConfig": {"llmType": "fast"}},
  {"type": "aggregation"},
  {"type": "llm", "mode": "answer", "model": "pro"},
  {"type": "llm", "mode": "compliance_review", "model": "fast"},
  {"type": "moderation", "mode": "post"},
  {"type": "analysis"}
]
```

## 17. 实现时的注意事项

1. 不要破坏现有 flow。
2. 所有新增 config 字段都应 optional。
3. `agent` 不填 mode 时默认 `supervisor`，保持老行为。
4. `llm:answer` 没有 `aggregated_evidence` 时，继续兼容 `ranked_documents/documents`。
5. Tool 不要只返回 `None`，planner 需要 `ToolObservation`。
6. Tool 不要把完整 raw result 都塞回 prompt。
7. 平台 common prompt 不出现具体业务词。
8. wealth 规则只放 domain/tenant/skill 层。
9. Open agent 可以没有业务 safeguard，但必须保留 runtime boundary。
10. 每个阶段都要有测试，尤其是老 flow 的兼容测试。

## 18. 推荐文件改造清单

新增：

```text
app/models/workflow.py                   ✅ DONE
app/services/handlers/aggregation.py     ✅ DONE
app/prompts/registry.py                  （未来需要，当前未实现）
app/prompts/contracts.py                 （已通过 DomainConfig/TenantOutputConfig + LayeredPromptBuilder 替代）
```

修改：

```text
app/config/models.py
app/services/flow_context.py
app/services/tenant_manager.py
app/services/handlers/llm.py
app/services/handlers/agent.py
app/services/handlers/analysis.py
app/agents/tool_registry.py
app/agents/tools.py
app/skills/schema.py
app/services/events.py
app/api/schemas.py
```

新增测试：

```text
tests/unit/test_flow_engine.py
tests/unit/test_flow_routing.py           ✅ DONE (12 tests)
tests/unit/test_prompt_builder.py          ✅ DONE (14 tests)
tests/unit/test_agent_handler_modes.py
tests/unit/test_tool_registry.py
tests/unit/test_workflow_contracts.py
tests/unit/test_aggregation_handler.py
tests/integration/test_existing_rag_flow.py
tests/integration/test_agent_planner_flow.py
```

## 19. 给后续实现 agent 的执行建议

如果另一个 agent 要按本文实现，请按以下顺序做，不要跳步：

1. 先读：
   - `app/services/flow_engine.py`
   - `app/services/flow_context.py`
   - `app/config/models.py`
   - `app/services/tenant_manager.py`
   - `app/services/handlers/llm.py`
   - `app/services/handlers/agent.py`
   - `app/agents/tool_registry.py`
   - `app/skills/schema.py`
2. 先保证老测试/老 flow 不坏。
3. 新增 contracts，不改行为。
4. 改 registry metadata，保留旧接口。
5. 改 agent modes，默认 supervisor。
6. 加 planner mode。
7. 加 aggregation。
8. 加 compliance review。
9. 最后再做 wealth tools 和 wealth prompt pack。

每一步完成后都要跑相关测试，并检查 `config.json` 中现有两个 case 是否还能加载。

## 20. 最终判断标准

平台改造完成后，应满足：

- 一个 tenant 可以完全不使用 agent，只执行固定 workflow。
- 一个 tenant 可以使用 `agent:planner` 做 skill + tool evidence collection，然后由后续 LLM 输出答案。
- 一个 tenant 可以使用 `agent:supervisor` 做单 agent + tools 的 open agent。
- tools 可以按 tenant 和 skill 动态组装。
- prompt 可以按 platform/domain/tenant/skill/node 分层组合。
- wealth 这类高合规业务可以做到 evidence-based、review-before-release。
- 普通 RAG 业务不被高合规流程拖慢或复杂化。

这就是本 repo 推荐的演进方向：workflow-first，agent-optional，tenant-configured，tool/evidence-auditable。
