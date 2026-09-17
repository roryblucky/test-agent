# Agent validation 的开源最佳实践基准

**研究日期 / 访问日期：** 2026-09-08
**范围：** LangGraph、Pydantic / PydanticAI、OpenAI Agents SDK / OpenAI 官方 agent 指南、OWASP GenAI / AISVS 官方项目。
**用途：** 为 agent orchestration 的 validation 设计提供外部基准和可执行审查清单。本文不审阅本仓库实现，也不对其作 PASS / FAIL 判断。

## 摘要

成熟 agent 系统不应只有一个“总 validator”，也不应把所有校验都交给模型或结构化输出。较一致的开源实践是：**在每个信任边界，由拥有该事实的确定性组件验证一次，并让权限只能收窄、不能由模型扩大。**

需要先区分三类完全不同的校验：

1. **形状校验（shape）**：字段是否存在、类型、枚举、长度、判别联合、额外字段。它适合交给 JSON Schema / Pydantic / 框架。
2. **语义与运行不变量（semantic / invariant）**：引用是否存在、依赖是否闭合、预算是否足够、结果是否属于本轮、citation 是否有证据支持。它需要应用代码验证。
3. **授权与信任（authority / provenance）**：谁能调用什么、可访问哪些数据、哪些来源可信、能否发布。它必须由模型之外的可信控制面决定。

OpenAI 明确说明 Structured Outputs 保证 schema adherence，但不能防止对象字段内部的模型错误；因此“已经通过 schema”不等于“决策正确、获授权、证据充分”。[OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) PydanticAI 同样把结构化输出校验、output validator、tool validation 与 retry 分成不同机制。[PydanticAI Output](https://pydantic.dev/docs/ai/core-concepts/output/) OWASP 则要求授权在下游系统完整中介，而不是让 LLM 判断某个动作是否允许。[OWASP LLM06: Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/)

因此，**多层 validation 本身不是过度设计**。是否合理，要看它们是否位于不同信任边界、验证不同事实、阻止不同失效模式。若相邻函数对同一个不可变对象执行同一个纯 predicate，期间没有反序列化、并发、状态变化或权限变化，而且失败语义也相同，则更像重复代码。

## 1. Validation 所有权矩阵

| 层 | 应承担的校验 | 不应承担的校验 | 推荐失败语义 |
|---|---|---|---|
| 模型 structured-output schema | JSON 形状、必填字段、枚举 / 判别联合、局部数值与长度约束、禁止额外字段 | 权限、registry 存在性、预算全局不变量、事实正确性、证据充分性 | 可修复的 schema 错误进入**有界** output retry / repair；耗尽后形成 typed failure |
| 框架 / graph compile | 图结构、已声明节点 / edge、reducer 规则、静态 input/output state 投影、框架级 recursion limit | 中间 state 的完整运行时有效性、业务授权、模型决策正确性 | 配置错误在启动 / compile 时失败；运行时越界 fail closed |
| 图入口 | 外部 request envelope、身份 / tenant / conversation 绑定、大小限制、允许的 mode、请求级 state 初始化 | 从模型输出推导权限；把旧 checkpoint 的权限直接当成当前权限 | 在任何 actor / tool 运行前拒绝；错误不进入模型修复循环 |
| trusted policy / scope resolution | 用可信配置把 intent 映射为能力子集；当前主体权限；scope 非扩张；政策版本 | 接受模型直接给出的 tool、agent、节点或权限集合 | 未知 intent / policy、越权、scope 扩张均 fail closed；必要时澄清而非猜测 |
| dispatch / registry | specialist / tool 名称存在且在 allowlist；任务 ID、依赖闭合、轮次、并发与预算；目标与输入 DTO 匹配 | 重新判断自然语言意图；复制 tool 内部业务规则 | dispatch 前同步拒绝；只有明确可修复的规划错误才反馈模型 repair |
| tool / evidence 边界 | tool 参数形状；调用者与资源授权；业务前置条件；超时 / 限额；外部响应 schema；来源、hash、ownership、大小和敏感数据；结果与本次调用关联 | 信任模型声称的身份、来源或授权；把任意 tool 文本直接提升为可信 evidence | 预期不可用返回 bounded typed result；授权 / invariant / 未知异常 fail closed；高风险动作 approval |
| checkpoint restore | checkpoint schema / graph / policy 版本；tenant/thread/request 绑定；反序列化类型 allowlist；恢复后的 state 不变量；当前授权重新解析 | 因为 checkpoint 来自数据库就假定其仍兼容、仍获授权 | 不兼容或归属不明时阻止恢复；显式迁移或安全终止 |
| fan-in / acceptance | 完整 manifest、每个结果唯一且属于已接受 task / attempt、依赖与 provenance 闭合、partial batch 不晋升 | 用“多数结果成功”掩盖缺失或冲突；按完成顺序决定语义 | 原子接受完整批次，或形成明确 partial / failed 状态 |
| finalization / publication | 最终 output schema、citation/evidence closure、用户可见数据授权、敏感信息与内容安全、完成状态一致、长度与协议约束 | 让 synthesizer 自己决定其输出是否可发布 | 只渲染已接受对象；失败走安全 fallback / abstain；不得先流出后校验 |

下面解释各层的外部依据。

## 2. 模型 schema：负责“可解析”，不负责“可执行”

PydanticAI 使用 Pydantic 为 structured output 生成 JSON Schema，并验证模型返回的数据；它也允许 output validator 执行 schema 难以表达、甚至需要异步 I/O 的检查，并通过 `ModelRetry` 请求模型修正。[PydanticAI Output：structured data 与 output validators](https://pydantic.dev/docs/ai/core-concepts/output/)

推荐边界：

- 用 discriminated union / enum 表达闭集结果，例如 `DispatchBatch | Finish`、`Succeeded | Failed`。
- 对模型产生的控制 DTO 默认 `extra='forbid'`；Pydantic 默认其实是忽略额外字段，只有显式 `extra='forbid'` 才会拒绝。[Pydantic ConfigDict.extra](https://pydantic.dev/docs/validation/latest/api/pydantic/config/#pydantic.config.ConfigDict.extra)
- 对安全敏感 ID、整数、布尔值谨慎使用 strict mode，避免字符串到数字等静默 coercion；Pydantic 默认会尝试转换类型。[Pydantic strict mode](https://pydantic.dev/docs/validation/latest/concepts/strict_mode/)
- schema 内只放**局部、稳定、无外部状态**的约束。registry membership、当前授权、累计预算等必须在 schema 后由应用验证。
- repair 只适合模型能依据错误反馈修正的内容。认证失败、配置缺失、程序不变量破坏不能伪装成 output retry。

为什么 schema 后还需要语义校验：OpenAI 官方明确指出 Structured Outputs 仍可能在 JSON 值内部犯错。[OpenAI Structured Outputs：limitations](https://openai.com/index/introducing-structured-outputs-in-the-api/#limitations-and-restrictions) 所以一个 schema-valid 的 `specialist="admin"`、`evidence_id="invented"` 或 `budget=999` 仍必须被可信代码拒绝。

## 3. LangGraph state：schema 不是逐节点 invariant engine

LangGraph 支持 `TypedDict`、dataclass 与 Pydantic state，并支持单独的 input、internal、output schema 来限制图的公开输入 / 输出投影。[LangGraph Graph API：state 与 multiple schemas](https://docs.langchain.com/oss/python/langgraph/graph-api#state)

但官方文档明确列出一个关键限制：使用 Pydantic state 时，运行时 validation 只发生在**第一个节点的输入**，不覆盖后续节点或 graph output。[LangGraph Use Graph API：Pydantic state limitations](https://docs.langchain.com/oss/python/langgraph/use-graph-api#use-pydantic-models-for-graph-state)

由此得到的可操作基准是：

- state schema 主要是共享状态协议和静态可读性，不应被误认为所有 node update 都经过递归业务校验。
- 在 `model output -> dispatch`、`tool output -> evidence`、`parallel results -> accepted batch`、`checkpoint -> resumed execution`、`draft -> publication` 等信任跃迁处，应使用显式 typed adapter / deterministic validator。
- 不必在每个只做纯投影的节点重复验证完整 state；应在“数据的信任等级发生变化”或“副作用即将发生”的 seam 验证。
- 使用明确的 input/output schema 隐藏内部控制字段是好实践，但它只控制投影，不替代授权。

LangGraph compile 会执行基本图结构检查，例如 orphan node；edge / `Command` 则负责控制流。[LangGraph Graph API：compiling](https://docs.langchain.com/oss/python/langgraph/graph-api#compiling-your-graph) 官方还建议同一节点只选静态 edge 或动态 routing 机制，不要混用，否则两条路径都可能执行。[LangGraph Graph API：edges 与 Command](https://docs.langchain.com/oss/python/langgraph/graph-api#edges) 因而，routing target 的业务 allowlist 仍应在发出 `Send` / `Command` 前由 dispatch 层检查。

## 4. Trusted policy / scope：模型只能提议，代码决定权限

OWASP LLM06 把 excessive functionality、permissions、autonomy 列为 Excessive Agency 根因，并建议：

- 只暴露任务所需的最小工具；
- 用最小 downstream permissions；
- 以用户身份和最小 scope 执行；
- 对高影响动作加入人工批准；
- 在下游系统实现 complete mediation，而不是依赖 LLM 判断授权。

来源：[OWASP LLM06:2025 Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/)

OWASP AISVS 1.0 进一步要求 AI 资源使用 explicit allowlists 与 default deny，RAG 查询每个 retrieval / assembly 阶段都携带最终用户授权上下文，并把 agent authorization 的 policy decision point 与 agent execution environment 隔离。[OWASP AISVS C5.2](https://github.com/OWASP/AISVS/blob/main/1.0/en/0x10-C05-Access-Control-and-Identity.md)

因此较稳妥的流程是：

```text
untrusted model intent
        │
        ▼
trusted policy lookup ── current subject / tenant authorization
        │
        ▼
resolved scope (subset only)
        │
        ▼
dispatch registry / tool binding
```

这意味着模型可以输出 intent、task proposal 或 logical role，但不应输出最终生效的 credentials、tool instance、graph node、SQL permission、tenant filter 或 evidence provenance。所有这些都应由可信 registry / dependency injection / policy resolution 生成。

PydanticAI 的 dependencies 被注入 system prompt、tools 和 output validators，并通过 `RunContext` 访问；官方把它定位为 typed、可测试的服务与数据注入机制。[PydanticAI Dependencies](https://pydantic.dev/docs/ai/core-concepts/dependencies/) 对安全边界而言，dependencies 应由调用应用构造，而不是从模型输出反序列化。

## 5. Dispatch / registry：二次检查名称不是多余

structured output 中把 specialist 名限制成 enum，是第一道“模型协议”校验；dispatch 时再查可信 registry / allowlist，是第二道“可执行能力”校验。两者针对不同风险：

- enum 防止模型产生任意字符串；
- registry 防止配置漂移、插件卸载、tenant 不可用、版本不匹配；
- scope check 防止结构合法但对当前主体未授权；
- budget reservation 防止并发 branch 各自看来合法、合计却超额；
- dependency / manifest check 防止 task 引用未接受或别的 request 的结果。

LangGraph 的 `Send` 专门支持动态 map-reduce fan-out，并允许为每个 worker 传不同 state。[LangGraph Graph API：Send](https://docs.langchain.com/oss/python/langgraph/graph-api#send) 这说明动态 worker 是框架支持的模式，但并不意味着任意模型字符串都自动成为安全 node target；应用仍需先将模型 DTO 解析为可信 dispatch plan。

合理实现通常将这些规则集中在一个 dispatch admission seam：输入是 immutable candidate decision，输出是 code-assigned task IDs 和 accepted manifest。worker 只接收 accepted task，不自行扩大 scope。

## 6. Tool 与 evidence：参数 schema、授权、外部数据验证缺一不可

PydanticAI 从 Python signature 构建 tool JSON Schema；tool 参数的 Pydantic `ValidationError` 会触发 tool retry，`max_retries=N` 表示最多 `N+1` 次尝试。[PydanticAI Function Tools：Tool Schema](https://pydantic.dev/docs/ai/tools-toolsets/tools/#tool-schema)、[PydanticAI Retries：Tool retries](https://pydantic.dev/docs/ai/core-concepts/retries/#tool-retries) OpenAI Agents SDK 同样从函数签名动态创建 Pydantic model，并允许通过 `Field` 约束参数，生成的 schema 和 runtime validation 都包含这些约束。[OpenAI Agents SDK：Function tools](https://openai.github.io/openai-agents-python/tools/#automatic-argument-and-docstring-parsing)

但参数 schema 只回答“值长得对不对”。tool binding 仍必须回答：

- 当前 subject / tenant 是否有权访问这个 resource；
- 该 tool 是否属于 resolved scope；
- 资源 ID 是否属于调用者，而不是仅仅格式正确；
- 动作是否只读、可逆或需要 approval；
- 当前请求是否仍在预算、deadline、rate limit 内；
- 外部响应是否符合预期 schema、大小和安全分类。

OpenAI Agents SDK 对 agent input/output guardrail 与 tool guardrail 的触发点有明确区分：agent input guardrail 只在链首运行，output guardrail 只在最终 agent 运行，而 tool guardrail 才会在每个 custom function-tool 调用前后运行。[OpenAI Agents SDK：Guardrail workflow boundaries](https://openai.github.io/openai-agents-python/guardrails/#workflow-boundaries) 官方还指出 hosted tools、handoff 等并不走同一个 tool-guardrail pipeline，因此应用不能假定一个 agent-level guardrail 覆盖所有执行路径。[OpenAI Agents SDK：Tool guardrails](https://openai.github.io/openai-agents-python/guardrails/#tool-guardrails)

对于 approval，OpenAI Agents SDK 会在执行前再次运行 tool input guardrail；即使配置为 approval 前预检，通过预检的调用在 approval 后仍会再检查一次。[OpenAI Agents SDK：Tool guardrails](https://openai.github.io/openai-agents-python/guardrails/#tool-guardrails) 这是典型且合理的 defense-in-depth：两次检查之间发生了用户决定与时间变化，存在 TOCTOU 边界。

### Evidence 的额外要求

tool output 和网页 / 数据库内容是外部数据，不因“由 tool 返回”就自动可信。evidence acceptance 至少应绑定：

- code-assigned `evidence_id`；
- `request_id / task_id / attempt_id / tool_call_id`；
- source locator 与读取时间；
- content hash / immutable body reference；
- tenant / classification / entitlement；
- 大小上限与可供模型使用的安全投影。

模型可以总结 evidence，但不应自行制造 provenance。OWASP AISVS 的 source-attribution 研究明确建议 citation 从实际 retrieval metadata 派生，而不是由模型生成，并要求验证 claim 是否被 cited source 支持。[OWASP AISVS C7.4 research](https://github.com/OWASP/AISVS/blob/main/1.0/research/chapters/C07-Model-Behavior/C07-04-Source-Attribution-Citation-Integrity.md) 这是 AISVS 官方项目的研究说明，不是本文当作框架自动保证的功能。

## 7. Checkpoint restore：必须视为新的反序列化与版本边界

LangGraph checkpointer 会在 superstep 保存 state，并支持 fault recovery、interrupt、memory 与 replay。[LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) 恢复或 replay 时，节点会重新执行；官方要求 side effect 幂等。[LangGraph Graph API：re-execution and idempotency](https://docs.langchain.com/oss/python/langgraph/graph-api#re-execution-and-idempotency)

同时，LangGraph 明确提醒：state key 的不兼容类型变更可能让旧 thread 出问题，暂停中的 thread 对删除 / 重命名节点也有限制。[LangGraph Graph API：graph migrations](https://docs.langchain.com/oss/python/langgraph/graph-api#graph-migrations)

结合“Pydantic state 只校验首节点输入”的限制，恢复边界应显式做：

1. 验证 checkpoint / graph schema 版本和迁移路径；
2. 验证 `tenant_id / subject_id / conversation_id / request_id / mode` 绑定；
3. 只反序列化允许的类型；
4. 对将被继续使用的 DTO 执行 `model_validate` / `TypeAdapter.validate_python`；
5. 从当前可信配置重新 resolve policy / registry / credentials，而不是恢复旧权限对象；
6. 验证 pending / accepted manifest、task/result/evidence 引用闭合；
7. 无法证明兼容时 fail closed 或进入显式迁移，不要“尽量继续”。

这里在 graph 入口和 restore 后都验证身份绑定是合理重复：前者验证新 request，后者验证持久化 state 是否属于同一安全域；二者中间跨越了存储、版本和时间边界。

## 8. Retry 与 validation repair：必须有单一可见预算

LangGraph `RetryPolicy.max_attempts` 包含第一次尝试；官方源码默认值为 3，并允许用 exception type / predicate 限定 retry。[LangGraph `RetryPolicy` source](https://github.com/langchain-ai/langgraph/blob/1.1.10/libs/langgraph/langgraph/types.py#L404-L423)

PydanticAI 把 transport、provider SDK、durable workflow、model fallback、tool retry、output retry、model-request hook retry 分成互不共享预算的层；这些层会相乘。[PydanticAI Retries：layers 与 retry multiplication](https://pydantic.dev/docs/ai/core-concepts/retries/#retry-multiplication)

审查基准：

- 每层都显式配置；不能只设置 PydanticAI output retries，却忽略 provider SDK 或 graph node retry。
- retry 只针对可恢复失败：短暂网络错误、明确允许的 429 / 5xx、模型可依据反馈修正的 schema / semantic error。
- auth、permission、unknown registry key、预算耗尽、deterministic invariant、程序错误不 retry。
- “retry 次数”和“总 attempts”用一种口径，并做 exact / one-over 测试。
- 记录每层 attempt，且有一个 run-wide 上限阻止层间乘法失控。
- repair feedback 必须 bounded、脱敏，不能把 provider payload、secret 或整个不可信 document 回灌模型。

LangGraph 的 recursion limit 是 graph superstep 上限，而不是 node 内 model/tool attempts；它仍应作为 wiring bug 与无界 loop 的最后硬闸。[LangGraph Graph API：recursion limit](https://docs.langchain.com/oss/python/langgraph/graph-api#recursion-limit)

## 9. Finalization：最后一道校验有独立价值

Finalization 不是把前面所有 validator 再跑一遍，而是验证只有在“完整候选答案 + 完整 accepted evidence 集合 + 当前用户授权”同时可见时才能判断的发布不变量：

- output DTO 属于允许的 terminal variant；
- 每个 citation 指向本 run accepted evidence；
- 不存在 orphan citation、跨 tenant evidence、未授权字段；
- completion status 与 gaps / failures / zero-evidence 状态一致；
- final text / metadata / citations 来自同一个 accepted object；
- 长度、敏感数据、内容安全和协议字段满足公开边界；
- 校验失败时 abstain / safe fallback，不发布未接受 draft。

OpenAI 的 agent 指南把 guardrail 描述为 layered defense，并明确说它还必须与认证、授权、严格 access control 和常规软件安全措施结合。[OpenAI Practical Guide to Building Agents：Guardrails](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/#guardrails) OpenAI Agents SDK 也把最终 agent output guardrail 与逐 tool guardrail 分开，说明两者覆盖不同生命周期位置。[OpenAI Agents SDK：Guardrails](https://openai.github.io/openai-agents-python/guardrails/)

因此，tool output 已校验过并不能取消 finalization：tool 层证明单个结果可以被接受，finalization 证明组合后的公开答案可以被当前用户看到。

## 10. 哪些重复 validation 是合理的 defense-in-depth

重复校验满足以下任一条件时通常合理：

1. **跨越了新的信任边界。** 例如 model DTO parse 后在 dispatch registry 再检查 specialist。
2. **事实可能随时间变化。** 例如 approval 前预检、approval 后执行前再授权。
3. **发生过反序列化或恢复。** 例如 API 输入验证过，但 checkpoint 恢复后重新验证 owner/version。
4. **存在并发聚合。** 每个 task 独立预算合法，dispatch 时仍要原子 reserve aggregate budget；每个 result 合法，fan-in 仍要验证 manifest 完整。
5. **检查对象不同。** tool ingestion 验证 evidence record；finalization 验证 answer-to-evidence 引用闭合。
6. **一个是静态闭集，一个是动态可用性。** enum 限制候选名称；runtime registry 检查部署中当前存在且对 tenant 可用。
7. **一个负责安全，一个负责用户体验。** 权限 gate fail closed；模型 repair 只改善格式，不拥有许可权。
8. **一个验证单项，一个验证累计量。** 每次调用参数 / 限额与 run-wide steps、tokens、bytes、cost、delegation depth 同时存在。

特别值得保留的双重 gate：

- structured-output enum **+** trusted registry membership；
- graph/API identity validation **+** checkpoint owner validation；
- tool argument schema **+** downstream resource authorization；
- evidence ingestion provenance **+** final citation closure；
- task-level allowance **+** aggregate atomic budget reservation；
- approval 前风险检查 **+** side effect 前重新授权；
- per-node bounded retry **+** run-wide attempt / recursion cap。

## 11. 哪些重复 validation 是 code smell

以下信号说明校验可能已经失去边界意义：

1. 相邻两层对同一 immutable DTO 调用同一 predicate，中间没有 mutation、I/O、反序列化、并发或权限变化。
2. Pydantic 已表达字段约束，应用又手写完全相同的类型 / 必填 / enum 检查，却没有更强失败语义。
3. 每个 getter / node 都重新 `model_validate(model_dump())`，只是为了“保险”，但没有信任跃迁。
4. 多个 validator 各自维护不同的 allowlist、默认值或 normalization，导致同一输入在不同位置结论不同。
5. 先 coercion / truncate，后面又按 strict 原值判断；校验实际上看的是不同对象，却没有显式 canonical form。
6. validator 同时做 shape、授权、I/O、状态 mutation、logging side effect 和 routing，形成不可测试的 mega-validator。
7. 捕获所有 `ValidationError` / `Exception` 并统一交给模型重试，使 auth、bug、provider config error 被掩盖为“模型没答好”。
8. 多层都自行 retry，却没有共享 ledger，造成 `graph × agent × provider × transport` 尝试乘法。
9. 把可信 policy 全量放进 prompt，让模型“自我检查”后，代码仍不得不执行相同授权；模型侧检查没有形成安全边界，反而扩大泄露面。
10. 校验函数只有日志没有 gate；调用者忽略返回值；或遇到 policy service 不可用时默认放行。
11. 同一 domain rule 被复制到 query understanding、coordinator、specialist、synthesizer，而没有一个明确 owner 和复用的纯函数。
12. finalization 只是机械重复字段 schema，却不检查 citation closure、entitlement、completion consistency 等只有终局才可见的性质。

建议的去重原则：**每条 invariant 指定一个 authoritative owner；其他边界只调用该 owner 的纯函数或验证自己的附加前提，不复制规则。** 在代码 / 测试中记录：`invariant -> owner -> callers -> failure type -> retryable?`。

## 12. 可操作审查清单

### A. 模型输入与输出

- [ ] 所有控制流输出都是 closed typed variants，而非自由文本解析。
- [ ] 模型控制 DTO 禁止 extra fields；关键标识符和标量使用合适的 strictness。
- [ ] schema-valid 后仍有独立 semantic validator。
- [ ] 模型不能直接提供 credentials、tool instance、任意 node 名、最终 authority 或 provenance。
- [ ] repair 次数有上限，错误反馈 bounded 且脱敏。

### B. Graph state 与 routing

- [ ] input / internal / output state schema 职责清楚。
- [ ] 团队没有误以为 Pydantic state 会校验所有 node updates / output。
- [ ] 每个信任跃迁都有明确 validator，而不是每个节点全量重验。
- [ ] 同一节点不混用会同时执行的 static edge 与 dynamic routing。
- [ ] routing target 在 `Send` / `Command` 前已映射为可信 registry entry。
- [ ] reducer 的 identity、交换律 / 顺序无关、冲突行为和 reset 语义有测试。
- [ ] recursion limit 与业务 task/round/model/tool budgets 分开配置。

### C. Policy、scope 与 dispatch

- [ ] authorization decision point 位于 agent / prompt 外部并 default deny。
- [ ] intent 只能选择可信 policy，不能携带生效权限。
- [ ] 下游 scope 是上游授权的子集，delegation 不可扩权。
- [ ] dispatch 原子验证 registry、allowlist、task IDs、dependencies、round、budget、tenant availability。
- [ ] code 分配 execution IDs；worker 只接收 accepted task。
- [ ] mutable policy / authorization 在真正副作用前重新检查。

### D. Tool 与 evidence

- [ ] framework/Pydantic 校验参数 shape；tool binding 另行校验 subject-resource authorization。
- [ ] 每个 tool 使用最小功能、最小 downstream credential 和明确 side-effect 等级。
- [ ] 高影响 / 不可逆动作在副作用之前 approval。
- [ ] hosted tool、handoff、MCP、subagent 等旁路没有被误认为受 custom function-tool guardrail 自动覆盖。
- [ ] 外部 tool response 经 schema、size、classification 和 injection-safe projection 验证。
- [ ] evidence ID / provenance 由代码生成并绑定 request/task/attempt/tool call。
- [ ] rejected / timed-out / abandoned attempt 的 evidence 不会进入 accepted catalog。

### E. Persistence 与恢复

- [ ] checkpoint 恢复验证 schema/graph/policy version 与 tenant/thread/request identity。
- [ ] serializer 使用最小类型 allowlist；checkpoint 不承载 live client、credential 或任意 executable object。
- [ ] 恢复后的 critical DTO 显式重验；旧权限从当前 policy 重新解析。
- [ ] incompatible state migration fail closed，并有版本兼容测试。
- [ ] re-executed node 的副作用有 idempotency key / upsert / compensation。

### F. Fan-in、finalization 与 publication

- [ ] partial branch writes 与 accepted canonical result 分离。
- [ ] barrier 验证完整 manifest、唯一 task/result、attempt ownership、dependency closure。
- [ ] finalization 验证 citation 指向 accepted evidence，且 evidence 对当前用户可见。
- [ ] completion status、failures、gaps、zero-evidence 一致。
- [ ] 最终文本、metadata、citations 从同一个 accepted object 渲染。
- [ ] publication 发生在 final validation 之后；失败时发布 typed safe fallback 或 abstain。

### G. Retry 与可观测性

- [ ] graph / actor / output / tool / provider / transport retry 均显式配置。
- [ ] 有单一 run-wide attempt ledger 和 exact/one-over 测试。
- [ ] retry exception 分类是闭集；auth/config/invariant/unknown 不 retry。
- [ ] logs / traces 记录 validator 名、边界、reason code、attempt、policy version，但不泄露原始 secret。
- [ ] 每条重复校验都能说明中间跨越了什么 trust/time/concurrency boundary；否则合并。

## 13. 结论性判据

判断一个 validation-heavy agent 设计是否合理，不应简单数 validator 数量，而应逐条回答：

1. **它保护哪个信任边界？**
2. **它验证的是 shape、semantic invariant，还是 authority/provenance？**
3. **这个事实由哪个组件权威拥有？**
4. **上一次检查后，数据、时间、并发、版本或授权是否可能变化？**
5. **失败是可由模型修复、可重试的 transient failure，还是必须 fail closed？**
6. **它是否在任何 token / tool side effect / durable promotion / publication 之前生效？**
7. **是否有 exact、one-over、cross-tenant、stale-checkpoint、partial-batch 与 unknown-exception 测试？**

若每个 validator 都能给出不同且清楚的答案，多层校验通常是 production-grade defense-in-depth。若多个 validator 的答案完全相同，且没有新的信任边界或时序变化，应合并到一个 authoritative owner。

## Primary sources

以下来源均于 **2026-09-08** 访问；只列官方文档、官方源码或官方安全项目。

### LangGraph / LangChain

- [LangGraph Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api)：state、multiple schemas、compile、routing、`Send` / `Command`、migrations、recursion limit、re-execution。
- [LangGraph Use Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)：Pydantic state 的运行时 validation 限制。
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)：checkpoint、thread、fault recovery、pending writes、replay。
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)：resume、输入 validation loop、节点从头重放与幂等性。
- [LangGraph 1.1.10 `RetryPolicy` source](https://github.com/langchain-ai/langgraph/blob/1.1.10/libs/langgraph/langgraph/types.py#L404-L423)：固定版本的 attempts 语义。

### Pydantic / PydanticAI

- [PydanticAI Output](https://pydantic.dev/docs/ai/core-concepts/output/)：structured output、Pydantic validation、output validators、`ModelRetry`。
- [PydanticAI Retries](https://pydantic.dev/docs/ai/core-concepts/retries/)：retry layers、multiplication、tool/output retry。
- [PydanticAI Function Tools](https://pydantic.dev/docs/ai/tools-toolsets/tools/)：tool signature 到 JSON Schema。
- [PydanticAI Dependencies](https://pydantic.dev/docs/ai/core-concepts/dependencies/)：typed dependency injection 与 `RunContext`。
- [Pydantic strict mode](https://pydantic.dev/docs/validation/latest/concepts/strict_mode/)：默认 coercion 与 strict validation。
- [Pydantic `ConfigDict.extra`](https://pydantic.dev/docs/validation/latest/api/pydantic/config/#pydantic.config.ConfigDict.extra)：默认 ignore 与显式 forbid。

### OpenAI

- [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)：schema adherence 与 semantic-error 限制。
- [OpenAI Agents SDK: Agents](https://openai.github.io/openai-agents-python/agents/)：typed output 与 orchestration patterns。
- [OpenAI Agents SDK: Tools](https://openai.github.io/openai-agents-python/tools/)：function-tool schema、Pydantic constraints、timeout。
- [OpenAI Agents SDK: Guardrails](https://openai.github.io/openai-agents-python/guardrails/)：input/output/tool guardrail 的生命周期边界和旁路限制。
- [OpenAI Practical Guide to Building Agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)：layered guardrails、tool risk、human intervention。

### OWASP

- [OWASP LLM06:2025 Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/)：最小功能 / 权限 / 自治、user-context execution、approval、complete mediation。
- [OWASP AISVS 1.0 C5](https://github.com/OWASP/AISVS/blob/main/1.0/en/0x10-C05-Access-Control-and-Identity.md)：AI resource allowlist/default deny、query-time authorization、policy decision point isolation。
- [OWASP AISVS C7.4 research](https://github.com/OWASP/AISVS/blob/main/1.0/research/chapters/C07-Model-Behavior/C07-04-Source-Attribution-Citation-Integrity.md)：官方项目研究材料；citation metadata 与 claim-support verification。本文明确不把该 research 页面视为框架保证。
