# Google ADK 迁移能力研究底稿

> 日期：2026-09-09
> 范围：Google Agent Development Kit（ADK）Python 的编排、运行时、持久化、恢复、工具、模型、评测与部署能力；本文不读取当前仓库实现，不替代调用点清点与迁移 spike。
> 证据原则：只使用 Google 官方 ADK 文档和 `google/adk-python` 官方源码/发行记录。滚动的 `main` 文档与锁定到 `v2.8.0` 的发行事实分开表述。

## 1. 结论

**ADK 2.x 已经是一个覆盖面很完整的 agent 应用框架，但不是“换一个 agent 类”这么小的替换。** 它能够接管图编排、事件循环、session/state、SQL 持久化、断点恢复、流式事件、生命周期 hook、MCP、评测与 Google Cloud 部署；与此同时，应用仍需自行保留或重建业务状态机、领域输出契约、幂等副作用、memory/artifact 后端选择和对外 API 兼容层。

与迁移决策最相关的事实是：

1. **编排能力匹配度高。** ADK 2.0 的 `Workflow` 把 agent、tool、普通函数和嵌套 workflow 都作为节点，支持显式边、条件路由、扇出/汇合、动态流程和 typed node data。官方定位是用确定性代码包住非确定性模型步骤，而不是把全部流程写进 prompt：[Graph workflows](https://adk.dev/graphs/)、[data handling](https://adk.dev/graphs/data-handling/)。
2. **持久化不是现有任意业务数据模型的等价替代。** `DatabaseSessionService` 能把 session、event 和 state 持久化到 PostgreSQL/MySQL/SQLite；memory 与 binary artifacts 是独立 service，且官方 Python 内置持久 memory/artifact 选择主要偏向 Google Cloud。因此“已有 PostgreSQL checkpoint + 领域表”不能假设可直接映射成一个 ADK service：[sessions](https://adk.dev/sessions/session/)、[memory](https://adk.dev/sessions/memory/)、[artifacts](https://adk.dev/artifacts/)。
3. **恢复语义有价值，但会新增幂等约束。** 内置 resume 能跳过已完成步骤和成功工具结果，却明确只保证工具“至少一次”，恢复时工具可能重复执行；付款、发布、写库等副作用必须用 idempotency key/业务去重保护：[resume](https://adk.dev/runtime/resume/)。
4. **OpenAI 等非 Google 模型会多一层适配。** Python 原生路径重点是 Gemini/Vertex/Claude/Agent Platform；OpenAI Python 通过 LiteLLM，而不是官方 OpenAI SDK 的一等原生适配。官方还专门记录了 2026-03 的 LiteLLM supply-chain 事件：[model support](https://adk.dev/agents/models/)、[OpenAI](https://adk.dev/agents/models/openai/)、[LiteLLM](https://adk.dev/agents/models/litellm/)。
5. **迁移风险主要在运行语义，不在样板代码。** ADK 2.0 于 2026-05-19 GA；截至 2026-08-25 的最新 `v2.8.0` 仍在密集修复 resume、parallel、tool-call、session rehydration、安全与事件正确性问题。应精确 pin 2.x、跑兼容性 spike，并以现有 golden tests/evals 作为迁移门槛：[ADK 2.0](https://adk.dev/2.0/)、[`v2.8.0` release](https://github.com/google/adk-python/releases/tag/v2.8.0)。

## 2. 版本与成熟度基线

- ADK Python 2.0 在 2026-05-19 GA；2.0 将执行核心从层级 agent executor 转为 graph workflow runtime，并引入了 event schema、custom executor 和异常处理的 breaking changes：[2.0 compatibility notes](https://adk.dev/2.0/)。
- 2026-09-09 可见的最新 2.x release 是 `v2.8.0`（release note 日期 2026-08-25）。它增加 custom eval metrics、最大 LLM 调用数、模型 client 注入和实验性 token/workflow telemetry，同时包含大量并发、恢复、重复工具执行、session rehydration、credential redaction 与 prompt-injection 修复：[v2.8.0](https://github.com/google/adk-python/releases/tag/v2.8.0)。
- “GA”说明 2.x 是正式采用路径，但不等于 API/行为已经低 churn。**判断**：生产迁移应精确 pin minor/patch、逐版本读 release notes，并用契约测试锁定 event、resume、tool 与 streaming 行为。
- Python 最低版本为 3.10；核心依赖包括 Pydantic 2、FastAPI 和 Google Gen AI SDK。MCP 是可选依赖；`v2.8.0` 的依赖闭包应以对应 tag 的 [`pyproject.toml`](https://github.com/google/adk-python/blob/v2.8.0/pyproject.toml) 为准。

## 3. 能力事实与迁移含义

### 3.1 工作流与 multi-agent 编排

| 能力 | 官方事实 | 对迁移的含义（判断） |
|---|---|---|
| Graph workflow | `Workflow` 以 node/edge 表达顺序、分支、扇出；节点可为 agent、tool、函数或 workflow；event `output` 把结果交给下游。[官方文档](https://adk.dev/graphs/) | 与显式状态图/有向工作流概念接近，是最值得做 spike 的替换面。 |
| Typed data | Python 节点可声明 Pydantic-compatible `input_schema`、`output_schema`、`state_schema`；`message` 是用户输出，`output` 是节点传值，`state` 是跨节点持久状态。[数据处理](https://adk.dev/graphs/data-handling/)、[`BaseNode` v2.8.0](https://github.com/google/adk-python/blob/v2.8.0/src/google/adk/workflow/_base_node.py) | 能收敛节点边界，但不能自动表达现有领域 invariant；领域 Pydantic models 仍应由应用拥有。 |
| Sequential/Parallel/Loop | 预置 template agents 仍可用且调度本身不由 LLM 决定；但 Python/Go 2.0 文档说它们已被更灵活的 graph/dynamic workflow **supersede**。[template workflows](https://adk.dev/agents/workflow-agents/)、[parallel](https://adk.dev/agents/workflow-agents/parallel-agents/) | 新迁移不宜把 legacy template classes 当长期核心；直接面向 2.x graph runtime 可少欠一轮升级债。 |
| Parallel isolation | `ParallelAgent` 分支并发执行，运行期间不自动共享 history/state，结果顺序可能不确定；共享数据需显式锁、外部存储或后处理。[parallel](https://adk.dev/agents/workflow-agents/parallel-agents/) | 现有 fan-out/fan-in 的结果排序、共享预算、取消和失败传播必须逐项做 parity test。 |
| Custom orchestration | 旧式 `BaseAgent._run_async_impl()` 可完全自定义，但官方称其为 advanced 且已被 graph/dynamic workflow supersede。[custom agents](https://adk.dev/agents/custom-agents/) | 用 custom agent 复刻原引擎会把框架内部语义再次背到应用层，技术债通常最高。 |

Graph LLM agent 必须按 single-turn/task 模式运行；一个 node execution 只能产生一个 `Event.output`，多次 output 会报 runtime error。Graph 的已知限制还包括“部分 third-party integrations 可能不兼容”，官方没有给出完整兼容矩阵：[graphs](https://adk.dev/graphs/)、[data handling](https://adk.dev/graphs/data-handling/)。这意味着实际使用的 model/tool/plugin 组合必须做运行 spike，不能仅凭类型能构造就判定兼容。

### 3.2 Session、state 与 SQL persistence

- 一个 `Session` 是单个 conversation thread，包含按时间排序的 events、session state 和更新时间；`SessionService` 管理创建、读取、追加 event、列举与删除。[session lifecycle](https://adk.dev/sessions/session/)。
- Python 官方实现包括无持久化的 `InMemorySessionService`、托管的 `VertexAiSessionService`，以及自管关系库的 `DatabaseSessionService`。后者支持 PostgreSQL/MySQL/SQLite，需要 async driver 与 `google-adk[db]`：[session implementations](https://adk.dev/sessions/session/#sessionservice-implementations)。
- `DatabaseSessionService` 对同进程同 session 使用锁，并在 PostgreSQL/MySQL/MariaDB 使用 `SELECT ... FOR UPDATE` 保护跨进程 append；官方也记录过 schema migration 要求。这是 session event 一致性保证，不等于业务写入与 agent checkpoint 的分布式事务：[concurrency and locking](https://adk.dev/sessions/session/#concurrency-and-locking)。
- State 具有四种 scope：无前缀为 session，`user:` 跨该用户 sessions，`app:` 跨全 app，`temp:` 仅当前 invocation；更新应通过 `EventActions.state_delta`/context 写入，由 runner 在 event commit 时持久化：[state](https://adk.dev/sessions/state/)、[event loop](https://adk.dev/runtime/event-loop/)。
- ADK 2.0 给 Event 新增 `node_info` 与 `output`。自定义 rigid-column session store 和使用 `additionalProperties: false` 的严格下游 event schema 都必须升级；直接向 `context.session.events` append 会绕过 graph runner，官方明确视为不安全：[2.0 event changes](https://adk.dev/2.0/)。

**迁移含义（判断）**：若现有数据库既承载对话/checkpoint，也承载任务、证据、审批、报告等领域实体，较稳妥的 seam 是先只让 ADK 拥有 invocation/session event log；领域表与事务继续由应用拥有。把所有业务状态塞入 ADK state 会形成弱 schema、跨 scope 污染和双写一致性债。

### 3.3 Memory 与 artifacts

- Session 是当前 conversation history；Memory 是跨 session 的可搜索信息。Python 官方列出的 core memory services 为：非持久/基础搜索的 `InMemoryMemoryService`、托管且会抽取/整合 semantic memory 的 `VertexAiMemoryBankService`、以及保存完整 session 并通过 Knowledge Engine/RAG 检索的 `VertexAiRagMemoryService`：[memory services](https://adk.dev/sessions/memory/)。
- 官方 core 文档未列出通用 `DatabaseMemoryService`。因此若已有 PostgreSQL/pgvector memory 或自定义检索，需要保留现有层，或实现 `BaseMemoryService` adapter；采用 Vertex memory 则引入 GCP resource、权限、数据驻留和供应商绑定。
- Artifact 是命名、版本化的 binary data，独立于 state/session；Python 内置 `InMemoryArtifactService` 和持久化 `GcsArtifactService`，载荷使用 `google.genai.types.Part`：[artifacts](https://adk.dev/artifacts/)。
- 官方 core 文档没有 PostgreSQL/S3 通用 durable artifact backend。**判断**：已有文件/对象存储不应为迁就 ADK 而搬迁；写薄 adapter 通常比数据迁移风险低。

### 3.4 Resume、replay 与副作用

- Resume 是 opt-in：在 `App` 上设置 `ResumabilityConfig(is_resumable=True)`，再通过原 `invocation_id` 调用 `Runner.run_async` 或 `/run_sse`。Web UI 与 ADK CLI 当前不支持发起 resume：[resume setup](https://adk.dev/runtime/resume/)。
- Sequential 从下一 child 恢复；Loop 保留 child/iteration；Parallel 只重跑未完成 branches。成功 tool result 会重放，失败/未完成 tool 会再次执行：[resume mechanics](https://adk.dev/runtime/resume/#how-it-works)。
- 官方保证是工具至少执行一次，恢复时可能多次；停止后改变 workflow 再 resume 不受支持。Custom Agent 默认不可恢复，必须显式定义 `BaseAgentState`、step checkpoints 与 completion marker：[resume cautions](https://adk.dev/runtime/resume/)。

**迁移含义（判断）**：ADK resume 能删除一部分手写 checkpoint 调度代码，但不能删除业务幂等、outbox、唯一约束或人工审批 token。若现有系统要求“exactly once”副作用，迁移后这部分债不会消失，反而必须显式化。

### 3.5 Events、streaming 与 API surface

- Runner 是 async-first；agent/tool/callback 以 `Event` 流与 runner 协作。Events 可表达 user/model content、function call/result、state/artifact delta、partial token、control/error，并带 `invocation_id`：[events](https://adk.dev/events/)、[event loop](https://adk.dev/runtime/event-loop/)。
- Token streaming 会立即产生 `partial=True` events；runner 只在最终 non-partial event 上处理 action/commit state 与 artifacts。同步 Python tool 会占用 event-loop thread，可能阻塞其他异步工作：[streaming behavior](https://adk.dev/runtime/event-loop/#important-runtime-behaviors)。
- 内置 API server 提供 session CRUD、`/run` 与 `/run_sse`；SSE 传输的是 event objects，开启 streaming 后包含 token chunks：[API server](https://adk.dev/runtime/api-server/)。

**迁移含义（判断）**：若现有外部 SSE event name、payload、重连游标或 replay contract 已被前端/客户端消费，不能直接把 ADK Event 泄露成公共 API。应保留稳定的 application event envelope，并在内部把 ADK events 映射过去；否则会把上游 event schema churn 变成客户兼容债。

### 3.6 Callbacks、plugins、tools 与 MCP

- Agent callbacks 覆盖 before/after agent、model、tool，可观察、修改或短路对应步骤，并能访问 state/services；适合 agent-local 策略、校验与替换：[callbacks](https://adk.dev/callbacks/)。
- Plugin 在 `Runner` 注册一次，hooks 作用于该 runner 下所有 agents/tools/models，适合 tracing、authorization、guardrail、metrics、cache 等横切能力：[plugins](https://adk.dev/plugins/)。
- Plain Python function 可自动包装为 tool；type hints/docstring 生成调用 schema，`ToolContext` 注入 state/actions/artifact 能力。同步与异步 functions 都支持：[function tools](https://adk.dev/tools-custom/function-tools/)。
- `McpToolset` 负责 MCP tool discovery、schema adaptation、调用与连接生命周期，支持 stdio、SSE 和 Streamable HTTP；部署文档要求 agent/toolset 定义同步可装载，并单独说明 Cloud Run/GKE/Agent Runtime 的连接模式：[MCP tools](https://adk.dev/tools-custom/mcp-tools/)。

**迁移含义（判断）**：授权、审计、tenant isolation 和错误归一化宜放 runner-level plugin 或 application boundary；若散落到每个 tool callback，会形成重复且容易漏配的 policy 债。MCP adapter 能复用，但连接池、认证、超时、部署拓扑仍是应用运维责任。

### 3.7 模型与 typed structured output

- ADK 直接支持 Gemini、Claude 和 Agent Platform hosted models，也提供 Apigee、LiteLLM、Ollama、vLLM、LiteRT-LM connectors，以及运行时 model routing/failover：[models](https://adk.dev/agents/models/)。
- Python 使用 OpenAI 必须经过 LiteLLM；LiteLLM 对外提供 100+ providers 的统一接口，但这也增加一层参数、错误、tool-call 与 structured-output 兼容矩阵。官方要求当前使用 `litellm>=1.84`，并记录 1.82.7/1.82.8 被植入未授权代码的供应链事件：[OpenAI path](https://adk.dev/agents/models/openai/)、[LiteLLM advisory](https://adk.dev/agents/models/litellm/)。
- `LlmAgent.output_schema` 支持 Pydantic models、list、primitive 与 schema dict；最终输出可验证并通过 `output_key` 写 state。但 tools + output schema 只有部分 models 能在同一 request 原生组合；fallback `set_model_response` 方式被官方标为可能不可靠，建议必要时拆 formatter agent。验证失败时 `output_key` 还可能保存 raw response string，而不是 parsed object：[structured input/output](https://adk.dev/agents/llm-agents/#structure-data-input-and-output)、[`LlmAgent` v2.8.0](https://github.com/google/adk-python/blob/v2.8.0/src/google/adk/agents/llm_agent.py)。

**迁移含义（判断）**：若系统主要使用 OpenAI 或多 provider，ADK 的 Google-native 优势会打折。迁移验收必须按“provider × streaming × tools × structured output”做契约矩阵，而不是只测试 Gemini happy path。

### 3.8 Retry、timeout 与运行上限

- 2.x graph `BaseNode` 有 `retry_config` 和 `timeout`；timeout 会取消节点、抛 `NodeTimeoutError`，并可进入相同 retry policy。`RetryConfig` 支持最大 attempts、指数退避、最大 delay、jitter 和 exception filtering：[`BaseNode` v2.8.0](https://github.com/google/adk-python/blob/v2.8.0/src/google/adk/workflow/_base_node.py)、[`RetryConfig` v2.8.0](https://github.com/google/adk-python/blob/v2.8.0/src/google/adk/workflow/_retry_config.py)。
- Node retry 与 model HTTP retry 是不同层。Gemini wrapper 接受 Google Gen AI `HttpRetryOptions`；provider connector 的 timeout/retry 能力不必然一致：[`Google LLM` source](https://github.com/google/adk-python/blob/v2.8.0/src/google/adk/models/google_llm.py)。
- `v2.8.0` 增加 `ADK_MAX_LLM_CALLS` 运行上限；这不是业务级 wall-clock、token、cost 或 fan-out budget 的完整替代：[release](https://github.com/google/adk-python/releases/tag/v2.8.0)。
- ADK 2.0 会捕获异常以实现 retry/telemetry/HITL；tool 内部宽泛 `except Exception` 会屏蔽 framework retry，捕获 `BaseException` 还可能破坏 HITL interrupt：[2.0 error handling](https://adk.dev/2.0/#error-handling-automatic-retries)。

**迁移含义（判断）**：需要先列出现有的 node、LLM、tool、整次请求四层 budget，再分别映射。只打开默认 retry 会放大费用与副作用，且不能证明整次请求有界。

### 3.9 Evaluation、测试与 observability

- ADK eval 支持 Web UI、`pytest` 的 `AgentEvaluator`、`adk eval` CLI 与可接 CI 的 conformance tests；既能评 final response，也能评 tool trajectory、rubric、hallucination、safety、user simulation 与 multi-turn success：[evaluation](https://adk.dev/evaluate/)、[criteria](https://adk.dev/evaluate/criteria/)。
- Environment Simulation 可通过 tool callback/plugin 注入受控 tool responses，用于离线、可复现的 API 错误/边界场景测试：[environment simulation](https://adk.dev/evaluate/environment_simulation/)。
- 内置 observability 覆盖 logs、metrics、traces，并支持 OpenTelemetry/Google Cloud 路径；记录 prompts/content 可能包含 PII，需要显式数据治理：[observability](https://adk.dev/observability/)。`v2.8.0` 的每 invocation/workflow token 与调用计数 telemetry 仍是 experimental、默认关闭：[v2.8.0](https://github.com/google/adk-python/releases/tag/v2.8.0)。

**迁移含义（判断）**：eval/trace 是 ADK 最明确的收益之一，但现有领域正确性（证据完整性、计算门、引用归属、审批等）仍需自定义 metric/assertion。不要用框架自带 trajectory score 代替业务 invariant tests。

### 3.10 部署

- Python 可用 `adk deploy cloud_run` 部署自管 Cloud Run；默认不包含 Web UI，`--with_ui` 才会加入：[Cloud Run](https://adk.dev/deploy/cloud-run/)。
- Cloud Run 若沿用默认 in-memory session/artifact services，instance recycle 时数据会丢失；生产必须显式配置 persistent services：[Cloud Run persistence](https://adk.dev/deploy/cloud-run/)。
- GKE 支持手工 FastAPI/container/Kubernetes 配置或 `adk deploy gke`：[GKE](https://adk.dev/deploy/gke/)。
- Google Cloud Agent Runtime/Agent Engine 提供 managed deployment path；需要项目、region、IAM 与相应资源配置：[Agent Runtime](https://adk.dev/deploy/agent-engine/deploy/)。
- 内置 `adk web` 面向开发与调试；生产仍需明确 auth、CORS、secrets、session/memory/artifact services、伸缩和 observability。官方认证文档也不建议把 refresh/access tokens 放进 session state，而应使用 secret manager 或加密 token store：[authentication](https://adk.dev/tools/authentication/)。

**迁移含义（判断）**：采用 ADK 不要求迁往 Google managed runtime；继续现有容器平台是可行的。若为了少写运维而采用 Agent Runtime、Vertex memory 和 GCS artifacts，收益会与 GCP lock-in、IAM 和数据治理成本一起出现。

## 4. 对 benefits / downsides / effort / 技术债的事实化拆解

### 4.1 可能获得的收益

- 删除或缩小通用 orchestration runtime：graph scheduler、event loop、typed node edges、parallel/loop、retry/timeout、resume。
- 删除或缩小通用 agent infrastructure：session/event persistence、SSE event generation、callback/plugin plumbing、MCP adaptation、eval harness、trace UI。
- 若已在 GCP/Gemini：model、Agent Runtime、Vertex sessions/memory、GCS artifacts 与 Cloud observability 的集成路径更短。
- 把 workflow control 从 prompt/手写 dispatcher 移到显式 graph，通常更易观察与评测。

### 4.2 确定的代价或缺口

- 需要把当前 orchestration state/checkpoint/event 语义重新建模到 ADK 2.x，而非机械换 import。
- 需要维护 application event/API adapter，除非允许客户端直接绑定 ADK schema。
- 需要保留领域数据库/事务；ADK session persistence 不替代业务数据模型。
- Durable generic memory/artifact backend 选择少；非 GCP 后端需要 adapter。
- Resume 为 at-least-once，副作用幂等是硬要求。
- OpenAI Python 走 LiteLLM；多 provider 的 capability parity 与供应链面扩大。
- 2.x 很新且 release churn 高；升级测试和版本 pin 是长期维护成本。

### 4.3 Effort 不能只按 LOC 估算

在未读取仓库的前提下，不给虚假的人日精度。应按下列 workstream 估算，每项都以 parity test 通过为完成：

| Workstream | 主要工作 | 风险 |
|---|---|---|
| Workflow | node/edge、fan-out/fan-in、dynamic routing、termination、cancel | 高；决定运行语义 |
| State/persistence | session/event schema、领域表 seam、transaction、migration/rollback | 高 |
| Resume/idempotency | checkpoint mapping、重复 tool call、防重/outbox | 高 |
| Streaming/API | ADK Event → 现有 SSE/API envelope、reconnect/replay | 中高 |
| Models/tools | provider matrix、structured output、MCP、auth、retry/timeout | 中高 |
| Memory/artifacts | 保留后端或实现 service adapters | 取决于现状 |
| Hooks/observability | callback/plugin、tenant policy、PII redaction、trace/metrics | 中 |
| Eval/cutover | golden cases、failure injection、shadow traffic、rollback | 高但不可省 |

**判断**：只有在 spike 证明“大部分领域节点可直接成为 ADK graph nodes，现有 API/DB seam 可保留”时，全面迁移才可能是中等 effort。若必须重写公共 streaming contract、checkpoint schema、OpenAI provider boundary 和 memory/artifact backends，它就是平台级重写，effort 与切换风险都会是高。

### 4.4 容易欠下的新技术债

1. **Framework leakage**：公共 API、数据库或领域模型直接依赖 ADK Event/Context/State。
2. **Dual source of truth**：ADK session state 与领域表双写，却没有明确所有权和事务策略。
3. **Retry amplification**：node retry、HTTP retry、上游队列 retry 叠加；费用和副作用次数失控。
4. **Provider illusion**：把 LiteLLM 的统一接口误当成功能完全等价，漏测 tools/schema/streaming。
5. **Custom-agent trap**：用 `_run_async_impl` 复制旧引擎，绕开 graph runtime、内置 resume 与未来主路径。
6. **Version drift**：不精确 pin 2.x、不读 release notes、不跑 event/resume regression suite。
7. **GCP coupling without payoff**：只采用部分托管 services，却仍保留大部分自管 adapters，形成两套运维面。

## 5. 建议的最小迁移验证（不是全面重写）

选择一条包含“并行 research → typed 汇合 → synthesis → 流式输出 → 中断恢复”的代表性路径，做 production-shaped spike：

1. 用 ADK 2.x `Workflow` 重建，但保留现有领域 models、数据库和公共 API envelope。
2. 同时跑 Gemini 与实际主 provider，覆盖 tool + structured output + streaming。
3. 对每个关键节点注入 timeout、429/5xx、进程退出、重复 resume 与 parallel partial failure。
4. 验证工具幂等、event ordering、state ownership、SSE replay、成本/调用上限和 trace redaction。
5. 用相同 golden conversations 对旧/新实现做 final output、tool trajectory 与领域 invariant 对比。

Spike 只有在以下证据齐全时才应进入全面迁移估算：关键语义 parity；无未定义双写；副作用重复安全；公共 API 无上游泄漏；主模型 provider 能力通过；升级/回滚路径可演练。
