# PydanticAI 1.93.0 → 2.39.0 migration 研究底稿

> 日期：2026-09-05
> 范围：当前仓库的 PydanticAI 依赖和调用点；不修改 LangGraph Agent Patterns spec，不引入新架构
> 证据原则：只使用 Pydantic/PyPI/GitHub 官方一手资料，并将滚动的 `main` 文档与锁定到 `v2.39.0` 的源码事实分开。

## 1. 结论

**目标版本应精确锁定为 `pydantic-ai==2.39.0`；迁移时使用最新 V1 patch `pydantic-ai==1.107.5` 作为 bridge。**

- PyPI 将 2.39.0 标记为最新稳定发行版，于 2026-09-04 发布，并给出与 `v2.39.0` tag 对应的可验证源码 provenance：[PyPI 2.39.0](https://pypi.org/project/pydantic-ai/2.39.0/)。
- 1.107.0 是 2026-06-10 发布、V2 最后一个 beta fork 所基于的 V1 代码线；此后官方又发布了 V1 maintenance patches，最后一个是 2026-08-14 的 `1.107.5`：[PyPI 1.107.5](https://pypi.org/project/pydantic-ai/1.107.5/)、[GitHub v1.107.5](https://github.com/pydantic/pydantic-ai/releases/tag/v1.107.5)。官方 V2 upgrade guide 要求先升到最新 V1 并清理 deprecation warnings，再进入 V2：[官方 upgrade guide](https://github.com/pydantic/pydantic-ai/blob/main/docs/changelog.md#upgrade-guide)。
- 不建议从 1.93.0 直跳 2.39.0。直跳并非技术上不可行，但会丢失最新 V1 中专门为 V2 迁移提供的弃用警告路径。
- 本仓库不需要双版本兼容层、adapter framework 或 feature flag。两个短切换步骤即可：`1.93.0 → 1.107.5` 清 warning，然后 `1.107.5 → 2.39.0` 做一次性 API 替换和行为锁定。

**当前不能直接升版。** 至少有五类确定的 breaking/behavioral changes：

1. `history_processors=` 已被 `ProcessHistory` capability 取代。
2. 流式结果的 `usage()` 变为 `usage` property；`new_messages()` 仍是方法。
3. `MCPServer*` transport classes 收敛到 `MCPToolset`，而 `MCP(url=...)` 在 V2 默认改为 local-only。
4. GCP Vertex 用法从 `GoogleProvider(project=...)` 迁到 `GoogleCloudProvider`。
5. Agent 默认 `end_strategy` 从 `early` 改为 `graceful`，可改变同一 model response 中 output tool 与 function tool 并存时的工具执行轨迹。

## 2. 版本与证据基线

### 2.1 仓库现状

`pyproject.toml` 仅给出下限 `pydantic-ai>=1.93.0`，`uv.lock` 实际锁定：

| 包 | 当前 lock |
|---|---:|
| `pydantic-ai` | 1.93.0 |
| `pydantic-ai-slim` | 1.93.0 |
| `pydantic-graph` | 1.93.0 |
| `pydantic-evals` | 1.93.0 |
| `openai` | 2.36.0 |

2.39.0 官方 wheel metadata 要求同版的 `pydantic-ai-slim==2.39.0`，并在对应 extra 中带入同版 `pydantic-graph` / `pydantic-evals`；OpenAI extra 要求 `openai>=3.8.0`。因此 lock 刷新会同时带来 OpenAI Python SDK 的 major upgrade，不能只验证 PydanticAI import。

推荐在迁移 spec 中使用精确目标 `pydantic-ai==2.39.0`，然后由 `uv.lock` 锁定完整闭包；不保留 `>=1.93.0` 这种无上界声明。PydanticAI 的版本策略明确说明 minor release 也可以带来 breaking changes，这更支持应用端精确 pin：[官方 version policy](https://github.com/pydantic/pydantic-ai/blob/main/docs/version-policy.md)。

### 2.2 如何阅读本文中的“官方文档”

- 迁移和概念决策来自官方 [V2 migration map](https://github.com/pydantic/pydantic-ai/blob/main/docs/migration.md)、[retries](https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md)、[toolsets](https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md)、[MCP capability](https://github.com/pydantic/pydantic-ai/blob/main/docs/capabilities/mcp.md)、[message history](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md) 和 [testing](https://pydantic.dev/docs/ai/guides/testing/) 文档。
- 上述 `main` 文档是滚动的。文中关于构造函数签名、property/method 形态和 provider 类的断言，另外对照了 PyPI 签名的 2.39.0 wheel 和官方 [`v2.39.0` source tag](https://github.com/pydantic/pydantic-ai/tree/v2.39.0)。

## 3. 仓库调用点地图与处置

### 3.1 Agent 构造、retry 与 end strategy

当前仓库的 Agent 构造入口主要是：

- `app/core/model_registry.py:80-141` 的 `ModelRegistry.create_agent()`；
- `app/services/handlers/agent.py:265-274` 的 tenant Agent 直接构造；
- `app/agents/router_agent.py:40-62` 的 Router Agent 直接构造。

其他使用方位于 `intent_recognition.py`、`query_understanding.py`、`rag_answer.py`、`refine_question.py`、`router_agent.py` 内部 sub-agent、`agents/tools.py` 内部 agent、`langgraph_v2/{answer,groundedness,question_refinement}.py` 和 `citation_extractor.py`，都通过 registry 入口构造。

#### retry 决策

V2 将 Agent retry 配置收敛为：

```python
retries={"tools": 1, "output": 1}
```

- `tools` 与 `output` 是两个独立限额；传入整数会同时设两者，字典中缺失的 key 默认为 1。
- tool retry 是“每个 tool”的连续失败计数，成功后重置；它不是跨 tool 的全局原子预算。
- `ToolFailed` 是模型可见的适应轮，**不消耗 retry 预算**；若业务需要限制总模型请求，应用 `UsageLimits(request_limit=...)`，不能用 `retries=0` 假设所有修正轮都被关闭。详见官方 [retry layers](https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md)。

仓库当前没有 `output_retries=` 或显式 `retries=`，V1/V2 对这两个 Agent-level 限额的默认都相当于 1。迁移不应借机改变重试策略，但应在三个构造入口显式传入 `retries={"tools": 1, "output": 1}`，避免以后的库默认漂移。

#### `end_strategy` 决策

V2 默认从 `early` 改为 `graceful`。当模型在同一 response 中同时返回成功 output tool 和 function tool calls 时，`graceful` 会继续执行 function tools；V1 的 `early` 会在成功输出后尽快结束。

本次是纯迁移，推荐在三个构造入口显式设置 `end_strategy="early"`。若以后希望采用 graceful，应作为独立行为变更，通过 tool trajectory 评测后切换。

#### generic 类型

V2 的无 dependencies 默认类型从 `None` 收紧为 `object`。将以下静态标注一次性改为 `Agent[object, ...]`：

- `app/core/model_registry.py:97,117`；
- `app/services/handlers/llm.py:205,210,215`；
- `app/agents/intent_recognition.py:40`、`query_understanding.py:184`、`refine_question.py:64`。

这是类型对齐，不是运行时 dependencies 重构。

### 3.2 History processors

确定的 breaking change：`Agent(history_processors=[...])` 改为 `ProcessHistory` capabilities。官方 migration map 给出了直接替换：[V2 migration — history processors](https://github.com/pydantic/pydantic-ai/blob/main/docs/migration.md#history-processors)。

受影响处：

- `app/agents/intent_recognition.py:55`
- `app/agents/query_understanding.py:192`
- `app/agents/rag_answer.py:89`
- `app/agents/refine_question.py:79`
- `app/agents/router_agent.py:46`

目标形态：

```python
from pydantic_ai.capabilities import ProcessHistory

capabilities=[
    ProcessHistory(trim_history(20)),
    ProcessHistory(filter_thinking()),
]
```

`ProcessHistory` 按 capability 顺序修改 `request_context.messages`，所以保留当前“先 trim，后 filter thinking”的顺序。`rag_answer` 和 Router 已可能有其他 capabilities，应把两个 `ProcessHistory` 追加到同一 list，不再存在第二套 history 适配层。

`app/skills/capability.py` 的自定义 `SkillsCapability(AbstractCapability)` 使用的 `get_instructions`、`get_toolset`、`for_run` hooks 在锁定的 2.39.0 中仍存在；`FunctionToolset(functions)` 构造也仍接受当前带 `RunContext` 的 callable。这部分不需要重写。

### 3.3 Usage、messages 与 streaming

#### `usage` 是 property，`new_messages()` 仍是 method

2.39.0 锁定源码中 `StreamedRunResult.usage` 是 `@property`，`new_messages()` 仍是方法：[`v2.39.0 result.py`](https://github.com/pydantic/pydantic-ai/blob/v2.39.0/pydantic_ai_slim/pydantic_ai/result.py)。

必改的 `stream.usage()` 调用：

- `app/services/citation_extractor.py:179`
- `app/services/handlers/agent.py:313,352`
- `app/services/handlers/llm.py:271,295,322,387`

将它们改为 `stream.usage`，但不改 `stream.new_messages()`。所有流式 fake 也要从 `def usage(...)` 改为 property 或直接属性，当前位置包括：

- `tests/unit/test_agent_handler_modes.py:63`
- `tests/unit/test_query_resolver_handler.py:46`
- `tests/unit/test_query_understanding_handler.py:49`
- `tests/unit/test_llm_answer_aggregation.py:46`
- `tests/unit/test_langgraph_v2_answer.py:57`
- `tests/integration/test_workflow_flows.py:96`
- `tests/integration/test_langgraph_v2_uvicorn_disconnect.py:66`
- `tests/integration/test_langgraph_v2_groundedness.py:457`

`app/langgraph_v2/model_usage.py:9-17` 现在专门探测 callable，因此在 V2 会把真实 usage 误当为空字典。不要做双版本 introspection，直接读 `result.usage`，并对 dataclass 做 `asdict()`。

`RunUsage` 在 V2 包含更多 token detail 和 best-effort cost。`app/services/flow_context.py:72-86` 现在手工累加四个字段，会丢失新字段。最小替换是 `self.total_usage.incr(usage)`；这是 2.39.0 `RunUsage` 提供的官方原语：[`v2.39.0 usage.py`](https://github.com/pydantic/pydantic-ai/blob/v2.39.0/pydantic_ai_slim/pydantic_ai/usage.py)。

#### 流式 API

`stream_output()`、`stream_text()` 和 `get_output()` 在 2.39.0 仍存在，所以 `app/langgraph_v2/answer.py` 的 structured-output 流不需要改模式。

但 `app/services/handlers/llm.py:375-382` 有一个既有正确性风险：`stream.stream_text()` 默认 `delta=False`，返回递增的累计 snapshot；当前代码却将每个 snapshot append 再 `"".join`，并将整个 snapshot 作为 token 发送，会重复答案。现有 fake 返回 delta，因而没有捕住。

推荐像 `AgentHandler` 一样记录 `previous_text`，对 snapshot 取 suffix，最终答案取最后一个 snapshot。不建议简单改为 `delta=True`：官方 streaming 文档说明 delta 模式下最终 result message 不会被加入 messages，而本仓库需要 `new_messages()` 做会话持久化：[官方 streaming output](https://pydantic.dev/docs/ai/output/#streaming-output)。这一项应作为迁移 gate 内的小型 bug fix，但不扩大为 streaming 重写。

### 3.4 Message history 与 checkpoint 兼容

PydanticAI 官方对会话历史的 JSON 边界是 `ModelMessagesTypeAdapter.dump_json()` / `validate_json()`：[官方 message history](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md)。

`app/memory/session_store.py` 读路径正确使用 `validate_json()`，但写路径用的是泛化 `pydantic_core.to_json(messages)`。迁移时应改为：

```python
raw = ModelMessagesTypeAdapter.dump_json(messages)
```

2.39.0 仍保留 V1 message `part_kind` wire discriminator，且 usage 字段保留 `request_tokens` / `response_tokens` validation aliases，因此“V1 JSON → V2 读取”是有官方源码依据的前向兼容路径。但官方并未承诺“V2 写入的历史可由 V1 完整回读”。因此发布语义应是：

1. 切换前保存一份 V1 真实消息历史 fixture，在 2.39.0 下验证结构和语义；
2. 切换开始后把消息存储视为单向升级；
3. 若生产环境已有非 InMemory 实现，切换前做 raw store 备份；回退 V1 时同时回滚到该备份，不尝试在原存储上双写/双读。

仓库同时提供 `InMemorySessionStore` 和由 `SESSION_STORE_URL` 选择的
`RedisSessionStore`。不需要在 POC 内建立 history migration job，但若部署
配置启用 Redis，应在切换前备份该 key namespace；若 V2→V1 反向 fixture
不兼容，回滚时恢复切换前备份，不清空用户历史。

LangGraph v2 checkpoint state 存储的是 LangChain `BaseMessage`/仓库 Pydantic model 及 usage dict，不是 PydanticAI `ModelMessage` 或 `RunUsage` object。`app/langgraph_v2/conversation_context.py` 会在请求本地投影为 PydanticAI messages。因此本次升级**没有 checkpoint schema migration**；但仍需一条升级前 checkpoint 的 resume smoke，以验证 import/type 变化没有间接影响 graph resume。

### 3.5 MCP：local toolset 与 provider-native capability 是两条路径

#### 低层 MCP toolsets

`app/core/mcp.py` 使用的 `MCPServerSSE`、`MCPServerStreamableHTTP`、`MCPServerStdio` 在 V2 被移除，官方要求迁到 `MCPToolset`：[V2 migration — MCP](https://github.com/pydantic/pydantic-ai/blob/main/docs/migration.md#mcp)。

最小目标映射：

```python
from fastmcp.client.transports import StdioTransport
from pydantic_ai.mcp import MCPToolset

# HTTP / SSE: FastMCP 从 URL 推断 transport
toolset = MCPToolset(
    cfg.url,
    max_retries=1,
    tool_error_behavior="retry",
).prefixed(cfg.name)

# stdio
toolset = MCPToolset(
    StdioTransport(command=cfg.command, args=cfg.args, env=cfg.env),
    max_retries=1,
    tool_error_behavior="retry",
).prefixed(cfg.name)
```

V1 MCP server 默认 `max_retries=1`、init timeout 5s、read timeout 300s。V2 `MCPToolset.max_retries=None` 表示运行时继承 Agent tool retry；如果 Agent 也被显式锁定为 1，两者结果等价。为减少隐式联动，底层 MCP 建议仍显式 `max_retries=1`。`tool_error_behavior="retry"` 是 V2 默认，显式写出用于锁定当前“工具错误返给模型修正”语义。

不应在未看到实际运维 SLA 前自行改变 5s/300s timeout。V2/FastMCP 构造器的默认也是该数值，可保留默认，并在 MCP smoke 中测连接失败与 tool error，而不再造一层 timeout policy。

`prefixed(cfg.name)` 是 V2 toolset wrapper，用于取代 `tool_prefix=cfg.name`：[官方 toolset composition](https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#prefixing-tool-names)。

#### provider-native MCP capability

`app/services/handlers/agent.py:194-203` 使用 `MCP(url=..., id=..., allowed_tools=...)`。V1 是 provider-native preferred，必要时 local fallback；V2 中只有在 `native=True` 时才保留该行为，否则 `MCP(url=...)` 默认只走 local。因此本仓库为保持 V1 语义，应改为：

```python
MCP(
    url=mcp_cfg.url,
    id=mcp_name,
    allowed_tools=mcp_cfg.allowed_tools,
    native=True,
)
```

详见官方 [MCP capability migration](https://github.com/pydantic/pydantic-ai/blob/main/docs/migration.md#mcp-capability)和 [MCP capability docs](https://github.com/pydantic/pydantic-ai/blob/main/docs/capabilities/mcp.md)。必须用一个支持 native MCP 的真实模型和一个不支持的模型各做一次 canary，验证 native 和 local fallback 都可达。

两条 MCP 路径不应被统一成一个仓库 wrapper：`app/core/mcp.py` 是 local toolset，`AgentHandler` 是 model capability，它们有不同的执行边界和 fallback 语义。

### 3.6 Azure/OpenAI 与 Google Cloud provider

#### Google Cloud

`app/core/model_registry.py:205-216` 现在导入 `pydantic_ai.providers.google.GoogleProvider` 并传 `project=`。V2 将 Vertex/Google Cloud 认证路径拆到：

```python
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider(
    project=project_id,
    http_client=...,
)
```

`GoogleModel` 和当前 `GoogleModelSettings/google_thinking_config` 仍可用。详见官方 [Google provider migration](https://github.com/pydantic/pydantic-ai/blob/main/docs/migration.md#google)和锁定的 [`GoogleCloudProvider` source](https://github.com/pydantic/pydantic-ai/blob/v2.39.0/pydantic_ai_slim/pydantic_ai/providers/google_cloud.py)。

#### Azure/OpenAI

`OpenAIChatModel` 与 `AzureProvider(azure_endpoint=..., api_key=..., http_client=...)` 在 2.39.0 仍存在，不需要换成 Responses model。V2 中裸 model string `openai:...` 默认变为 Responses API，但本仓库明确构造 `OpenAIChatModel`，所以不受这一默认影响。

风险来自依赖闭包：2.39.0 OpenAI extra 要求 OpenAI SDK 3.8.0+，而当前 lock 是 2.36.0。需在真实 Azure endpoint 上验证：普通 text output、structured output、streaming、tool call、reasoning effort，以及用量字段。这是 canary，不是请求为 OpenAI SDK 建双版本 wrapper。锁定类签名见 [`AzureProvider` source](https://github.com/pydantic/pydantic-ai/blob/v2.39.0/pydantic_ai_slim/pydantic_ai/providers/azure.py)。

#### `httpx` 与 `httpx2`

2.39.0 provider 层内部已使用 `httpx2`，但仍显式接受 legacy `httpx.AsyncClient`，并发出一个预定在 V3 移除的弃用警告。当前 `HttpClientPool` 还同时服务仓库其他 HTTP/GCP 调用，本次不应把整个 pool 重写为 `httpx2`。

最小且干净的处置是在 `HttpClientPool` 中新增一个专供 PydanticAI provider 的 `httpx2.AsyncClient` getter，保持原 `get()` 不动，并由 Azure/GoogleCloud builder 使用新 getter。若当期不愿引入这个小改动，V2 仍能运行，但必须将该唯一已知 PydanticAI deprecation warning 记为 V3 debt；不应将所有 warning 整体忽略。

### 3.7 Pydantic Evals 和测试

仓库当前没有 `pydantic_evals` import 或 runtime usage；`pydantic-evals==1.93.0` 只是 `pydantic-ai` extra 带来的 transitive package。因此本次只需随 lock 同步到 2.39.0，**不要借升级新建 eval framework**。

若后续开始使用 Pydantic Evals，V2 需注意：`Dataset` 需要 `name`，`evaluate()` 参数改为 keyword-only，`Evaluator.name` 改为 `get_serialization_name()`，`evaluation_name/evaluator_version` class attributes 改为 methods。这些并非当前代码改动，仅作未来边界记录。数据集 wire format 参见官方 [dataset serialization](https://ai.pydantic.dev/evals/how-to/dataset-serialization/)。

官方 testing guide 推荐用 `TestModel` / `FunctionModel`、`Agent.override()`、`capture_run_messages()`，并在测试环境设 `models.ALLOW_MODEL_REQUESTS = False` 防止意外访问真实 provider：[官方 testing guide](https://pydantic.dev/docs/ai/guides/testing/)。本仓库已有手写 fake，可保留；只需：

1. 修正 fake 的 `usage` 形状；
2. 在 test bootstrap 全局禁用真实 model requests；
3. 用少量 `FunctionModel` 覆盖 retry/end-strategy/trajectory，而不重写全部 fake。

V2 的 `capture_run_messages()` 会捕获中断运行并以 `state="interrupted"` 表示，本仓库当前没有该 API 调用，无需迁移。

## 4. 建议的最小迁移顺序

### Gate A：1.93.0 → 1.107.5 bridge

1. 将依赖精确 pin 到 `pydantic-ai==1.107.5`，重新 lock。
2. 启动和测试时将 PydanticAI deprecation warnings 提升为 error，清理所有 V2 相关 warning。
3. 导出一份真实 V1 `ModelMessagesTypeAdapter.dump_json()` history fixture，以及当前 provider/tool trajectory 的基线记录。
4. 不改变 retry/end-strategy 业务语义。

### Gate B：1.107.5 → 2.39.0 机械 API 替换

1. Pin `pydantic-ai==2.39.0`，重生 lock，确认 `pydantic-ai-slim/pydantic-graph/pydantic-evals` 同版。
2. `history_processors` → `ProcessHistory`。
3. `stream.usage()` → `stream.usage`；`new_messages()` 不变；修正 usage fake 和 `model_usage_payload()`。
4. `MCPServer*` → `MCPToolset`；`tool_prefix=` → `.prefixed()`；capability 路径补 `native=True`。
5. `GoogleProvider(project=...)` → `GoogleCloudProvider(project=...)`。
6. `Agent[None, ...]` → `Agent[object, ...]`。
7. 在三个 Agent 构造入口锁定 `retries={"tools": 1, "output": 1}` 与 `end_strategy="early"`。
8. Session history 写路径改用 `ModelMessagesTypeAdapter.dump_json()`；usage 累加改用 `RunUsage.incr()`。
9. 修正 `LLMHandler` 将 snapshot 当 delta 的问题。

### Gate C：行为与 provider 验证

1. 先跑 deterministic CI，再跑 Azure/GCP/MCP canary；canary 不进默认 CI。
2. 通过后单向切换。若必须回退，回退代码和 lock，且使用切换前 history 备份；不在应用中保留两套 API 分支。

## 5. 验证矩阵

| 层级 | 场景 | 期望/失败门槛 |
|---|---|---|
| Dependency | `uv lock` / `uv sync --frozen` | 四个 Pydantic package 均为 2.39.0；无未解冲突；OpenAI SDK 满足 3.8.0+ |
| Static | import + pyright | 无 `MCPServer*`、`history_processors`、`GoogleProvider(project=...)`、`Agent[None,...]` 残留 |
| Unit | history processor 顺序 | trim 后 filter thinking；同一输入在 V1 baseline/V2 得到等价 model history |
| Unit | usage API | stream/result usage 不为空；`new_messages()` 仍可读；details/cost 累加不丢失 |
| Unit | retry | tool schema error、`ModelRetry`、output validation 分别不超过显式 1 次；证明 `ToolFailed` 不被错认为 retry 预算 |
| Unit | end strategy | 同一 response 同时含 output tool/function tool 时，`early` 不额外执行 function tool |
| Unit | text stream | fake 返回 `h`/`he`/`hello`，SSE 仅发 `h`/`e`/`llo`，最终 answer 仅为 `hello`，且最终 message 存在 |
| Unit | MCP prefixes | HTTP/SSE/stdio 都暴露 `<server>_<tool>`，无同名冲突 |
| Unit | MCP error | server ToolError 在 `tool_error_behavior="retry"` 下返给模型，到 `max_retries=1` 后终止；连接失败不伪装成成功 |
| Unit | custom capability | Skills instructions/toolset/deps 在每个 run 仍正确注入 |
| Serialization | V1 fixture → V2 | `ModelMessagesTypeAdapter.validate_json()` 成功；role/part/content/tool-call ID/usage 语义不变 |
| Checkpoint | 升级前 LangGraph checkpoint resume | graph 可 resume；无 PAI class 序列化错误；usage dict 仍可用 |
| Integration | Agent/LLM/RAG flows | 结构化输出、citation、history、stream 全通过 |
| Canary | Azure OpenAI | text/structured/stream/tool/reasoning/usage 各一次，无 SDK 3.x 认证或 response-shape 回归 |
| Canary | Google Cloud | ADC/project/location 可用，thinking config 与 stream/usage 正常 |
| Canary | MCP local | 一个 HTTP/SSE 服务和一个 stdio fixture 均完成 list/call/close |
| Canary | MCP native/fallback | 支持 native 的 provider 走 native；不支持的 provider 退到 local；`allowed_tools` 仍生效 |

所有 deterministic test 应启用 `models.ALLOW_MODEL_REQUESTS = False`。只有显式的 provider canary job 可打开真实请求。

## 6. 实测证据

在不修改仓库的前提下，使用隔离的临时 venv 安装官方 `pydantic-ai==2.39.0` 和当前项目依赖，得到：

- 针对 Agent handler、query understanding/resolution、answer aggregation、LangGraph v2 answer、citation extractor 和 workflow 的选定测试：**51 passed**。
- 另 10 个 LangGraph v2 integration tests 在 setup 阶段因未设 `LANGGRAPH_V2_TEST_DATABASE_URL` 而中止，未进入待测代码；这不能计为 pass/fail。
- 全仓 `pytest --collect-only` 在收集到 239 个测试后，被根目录的两个实验脚本中止：
  - `test_stream_union.py` 的 `TestModel(custom_output_text=...)` 与 union output tool 组合不再兼容；
  - `test_union2.py` 让 `TestModel` 调用了一个未注册的 `CL` tool。

这两个文件在 module import 时直接 `asyncio.run(main())`，本质是实验脚本而不是可收集测试。最小处置是将执行放入 `if __name__ == "__main__":` 或移出 pytest 发现路径，然后只保留真正需要的 union-output 契约测试。不应为了兼容这两个脚本而改动生产 Agent API。

## 7. 非目标和取舍

- 不建双版本 adapter/shim；不保留 `callable(result.usage)` 这类分支。
- 不在本次改成 OpenAI Responses API、不换模型、不调整 prompt。
- 不改 retry 数量、MCP timeout 或 end behavior；只显式锁定 V1 行为。
- 不把 config 中尚未接入 runtime 的 `UsageLimits` 借机扩展成全局 budget 系统。
- 不引入 Pydantic Evals dataset；当前只需保证 transitive package 可 import。
- 不修改 LangGraph checkpoint schema，不引入事件溯源或消息双写。
- 不把此次升级与 LangGraph Agent Patterns 的新 Agent 实现捆绑。应先让当前调用点通过 V2 gate，新 pattern 再只针对 V2 API 实现。

## 8. 待环境验证项（不是待设计决策）

已能从仓库意图决定的事项都已在上文收敛。仍需在实际环境获得证据的只有：

1. 当前 tenant 实际配置的 MCP URL 分别是 SSE 还是 Streamable HTTP，以及 server 是否依赖特定 FastMCP 2/3 行为。
2. 当前 Azure deployment 和 Google model 名称是否被 OpenAI SDK 3.8+/Google provider 实际接受；这必须由带凭据 canary 确认。
3. 生产是否存在仓库外的 `BaseSessionStore` 持久实现。如果有，需在切换前执行 raw backup 和 V1 fixture 验证；如果没有，该步骤可跳过。
4. 是否接受在 V2 期间暂时保留 legacy `httpx.AsyncClient` 的 V3 deprecation warning。本文推荐增加一个小型 PydanticAI-only `httpx2` client getter，但该决策需以当前应用的 HTTP client 观测/代理配置要求为准。

## 9. 最终判定

**Migration design: PASS with required gates.**

技术路径是清晰、最小且可实施的：先 1.107.5 bridge，再一次性切到 2.39.0，不建兼容框架。但在第 3–5 节所列 API 替换、行为锁定和第 5 节验证矩阵完成前，**当前代码对直接更新 lock 的就绪度是 FAIL**。
