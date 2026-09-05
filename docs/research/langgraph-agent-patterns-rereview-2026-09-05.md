# LangGraph Agent Patterns 独立复审（2026-09-05）

## 结论

**FAIL。P0 = 0，P1 = 5，P2 = 3，unresolved = 8。**

主架构方向是合理的：`Send` 批量 fan-out、单步 barrier promotion、reducer-backed staging、accepted-attempt 数据隔离、普通 typed `ToolReturn` 表示预期不可用、以及 `finalize_state -> publish` 的 state-before-first-answer 顺序，都符合官方运行时语义。本轮没有发现需要引入 Run hard deadline、PostgreSQL advisory lock、SSE replay、同-thread admission、跨分支原子预算、事件存储或额外 Artifact/Gap repository 的理由。

但当前 spec 仍有五个会影响正确性或可实施性的 P1：它把当前仓库不存在的同-thread 串行保证写成既有事实；“关闭 provider retry”没有落实到当前 Azure/OpenAI client 构造边界；取消语义与先提交后发布互相矛盾；零 Evidence 路径可能发布 `completion_status=complete`；以及 Specialist outer retry/`TaskFailed` 的闭合集合没有定义且错误地允许重试确定性的 prior-context overflow。

## 范围与版本口径

- 审查对象：[spec.md](/Users/rory/mycode/ai/agent-kms/.scratch/langgraph-agent-patterns/spec.md)。只审 Agent Patterns，不审 PydanticAI V1 -> V2 migration，也不审 issue 编排。
- POC/MVP 约束按题设固定：仅 read/fetch/calculate；不新增 Run hard deadline、whole-Specialist timeout、advisory lock、SSE replay、同-thread admission 或全局原子预算。
- 当前锁定实现事实来自 [pyproject.toml](/Users/rory/mycode/ai/agent-kms/pyproject.toml:7) 与 [uv.lock](/Users/rory/mycode/ai/agent-kms/uv.lock:3564)：LangGraph `1.1.10`、`langgraph-checkpoint` `4.2.0`、PostgreSQL saver `3.1.2`、PydanticAI `1.93.0`。线上 PydanticAI 文档已经描述 V2 API；凡属版本敏感结论，下文明确区分“锁定源码事实”和“当前线上文档事实”。
- `ToolFailed` 和 V2 retries 字典的迁移不属于本轮。这里审查的是不随该迁移改变的行为边界：预期不可用必须成为正常 Tool 返回，provider/transport retry 必须独立关闭，异常与取消不能被伪装成业务结果。

## 一手资料确认的运行时事实

### LangGraph

1. LangGraph 以 superstep 执行；同一 superstep 的节点并行运行，更新到下一步才可见。`Send` 是官方 map-reduce/orchestrator-worker 的动态 fan-out 机制；并行更新的完成顺序不稳定，因此使用稳定 ID 的交换律 reducer，再在 barrier 按稳定业务键排序，是正确做法。[Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api#send)、[orchestrator-worker 示例](https://docs.langchain.com/oss/python/langgraph/workflows-agents#creating-workers-in-langgraph)、[并行节点语义](https://docs.langchain.com/oss/python/langgraph/use-graph-api#run-graph-nodes-in-parallel)
2. 并行 superstep 中若一个分支抛异常，该步不会形成完整 state checkpoint；已经成功的 sibling 写入会以 pending writes 持久化，供恢复时复用。由 barrier 单独验证完整 manifest 并一次 promotion，能阻止半个 batch 进入 canonical accepted state。[Persistence: pending writes](https://docs.langchain.com/oss/python/langgraph/persistence#pending-writes)
3. reducer 字段返回空容器不会清空旧值；应使用 `Overwrite(empty_value)`。`Overwrite` 只用于 reducer-backed channel，同一步多个 overwrite 会失败。spec 的“initializer 唯一写 reset；scalar 写普通 `None`”方向正确。[Graph API: resetting reducer fields](https://docs.langchain.com/oss/python/langgraph/graph-api#resetting-a-reducer-field)、[LangGraph 1.1.10 `Overwrite` 源码](https://github.com/langchain-ai/langgraph/blob/1.1.10/libs/langgraph/langgraph/types.py#L831-L856)
4. `durability="sync"` 的保证是：当前 step 的 checkpoint 在下一 step 开始前写完。因此不发 answer 的 `finalize_state` 成功后，下一步 `publish` 才发 answer，可以保证 state-before-first-answer；它不提供 PostgreSQL 与网络之间的原子事务或 exactly-once delivery。[Durability reference](https://reference.langchain.com/python/langgraph/types/Durability)、[LangGraph 1.1.10 固定源码](https://github.com/langchain-ai/langgraph/blob/1.1.10/libs/langgraph/langgraph/types.py#L85-L90)
5. `recursion_limit` 计算的是 Graph supersteps，不是节点内部的 PydanticAI model request、Tool call 或 outer attempt。[Graph API: recursion limit](https://docs.langchain.com/oss/python/langgraph/graph-api#recursion-limit)

### PydanticAI 与 provider SDK

1. `ToolReturn.return_value` 被序列化给模型；`ToolReturn.metadata` 只供应用读取，不发送给模型。把 bounded `ToolUnavailable` 放在 `return_value`、把受信 provenance 放在 metadata，并从 accepted run 的新消息提取，是官方 API 支持的用法。[Advanced Tool Returns](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#advanced-tool-returns)、[锁定 1.93.0 `ToolReturn` 源码](https://github.com/pydantic/pydantic-ai/blob/v1.93.0/pydantic_ai_slim/pydantic_ai/messages.py#L860-L884)
2. PydanticAI 会并发执行同一个 model response 中的多个 function tools；未捕获的异常会终止该 tool-processing step，并取消、drain 尚未完成的 siblings。反之，返回普通 `ToolReturn` 是成功完成的 Tool result，不触发 `ModelRetry` 或 `ToolFailed` 路径。因此 spec 的 binding-owned timeout -> typed return 设计确实能让 multi-hop/fan-out 继续。[Parallel tool calls](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#parallel-tool-calls--concurrency)、[锁定 1.93.0 `_call_tools` 源码](https://github.com/pydantic/pydantic-ai/blob/v1.93.0/pydantic_ai_slim/pydantic_ai/_agent_graph.py#L1784-L1883)
3. Tool/output retries、provider SDK retries、HTTP transport retries和 workflow retry 是互相独立、会相乘的层。PydanticAI `UsageLimits.request_limit` 只统计 agent model requests，看不到下层 wire attempts。[PydanticAI retries：layers 与 multiplication](https://pydantic.dev/docs/ai/core-concepts/retries/#retry-multiplication)
4. OpenAI 官方 Python SDK 对 connection error、408、409、429 和 5xx 默认自动重试两次；`max_retries=0` 才关闭。当前仓库 Azure provider 未传预构造 client，因此走该默认值。[OpenAI Python SDK retries](https://github.com/openai/openai-python#retries)、[model_registry.py](/Users/rory/mycode/ai/agent-kms/app/core/model_registry.py:178)
5. `capture_run_messages()` 的一个 context 只捕获其中第一次 `run*`；成功 run 应优先读 `result.new_messages()`，失败/取消 run 才需要逐 invocation 的 capture context。[PydanticAI agent API](https://pydantic.dev/docs/ai/api/pydantic-ai/agent/#capture_run_messages)、[testing guide](https://pydantic.dev/docs/ai/guides/testing/)

## Findings

### P1-1：spec 把不存在的 same-thread 串行化写成“既有保证”

**位置**：spec 129-135、734-738；当前实现 [api.py](/Users/rory/mycode/ai/agent-kms/app/langgraph_v2/api.py:343)、[main.py](/Users/rory/mycode/ai/agent-kms/app/main.py:32)、[rate_limit_middleware.py](/Users/rory/mycode/ai/agent-kms/app/core/rate_limit_middleware.py:92)。

**证据与风险**：`/v2/query/stream` 为同一 Tenant/Subject/mode/Conversation 派生稳定 `thread_id`，但 route 在 identity read-check 后直接返回 request-owned stream，没有按 `thread_id` 串行。全局 concurrency middleware 明确跳过该 v2 route；Tenant concurrency 只是按 Tenant/endpoint 计数，而且 middleware 在返回 `StreamingResponse` 时就会释放，不是一个 per-thread、覆盖完整 body iteration 的保证。LangGraph checkpointer 把一个 thread 定义为多次 run 的累计状态游标；raw OSS graph invocation 不会替 self-hosted FastAPI 提供 Agent Server 的 multitask admission。[LangGraph persistence: threads](https://docs.langchain.com/oss/python/langgraph/persistence#threads)

两个请求可同时通过可竞争的 `validate_checkpoint_request_identity()`，从同一旧 checkpoint 启动，各自 reset run-local state 并提交后续 checkpoint。`request_id` 不能充当 fencing token。结果可能是 Conversation Message 丢失、后提交覆盖前提交，或一个 Run 读取另一个 Run 的 active state。

**最小修复**：不新增本轮明确排除的 lock/admission。把“existing request-runtime guarantee”改为真实陈述：

> Overlapping Requests for one derived `thread_id` are unsupported in this POC. The current repository does not enforce their serialization. Agent-mode rollout therefore requires a separately owned, externally enforced and integration-tested per-thread serialization precondition; until that evidence exists, correctness is claimed only for non-overlapping Requests.

测试章节不能把它称为已验证 contract；应记录部署证据的 owner，或明确这是 POC 使用限制。若产品仍允许 public clients 重叠调用同一 Conversation，则这个 P1 无法仅靠文档关闭。

### P1-2：关闭 PydanticAI Tool/output retry 不会关闭当前 Azure/OpenAI provider SDK retry

**位置**：spec 81-83、107-110、273-277、800-815；当前实现 [model_registry.py](/Users/rory/mycode/ai/agent-kms/app/core/model_registry.py:178)。

**证据与风险**：spec 正确要求关闭 hidden provider/transport retry，但没有把它落成 provider-construction contract。当前 `AzureProvider(azure_endpoint=..., api_key=..., http_client=...)` 内部创建 OpenAI client；OpenAI SDK 默认 `max_retries=2`。因此一个 spec 所称的“一个 model request”最多产生三次 wire attempts；60 秒 timeout 也可能按 attempt 重置。`retries=0`/V2 的 `retries={'tools': 0, 'output': 0}` 都不影响这个层。

**最小修复**：在 Agent Patterns spec 中补版本中立但可验收的 provider 约束：Actor 使用的 provider client 必须显式关闭 SDK retry；Azure/OpenAI 当前实现通过预构造 `AsyncOpenAI`/`AsyncAzureOpenAI(max_retries=0, http_client=...)` 注入 provider，且不得给共享 HTTP client 安装 retrying transport。LangGraph 节点不配置 `retry_policy`，POC MCP adapter 也不配置 reconnect/call retry。增加 429、5xx 和 connection error contract tests，断言一个逻辑 model request 只有一个 wire attempt。具体 V1/V2 import/API 名由独立 migration spec 决定。

### P1-3：external cancellation 的“无 final assistant Message”与先提交后发布互相矛盾

**位置**：spec 402-405、635-647、752-758。

**证据与风险**：`finalize_state` 必须先把 canonical response、citation 和 assistant Message 以 sync durability 提交，下一 superstep 的 `publish` 才能开始。这一设计正确地实现 state-before-first-answer。但如果客户端在 finalize 已提交后、publish 开始前或 publish 中途断开，[`_RequestOwnedStreamingResponse`](/Users/rory/mycode/ai/agent-kms/app/langgraph_v2/api.py:61) 会取消 graph iterator；取消不能回滚已经成功的 PostgreSQL checkpoint。因此“external cancellation/disconnection ... no final assistant Message”不可能在所有时点成立，也与 spec 自己“不承诺 DB/network atomicity 或 replay”矛盾。

**最小修复**：把契约按 commit boundary 切开：

- finalize commit 前取消：无 canonical assistant Message、answer/citation/done frame；
- finalize commit 后取消：canonical assistant Message 可以存在，客户端可能收到零个或部分 answer frames；
- 仍然不提供 replay、exactly once 或跨 DB/network 事务。

增加三个取消注入测试：finalize 前、finalize commit 后但 publish 前、publish 中途。现有 blocked/failed final-state saver -> zero answer frames 测试应保留。

### P1-4：零 eligible Evidence 时，`IncompleteResearch` 可能为空并发布 `completion_status=complete`

**位置**：spec 358-397、423-427、743-745、826-831。

**反例**：第一轮 Coordinator 直接 `Finish`，或所有 accepted `TaskSucceeded` 均返回空 `evidence_ids` 且没有 DataGap。spec 没有规定 successful Finding 至少一个 Evidence ID。此时 Graph 正确跳过 Synthesis并生成 insufficient-Evidence answer，但 `IncompleteResearch` 的闭合集合只含 failed Task、accepted DataGap 与六种 structural reason；它为空。按 393-394，metadata 就不会被设为 `incomplete`。用户看到的正文说 Evidence 不足，机器状态却可表示 complete。

**最小修复**：采用最简单的代码拥有语义：把 `insufficient_evidence` 加入 structural reason enum；在 prepare-synthesis 发现 eligible Evidence 集合为空时加入该 reason，并令 `completion_status=incomplete`、`termination_reason=partial_results`（或明确选定的既有 wire reason）。不引入 completeness model 或 gap repository。增加 `Finish`-first、zero-Evidence success、以及只有 authoritative-negative Evidence 的三个反例测试。

### P1-5：outer retry/`TaskFailed` allowlist 不闭合，且确定性的 prior-context overflow 被错误列为可重试

**位置**：spec 203-221、290-329、330-338、358-364、800-815。

**证据与风险**：spec 多次使用“allowlisted failure”“eligible transient failure”，却没有列出 closed categories 或到当前 PydanticAI/provider exception 的映射。实现者无法在不写 broad catch 的前提下判断 model timeout、429/5xx、request-limit exhaustion、output-invalid、provider auth/config error、未知异常分别应 retry、`TaskFailed` 还是 fatal。与此同时，203-209 已要求 dispatch 前验证 bounded context，298-299 却说 prior-context overflow 可以消耗 Specialist outer retry。该 overflow 对同一个 Task 和同一组 accepted prior Results 是确定性的； fresh Agent run 不会改变输入大小，重试只能浪费 actor allowance，并可能把 Coordinator-owned invalid decision伪装成 Task failure。

**最小修复**：增加一个很小的 closed adapter table，不把技术原因暴露给 Coordinator：

| adapter category | outer retry | terminal behavior |
|---|---:|---|
| allowlisted transient model timeout/connection/429/5xx | 最多两次，受原 Task 累计限额约束 | exhausted 后 `TaskFailed` |
| allowlisted structured-output invalid | 最多两次，fresh run 接收 bounded deterministic validation feedback | exhausted 后 `TaskFailed` |
| actor-local request/Tool count exhausted | 否 | `TaskFailed` |
| expected Tool unavailable | 否 | typed `ToolReturn`，继续同一 run |
| cancellation、auth、config、invariant、programmer、unknown | 否 | re-raise，fatal Run |

具体 exception class 必须是显式集合并由 migration spec 固定版本映射；不得 `except Exception -> TaskFailed`。prior-context 的 canonical byte measurement 移到 Coordinator decision validation；超过 64 KiB 走同 round repair，若 validation 后执行时不一致则是 fatal invariant。增加每类 exact exception 与一个未知 subclass 的分类测试。

### P2-1：`recursion_limit=40` 的证明把 node 内 actor 调用误算成 Graph path

**位置**：spec 288、398-401、835-838。

`recursion_limit=40` 本身合理，但 Specialist 的三次 outer attempts 和 Coordinator/Synthesis adapter repair 都发生在一个 LangGraph node 内，不增加 superstep。当前措辞会让测试按 model call 数而非 Graph step 数证明错误的上界。

**最小修复**：按静态 node/superstep 路径枚举最长合法路径，并在测试中断言 `langgraph_step`/checkpoint history 小于 40；actor requests、Tools、outer attempts 用各自 adapter tests 证明。

### P2-2：`JsonPlusSerializer` 的“strict mode”没有给出当前锁定版本的无歧义配置

**位置**：spec 671-675、835-837；当前构造 [postgres.py](/Users/rory/mycode/ai/agent-kms/app/langgraph_v2/postgres.py:69)。

checkpoint `4.2.0` 中，无环境变量时 `JsonPlusSerializer()` 的 `allowed_msgpack_modules` 默认是 permissive `True`（警告但允许未注册类型）；`pickle_fallback=False` 只关闭 pickle，不能自动得到 strict MessagePack。官方安全公告也要求 self-hosted 手动启用 strict/allowlist。[固定 4.2.0 源码](https://github.com/langchain-ai/langgraph/blob/checkpoint%3D%3D4.2.0/libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py#L82-L119)、[GHSA-g48c-2wqr-h844](https://github.com/langchain-ai/langgraph/security/advisories/GHSA-g48c-2wqr-h844)

**最小修复**：写出构造值：`JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=None, allowed_msgpack_modules=None)`，再传给共享 `AsyncPostgresSaver(..., serde=serde)`；只有 round-trip 证明必需时增加 exact-symbol allowlist。测试同时验证 approved `BaseMessage`、JSON-native Agent state 能 round-trip，未知 dataclass/Pydantic type 与 pickle payload fail closed。

### P2-3：多次 repair/outer attempt 的 message-capture 测试需要逐 invocation context

**位置**：spec 687-689、800-815。

一个 `capture_run_messages()` context 只记录第一次 `run*`。若测试把最多三次 Specialist attempt 或两次 Coordinator/Synthesis invocation 包在同一 context 中，后续 attempt 的 metadata 泄漏、hidden retry 或消息数量都不会被观察到。

**最小修复**：规定每次 actor invocation/outer attempt 独立 capture；成功结果直接保存 immutable `result.new_messages()`，异常 attempt 用单独 `capture_run_messages()`，然后由 adapter-test harness 聚合快照。对每个 abandoned attempt 断言其 metadata 没进入 accepted result。

## 已通过的重点检查

- **Expected Tool unavailable / multi-hop / internal fan-out-in：PASS。** 普通 bounded `ToolReturn` 是正常 Tool result；binding-owned timeout 在异常离开并行 executor 前转换，成功 siblings 不会因为该预期失败被取消。spec 还正确禁止了 broad `except Exception`、`ModelRetry` 与 `ToolFailed`。
- **DataGap provenance 与 accepted-attempt isolation：PASS。** model 不能自行声明 gap；metadata 绑定 Run/Task/attempt/Tool call，accepted draft 后由 adapter 一对一派生；abandoned/failed attempt 不 promotion；冲突 provenance fatal；保守保留后续 fallback 已覆盖的 gap，避免引入 `coverage_key` 与 resolved/unresolved 推断。
- **Partial batch 与 pending writes：PASS。** branch 只写 immutable staged contribution；barrier 验证完整 manifest 并一次形成 `AcceptedBatch`；fatal sibling 只留下 ineligible pending writes，不形成半批 canonical state。Evidence bodies 不进入 checkpoint/pending writes。
- **Reducer reset：PASS。** 唯一 initializer 使用 `Overwrite` 清 reducer channel、普通值清 scalar，且任何 actor 都不能先于 initializer。
- **Bounds：除 P1-5/P2-1 外 PASS。** Tasks/Rounds、per-actor requests、Tool calls、bytes、Evidence cache、Calculation collection、Prepared Synthesis、final Markdown 与 recursion 都有 exact/one-over 测试要求；并行 aggregate usage 明确只是 telemetry/stop-future-work signal，符合只读 POC 的风险模型。
- **Publication：在修正 P1-3 的取消措辞后 PASS。** `finalize_state` 不发 answer、下一 sync-durable step 才 publish，确实实现 canonical-state-before-first-token；spec 也正确拒绝 exactly once 与 replay 承诺。
- **CONTEXT/ADR 一致性：除本报告 findings 外 PASS。** rolling coordination、Task/Specialist 边界、typed Tool unavailability、conservative incomplete disclosure、无 Run deadline、Synthesis 不决定 global completeness 均与 `CONTEXT.md`、ADR-0001/0004/0006/0007 一致。

## 建议的最小 spec patch 顺序

1. 先修 P1-3 与 P1-4：它们直接决定公开响应与 durable Conversation truth。
2. 再闭合 P1-5，并把 prior-context overflow 前移到 decision validation。
3. 在 provider construction/test 中关闭 SDK retry（P1-2）。
4. 对 P1-1 诚实标注当前缺失的部署前提；不得以本 spec 偷渡 lock/admission。
5. 最后补 P2 的 superstep 证明、serializer exact args 与逐 invocation capture。

完成这八项后，可以在不扩大 POC 架构的前提下重新判定为 PASS。

## 主审裁决

最终裁决仍为 **FAIL：P0 = 0，P1 = 5，P2 = 3，unresolved = 8**，
但对独立 reviewer 的一项 finding 作了替换，并收窄了一项测试语义：

1. **Dismiss 原 P1-1（same-thread admission）**。产品负责人已经明确给出
   “一个完整 `thread_id` 不会出现重叠 Request”的系统不变量；spec 129-135
   已将它声明为 prerequisite，并把 admission/locking 排除在本 POC 外。因此本轮
   不要求仓库再实现第二套 admission。这个 disposition 只接受该产品前提，
   不声称当前 FastAPI route 自身实现了串行化。
2. **新增 P1 替换原 P1-1：fatal error 的公开投影没有闭合**。spec 406-408
   要求公开 failure 使用稳定、脱敏的消息，但当前
   `app/langgraph_v2/api.py:199-200` 直接把 `str(error)` 写入 SSE。Agent fatal
   exception 可能携带 provider payload、endpoint 或 credential 片段。最小修复是
   在共享 HTTP/SSE 边界加入 allowlisted public-error projector；未识别异常统一
   输出固定 generic 文案，内部异常仍保持 fatal，绝不转成 `TaskFailed`。增加一个
   含敏感原文的异常测试，断言 SSE、`done` 和 assistant Message 均不泄漏原文。
3. **P2 serializer finding 的精确语义**：checkpoint `4.2.0` strict mode 对未知
   msgpack constructor/Pydantic type 的保证是“不 import/实例化该类型”，并不保证
   serializer 一定抛异常；它可能降级为普通参数或 `dict`。因此验收应断言未知类
   不被重建，且降级值随后被显式 checkpoint-state 读取/验证边界拒绝，不能成为
   有效运行状态。`pickle_fallback=False` 与
   `allowed_msgpack_modules=None` 仍必须显式配置。

最终计数由以下未解决项组成：provider/transport retry 的真实关闭、finalize
commit 前后取消语义、zero-Evidence incomplete 状态、Specialist failure/retry
闭集、fatal SSE 脱敏投影五个 P1；superstep 上界证明、serializer 精确配置与
typed-boundary 拒绝、逐 actor invocation message capture 三个 P2。
