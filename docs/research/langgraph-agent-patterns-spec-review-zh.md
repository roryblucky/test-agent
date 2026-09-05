# LangGraph Agent Patterns Spec 评审

日期：2026-09-04

> 后续决策：本 POC 已删除 Run 级墙钟截止、Research Cutoff clock、整个
> Specialist Task timeout 和 deadline contribution。下文涉及这些机制的原始
> review finding 已由现行 spec/ADR-0007 取代。

评审对象：`.scratch/langgraph-agent-patterns/spec.md`，并对照根领域词汇、
ADR-0001 至 ADR-0007、拆分后的 14 个 issue、仓库锁定的 LangGraph 1.1.10、
PydanticAI 1.93.0，以及截至评审日的 LangGraph/LangChain/PydanticAI 一手资料。

## 结论

整体架构方向合理，建议保留，但当前状态不应是 `ready-for-agent`，而应是
“需要解决阻塞性契约问题后再实施”。

最值得保留的选择是：

- 用静态 LangGraph 承载确定性控制流，只让模型提出有界的
  `DispatchBatch | Finish`；
- 用滚动 Coordinator--Specialist 批次表达 fan-out/fan-in 和多跳研究，
  不提前建设动态 DAG 编译器或 workflow DSL；
- 用 `Send` 做动态并行、用稳定业务 Task ID 和满足结合律/交换律/幂等性的
  reducer 收集结果，并在消费前按稳定键排序；
- 把 Conversation、Coordinator、Specialist、Synthesis 的上下文分开；
- 把授权、Research Scope、Evidence/Calculation 引用校验、最终发布门禁和
  计算执行留在确定性代码中；
- clarification 作为一次正常完成的对话轮次，而不是在尚无待处理工作时
  引入 `interrupt/resume`；
- 对无界循环、自动重试、模型生成公式、Skill 扩权和未验证 token 外泄都
  明确设限。

这些选择与 LangGraph 官方的
[orchestrator-worker / `Send` 模式](https://docs.langchain.com/oss/python/langgraph/workflows-agents)、
[Graph API 的状态与 reducer 语义](https://docs.langchain.com/oss/python/langgraph/graph-api)
以及 LangChain 的
[subagents 上下文隔离建议](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)
一致。这里采用的是基于官方原语构建的产品架构，不是 LangGraph 自动提供的
完整 supervisor 产品能力。

## 阻塞性发现

### [P1] 同一 checkpoint thread 的 Run 隔离、并发准入和字段重置没有形成契约

位置：`spec.md:40-59`、`spec.md:107-121`、`spec.md:463-478`。

Spec 让一个 Conversation 的多个 Run 共用由 Tenant、Subject、mode 和
Conversation UUID 派生的 `thread_id`，但没有定义两个请求重叠时是 reject、
enqueue、interrupt 还是 rollback，也没有给 `Run` 一个明确、可序列化的稳定
身份。LangGraph 官方把这类情况称为 double texting，并明确说明这些并发策略
属于 Agent Server，**不由 OSS LangGraph 自动提供**。本项目是自建 FastAPI
runtime，因此必须自己实现其中一种策略
（[官方说明](https://docs.langchain.com/langsmith/double-texting)）。

即使请求完全串行，当前“初始化时把 Agent-local 字段设为空”也不足以保证
重置。官方文档明确指出，带 merge reducer 的字段收到空 list/map 后仍会保留
旧值，必须使用 `Overwrite` 或等价的显式替换
（[官方 reducer reset 说明](https://docs.langchain.com/oss/python/langgraph/graph-api#resetting-a-reducer-field)）。
这会直接影响前一 Run 的 Coordination Rounds、Task/Outcome map、Artifact map、
错误和计数器是否泄漏到下一 Run。

建议在实施前明确：

1. 定义 `run_id`，最简单的是把已验证的稳定 `request_id` 作为 Run 身份，并在
   Task、Evidence、Calculation Artifact 的内容绑定中统一使用；
2. POC 选择一种同-thread并发策略。若没有持久队列，优先 fail-fast reject；
   多实例部署必须有跨实例准入机制，进程内锁不够；
3. 对每个 reducer-backed Agent-local channel 明确使用 `Overwrite(empty)`，或把
   一次 Run 的可变控制状态放进一个可整体替换的 request-scoped 容器；
4. 增加“顺序两个请求”“同 Conversation 并发两个请求”“相同 request ID
   并发重试”三类 PostgreSQL 测试，验证无重复执行、无状态串线和无旧字段残留。

### [P1] 并行 Specialist 下的全局 model/Tool/retry 硬上限无法靠完成后的 reducer 保证

位置：`spec.md:107-118`、`spec.md:153-162`、`spec.md:174-190`、
`spec.md:427-449`。

每个 PydanticAI Specialist 是独立 run。分支结束后再把 usage/counter 合并到
Graph state，只能记账，不能阻止多个并发分支同时看到“还剩 1 次调用”并各自
再启动一次。`max_concurrency=8` 只限制并发度，不等于 Run-wide 请求和 Tool
调用预算。Spec 已经正确承认 aggregate token/cost 不是硬上限，但对
model requests、Tool calls 和 retries 又声明为硬上限；缺少原子 reservation
后，这个声明不能成立。

建议二选一并写入契约：

- dispatch 前按 Specialist 的最大可能消耗预分配 branch-local quota，barrier
  后结算未用额度；或
- 使用 request-scoped、并发安全的 `try_reserve(kind, n)` 预算器，在每次模型、
  Tool、retry 开始前原子取许可，并把稳定的 reservation/settlement 结果投影回
  checkpoint state。

Coordinator 首次调用、Coordinator repair、Specialist 内部循环、outer retry、
Synthesis 和 Synthesis repair 都必须走同一套计数所有权。测试要用同步闸门让
多个分支同时竞争最后一个许可，证明不会超发，而不只是最终计数大于上限后
报错。

### [P1] “硬截止时丢弃整个活动批次”与 LangGraph pending writes/当前 state delta 设计不自洽

位置：`spec.md:183-208`、`spec.md:350-366`、`spec.md:436-449`。

Spec 要求硬截止时活动批次全部不生效，只保留较早 barrier 已接受的结果；
同时又要求每个完成分支直接返回 `TaskOutcome`、accounting 和 Calculation
Artifact state delta。LangGraph 会把同一 superstep 中已完成 sibling 的写入作为
pending writes 持久化，某个分支失败或执行被中断时不会自动删除它们
（[官方 persistence / pending writes](https://docs.langchain.com/oss/python/langgraph/persistence)）。
因此，“取消最后一个分支”并不能自动推出“整个批次原子丢弃”。

需要把批次收集改成明确的两阶段协议：

1. 分支只写 `active_batch_outcomes`、`active_batch_artifacts` 和 branch usage；
2. barrier 在确认每个 Task 有一个终态且所有聚合校验通过后，才把
   这一批提升到 `accepted_*`；
3. expected `TaskFailed` 是可收集的终态；fatal/invariant failure 时不做提升；
4. Evidence body cache 中已经产生的 body 可以成为不可达 orphan，但 eligibility
   只能由 `accepted_*` 引用决定；
5. 外部 `CancelledError` 不能被转换成正常 `done`。

否则已完成 sibling 的 Artifact/Outcome 可能通过 reducer 留在 canonical state，
违反 Spec 自己的批次原子性。

### [P1] Synthesis 输入契约自相矛盾

位置：`spec.md:215-226` 对比 `spec.md:269-297`、`spec.md:350-366`。

前者规定 Synthesis 模型输入“只能”包含 standalone query、Intent、Specialist
Results 和 Evidence excerpts；后者又要求 Synthesis 从“提供的 catalog”中选择
Calculation Artifact alias 并输出 `[C#]` placeholder。没有模型可见的精简
Calculation catalog，Synthesis 无法合法生成 Calculation 引用。

建议把唯一契约改成：

```text
SynthesisInput = standalone_query
               + business_intent
               + accepted Specialist Results
               + eligible bounded Evidence catalog
               + eligible bounded Calculation catalog with stable aliases
```

同时明确 Calculation eligibility 是“来自成功 Task 的被接受 attempt，并在
barrier 提升”，以及 Synthesis repair 必须复用完全相同的两个 catalog 和 alias
mapping。是否允许 Synthesis 使用某个成功 attempt 中未被 Specialist Result
显式点名的 Calculation Artifact，也应在这里作为有意识的相关性权衡写清楚。

### [P1] Agent 的 answer token 仍可能早于最终 checkpoint 对外可见

位置：`spec.md:370-389`、`spec.md:450-454`，以及现有
[`stream.py`](../../app/langgraph_v2/stream.py)。

共享 Linear Core 已经正确使用 `durability="sync"`，并在 terminal checkpoint
失败时不释放 `done`；但当前 stream adapter 是从看到 `checkpoint_terminal` 事件
之后才开始缓存后续 frame。Agent spec 要求按现有形状发布经过门禁的 `token`、
`citations`、`done`。如果 publication node 先发 token/citations、最后才发 terminal
done，则前两者在 terminal state checkpoint 失败前已经到达客户端，仍会留下
“用户看到了答案，但 Conversation 没有该 assistant Message”的状态。

LangGraph 官方说明 `durability="sync"` 会在进入下一 step 前等待 checkpoint
完成；`custom` stream 本身不是持久 Event journal
（[官方 durability/streaming 说明](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)、
[官方 streaming 说明](https://docs.langchain.com/oss/python/langgraph/streaming)）。

建议使用明确的两节点边界：

```text
validate/gate -> finalize_state -> [sync checkpoint] -> publish -> END
```

`finalize_state` 只写 canonical report、citations、assistant Message 和 publication
manifest，不发答案内容；`publish` 只读取已经 checkpointed 的 canonical state，
再发 token/citations/done。另一种可接受实现是从 Agent publication 起点开始缓存
整段答案 frame，直到 terminal checkpoint 确认后再按原顺序释放。clarification
也应复用同一保证。测试必须让 terminal saver 失败，并断言 token、citations、
done 和 assistant Message 全部为零，而不只是没有 `done`。

### [P1] 测试计划证明了框架接线，但没有证明 Agent 行为可用

位置：`spec.md:404-495`。

脚本化 fake model 非常适合验证路由、reducer、权限、失败隔离和 SSE 顺序，应当
保留；但全部关键 Agent 决策都被脚本预先写好时，测试不能回答以下问题：

- Query Understanding 是否在真实措辞中正确澄清和选 Intent；
- Coordinator 是否会选对 Specialist、正确并行、避免重复工作并及时 Finish；
- Specialist 是否会选择正确 Tool/Skill 和参数；
- Synthesis 是否保持证据支持、披露限制，并正确使用 `[E#]`/`[C#]`；
- prompt 或模型版本改变后，完整 trajectory 是否退化。

LangChain 官方测试策略明确区分 fake-model unit tests、真实 provider integration
tests 和 trajectory evals，并指出 agent 应更重视 integration/eval
（[官方测试总览](https://docs.langchain.com/oss/python/langchain/test)、
[trajectory eval](https://docs.langchain.com/langsmith/trajectory-evals)）。

建议在 Agent mode 开放给 Tenant 前，至少增加一个小型版本化评测集：每个关键
Actor 先手工整理 5--10 个代表性/对抗性样例；对确定路径用 strict/subset/
unordered trajectory 检查，对研究质量、引用语义和完整性用 rubric 或人工抽检。
真实模型测试可以放在非阻塞的 pre-deploy job，不必让普通 CI 变慢或不稳定。

## 非阻塞但必须显式接受的风险

### [P2] “checkpoint 是 durable execution authority”与不可恢复 Evidence Run 的表述过强

位置：`spec.md:40-47`、`spec.md:140-145`、`spec.md:252-268`、
`spec.md:497-510`。

Spec 已经诚实说明 Evidence body 只在 request-owned memory 中，进程重启后即使
checkpoint 有成功 Outcome/Evidence ID 也只能 fail closed。由此得到的是“持久的
Conversation 与轻量控制 checkpoint”，不是一个自包含、可恢复的 durable Run。
LangGraph 官方把 checkpointer 的价值包括 fault tolerance；本方案主动放弃了其中
一部分，因为恢复所需 body 不持久化
（[官方 persistence 定义](https://docs.langchain.com/oss/python/langgraph/persistence)）。

POC 可以接受这一点，但文档、API 和运维约束必须一致：断线/进程故障后客户端
以新 request ID 整体重跑；不承诺 replay/resume；明确长连接对 LB/proxy
timeout 的部署要求；并为 checkpoint 增加 retention/pruning 方案。否则“durable”会
让调用方误以为能恢复长任务。

### [P2] 缺少来自 Evidence/Tool 输出的间接 prompt injection 与 checkpoint 反序列化防线

位置：`spec.md:252-282`、`spec.md:319-339`、`spec.md:423-460`。

Research Scope、只读 Tool、Skill 不扩权和 publication gate 已经很好地限制了
攻击半径，但 Evidence excerpt 本身仍是模型可见的不可信内容。现有门禁验证的是
来源身份和引用完整性，不验证 excerpt 中的指令是否诱导 Specialist/Synthesis
偏离任务。应在 ToolReturn 格式和 Actor instructions 中把检索内容明确标记为
data、禁止把其当作指令，并加入包含“忽略系统提示/调用其他 Tool/伪造引用”等
内容的对抗测试。LangChain 官方把 prompt injection 检测列为 agent guardrail
的常见用途
（[官方 guardrails 指南](https://docs.langchain.com/oss/python/langchain/guardrails)）。

此外，当前依赖的 LangGraph checkpoint serializer 默认并不等于显式 allowlist。
Agent state 将新增多种自定义类型，应该在 shared checkpointer 层配置 strict
msgpack 或明确的 `allowed_msgpack_modules`，并做完整 round-trip 测试；不要仅靠
ADR 中“built-in safe serialization”的文字假定安全
（[官方 checkpoint README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/README.md#serde)）。

### [P2] 框架级兜底上限、retry 口径和依赖版本边界没有钉死

位置：`spec.md:107-118`、`spec.md:153-181`、`spec.md:382-389`。

领域层已有 Round、Task 和 actor-local 调用次数限制，但没有显式设置 LangGraph
`recursion_limit`。一旦 routing bug 形成不消耗 Task budget 的环，只有框架默认值
兜底，而且默认值未根据本图最坏路径与 finalization 余量验算。应计算并显式配置
上限；正常 execution-limit completion 必须更早发生，`GraphRecursionError` 只表示
invariant failure
（[官方 recursion limit 说明](https://docs.langchain.com/oss/python/langgraph/graph-api#recursion-limit)）。

同时应把 “two eligible retries” 明确成 3 次总 attempt，或明确本意是 2 次总
attempt。LangGraph `RetryPolicy.max_attempts` 包含第一次执行；PydanticAI 的
output/tool retry 和 provider SDK retry 还可能与 outer retry 叠乘。Coordinator
和 Synthesis 的 `output_retries=0`、Specialist 的内外层 retry owner、失败调用的
计数时点都应成为代码配置和测试断言，而不只是一句“equivalent retry disabled”。

最后，`pyproject.toml` 使用 `pydantic-ai>=1.93.0`，而当前设计依赖 v1.93 的
`output_retries`、`ToolReturn.metadata` 和 message 提取语义。应精确 pin 或至少
限制为 `>=1.93,<2`，框架升级必须运行 checkpoint compatibility 与真实模型
canary suite。

## 建议的最小修订顺序

1. 先修正状态/执行契约：Run 身份与同-thread准入、`Overwrite` 重置、全局预算
   reservation、active-batch 两阶段提升，以及 checkpoint-before-publication。
2. 统一 SynthesisInput，显式加入 bounded Calculation catalog。
3. 把真实模型 integration/eval、间接 prompt injection、serializer allowlist、
   recursion limit、retry 口径和依赖 major-version 边界写成验收门。
4. 再把状态恢复为 `ready-for-agent`，实施时继续保留当前 01--14 的小步纵向测试；
   不需要因此退回完整 DAG、通用 Result framework 或 workflow DSL。

## 最终判定

- 架构方向：通过。
- POC 范围：偏大但仍有清晰边界；“POC”更接近 production-shaped vertical slice。
- 安全边界：基础较强，但需要补间接 prompt injection 与 serializer 配置。
- 并发/状态正确性：当前未通过。
- 可恢复性：作为明确限制可接受，不能表述成完整 durable execution。
- 测试充分性：确定性测试较强，真实 Agent 质量验证不足。
- 未解决评审意见：9（6 个 P1，3 个 P2）。
