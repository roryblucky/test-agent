# LangGraph Agent Patterns：开源最佳实践核对

**研究日期：** 2026-09-04  
**被审阅文档：** [`.scratch/langgraph-agent-patterns/spec.md`](../../.scratch/langgraph-agent-patterns/spec.md)  
**代码库基线：** `langgraph==1.1.10`、`langgraph-checkpoint-postgres==3.1.2`、锁文件中的 `langgraph-checkpoint==4.2.0` 与 `pydantic-ai==1.93.0`

## 结论摘要

当前方案的主干是合理的：用自定义 `StateGraph` 把确定性控制面（规划校验、预算、路由、结果门禁、发布）与非确定性 actor（coordinator、specialist、synthesizer）分开；用 `Send` 做动态 worker fan-out；让每个 specialist 只拿到显式、最小化的任务上下文；在发布任何答案文本之前做结构化验证。这些都与 LangGraph/LangChain 官方模式吻合。

但 spec 还不能原样作为可实现且可验收的运行时契约。至少有三个阻断问题和若干高优先级空白：

1. **同一 `thread_id` 的跨请求状态清理语义不成立。** 带 reducer 的 map 字段接收 `{}` 不会清空上一轮数据；旧 `task_id`、结果或 artifact 可能污染下一请求。必须用 `Overwrite({})` 在单一初始化节点清空所有请求级 reducer 字段，或为每个请求使用新 thread 并把长期消息另存。[S1][S5]
2. **OSS LangGraph + Postgres checkpointer 不提供跨实例的同-thread single-flight。** Agent Server 的 enqueue/reject/interrupt/rollback 是另一产品层的能力，不能隐含归因给 OSS checkpointer。应用必须按派生 `thread_id` 实现分布式排队、拒绝或锁，或明确接受竞态。[S14][S16]
3. **Agent 模式的“已发布内容必有精确 checkpoint”尚未被设计保证。** 共享 stream adapter 已显式使用 `durability="sync"`，但只在看见 terminal marker 后缓存 frame；spec 又要求 token/citations 位于最终 done 之前。应增加 `finalize/persist -> publish` 两节点边界，或从 Agent publication 开始缓存整段输出，使 canonical assistant message 的 checkpoint 先成功，再发布任何答案内容。[S12][S13]
4. **领域预算不能替代框架级上限。** 应显式配置 `recursion_limit`，明确 retry 是“重试次数”还是“总尝试次数”，并把 LangGraph 外层 retry、PydanticAI 内层 output/tool retry、provider retry 统一计入一个尝试预算。[S3][S9][S23]
5. **所谓 durable recovery 目前只覆盖对话与已完成 superstep 的轻量控制状态。** Evidence 正文只在请求/进程本地缓存，故 active run 的跨进程恢复按设计会失败关闭。这个取舍可以接受，但必须收窄措辞，并补充 Postgres 初始化、serializer allowlist、checkpoint 保留/体积与版本迁移契约。[S1][S15][S24]

综合判断：**方向正确，需完成上述阻断项后才达到 production-shaped POC 的自洽标准。**

## 证据标记

- **官方保证**：官方参考文档或版本锁定源码明确承诺的行为。
- **官方示例**：官方示例采用的模式，说明受支持，但不等于应用级语义保证。
- **应用策略**：spec 自己定义、需要应用代码和测试保证的政策。
- **合理推断**：由官方机制组合得出的工程结论；不是框架 API 的独立保证。
- **版本敏感**：当前线上文档可能描述晚于本仓库依赖的功能，必须以锁定版本测试或源码为准。

## 逐模式审阅

### 1. Clarification：完成一轮，而不是 interrupt/resume

**官方事实。** LangGraph 的 `interrupt()` 用于暂停同一次图执行。恢复必须使用同一个 `thread_id` 和 `Command(resume=...)`；恢复时节点从头执行，所以 interrupt 前的副作用必须幂等。[S2] 已经正常结束的图则应以普通输入字典开始下一次执行；`Command(update=...)` 不是开始新请求的方式。[S2]

**官方示例。** LangChain 的 Open Deep Research 参考实现把澄清问题写成一条 `AIMessage`，然后用 `Command(goto=END)` 正常结束；用户下一条消息通过对话历史进入新一轮，而不是恢复旧 interrupt。[S18]

**对 spec 的判断。** spec 选择“澄清是一个完整回合；下一条用户消息是同会话的新请求”，与官方示例吻合，而且比无必要地保留一个 suspended run 更简单。应保留这一决定。

**需要补强。**

- 明确新请求使用普通输入字典；不要使用 `Command(resume=...)`。
- 澄清回复也应走与最终答案相同的 `persist -> publish` 边界。
- 如果未来引入 approval 型 HITL，必须另建 interrupt 路径，并测试节点重放及副作用幂等，不能复用当前 clarification 语义。

### 2. Coordinator / specialist delegation

**官方事实与示例。** LangChain 把 central supervisor + isolated stateless subagents 列为正式 multi-agent 模式：主 agent 负责路由，subagent 获取受控上下文并只返回最终结果；对于少量固定能力，官方建议用枚举限制 agent 名称。官方还明确建议在“确定性逻辑与 agentic behavior 混合、路由复杂”时使用自定义 LangGraph workflow。[S6][S7]

**对 spec 的判断。** 以下选择是合理的：

- coordinator 只输出 typed `DispatchBatch | Finish`，不执行业务工具；
- 每个 specialist 是隔离的单次 actor invocation，只收到显式 `SpecialistTask`；
- dispatcher 进行 allowlist、预算、轮次、权限校验；模型输出不能扩大工具权限；
- coordinator 与 synthesizer 分开，避免规划 actor 顺便生成未经门禁的最终答案。

这些是良好的**应用策略**，不是 LangGraph 自动强制的安全边界。tool registry、scope binding、租户授权和输出校验仍由应用负责。LangGraph 官方 threat model 同样把 LLM 生成的 tool call 和用户注册工具/节点视为需要由项目治理的边界。[S24]

**建议。** 小而固定的 specialist roster 应直接成为判别联合或枚举；任务 ID、角色、轮次和 scope 在进入 `Send` 前由确定性节点分配。对 prompt injection 的验收应验证“evidence 文本中的指令不能改变 specialist/tool allowlist、预算、发布门禁或 citation ownership”。

### 3. Parallel fan-out / fan-in 与 `Send` / `Command`

**官方保证。** `Send(node, arg)` 是 LangGraph 为动态 map-reduce 提供的路由 primitive；官方 orchestrator-worker 示例正是由 orchestrator 结构化规划，条件边返回多个 `Send`，worker 把结果写入带 reducer 的共享 key，之后再合成。[S3][S4] 同一 superstep 的分支并行执行，fan-in 节点要等所有上游分支完成；并行 update 的顺序未定义。[S8]

`Command` 可以同时返回 state update 和 `goto`。如果同一节点既声明静态 edge 又返回 `Command(goto=...)`，两条路径都会执行；因此一个路由点必须只选择一种路由机制。[S3]

**对 spec 的判断。** `decision -> validate -> Send(worker...) -> barrier/collect -> coordinate` 是标准 orchestrator-worker / rolling map-reduce 形态。每个 worker 使用 generic `execute_specialist` 节点、在 `Send.arg` 中接收完整任务，也符合“结构固定、并行实例动态”的原则。

**需要补强。**

- 明确 fan-out 由 conditional edge **或** `Command(goto=[Send(...)])` 实现，不要同时保留同节点的静态 outbound edge。
- 在 fan-out 前持久化 accepted decision、稳定 task ID 和 round；worker 不得自行生成这些 ID。
- `max_concurrency=8` 只限制该 graph invocation 中可并行执行的 LangGraph task，不会自动限制每个 PydanticAI actor 内部的并行工具调用或 provider 并发。因此需要内层/租户级 semaphore，或明确禁止/限制 actor 内部 parallel tool calls。[S8][S22]
- barrier 后所有输入按稳定 key 排序；不得依赖完成顺序。

### 4. State、reducer、跨请求 reset

**官方保证。** 每个 checkpoint 保存 channel values；同一 `thread_id` 的后续 invocation 从该 thread 的现有 state 继续。对带 reducer 的字段，输入和节点 update 同样通过 reducer，而不是无条件替换。[S1][S3] 并行分支写同一个无 reducer key 会触发 `INVALID_CONCURRENT_GRAPH_UPDATE`。[S5] `Overwrite(value)` 可绕过 reducer，但同一 superstep 对同一字段出现多个 `Overwrite` 会报错。[S3][S19]

**阻断偏离。** spec 同时要求：

- 同一 conversation/mode 复用稳定 `thread_id`；
- 新请求只继承 `conversation_messages`，其他字段全部 reset；
- task/outcome/artifact 等集合使用 map reducer，空输入为 identity。

这三者不能靠“输入空 map”共同成立。对 merge reducer 而言，`merge(old, {}) == old`。在仓库的锁定版本上用 `InMemorySaver` 实测：第一轮写 `{'old': 1}`，第二轮同 thread 输入 `{}` 后仍为 `{'old': 1}`；输入 `Overwrite({})` 后才为空。这是框架语义的直接后果，不是 Postgres 特例。

**建议的最小改动。**

1. 入口增加唯一的 `initialize_request` 节点。
2. 它只读取经过验证的新请求 envelope，把所有请求级 reducer channel 写为 `Overwrite({})` 或 `Overwrite([])`，同时生成新的 `request_id`；`conversation_messages` 保留正常 reducer 语义。
3. 初始化节点不与其他写相同 key 的节点并行。
4. 所有 task/result/artifact key 都包含 `request_id`；即便遗漏清理，也能 fail closed，不能混入上一轮。
5. 两次连续请求的集成测试必须断言第二轮 checkpoint 与 prompt projection 中不存在第一轮 task/artifact ID。

另一种更强隔离方案是每次请求使用新 `thread_id`，把会话历史放入独立 store；但这会改变当前“对话即 thread”的模型，POC 没必要先承担这层复杂度。

**Reducer 代数。** spec 已要求 stable-map reducer associative、identity、重复同值幂等、同 ID 异值冲突，这是好的。还应显式要求 **pure、commutative/order-independent**：官方明确说并行 update 顺序不保证。[S8] 用全排列/property test 验证相同 update 集合在不同顺序下得到相同结果；冲突必须与顺序无关并稳定失败。

### 5. Structured output、validation 与 repair

**官方事实。** PydanticAI 支持结构化输出、output validator 和 output retry；但 retry 分多层，框架 retry 会重新调用模型。v1.93.0 的 `Agent` 暴露 `retries`、`tool_retries`、`output_retries`，所以“禁用内建 output retry”必须落实为该锁定 API 的显式配置，而不能只写成注释。[S20][S21]

**对 spec 的判断。** coordinator/synthesizer 分别采用：typed output → 纯确定性 validation → 最多一次显式 repair → 再次 validation → fallback，并在 validation 前不发布答案文本，是强于框架默认的良好应用策略。保留“原始候选 + 有界规范化错误 + 同一 catalog”作为 repair 输入也合理。

**需要补强。**

- 在 v1.93.0 中对 coordinator 和 synthesis 明确设 `output_retries=0`，并测试模型 actor 的总 invocation 数确为最多 2；不要只把 Pydantic validation failure 当成应用显式 repair。
- 决策 DTO 应拒绝额外字段；校验 task 数、role 枚举、scope、依赖、预算与重复 ID。模型不得提供可执行函数名、工具实例或任意 graph node 名。
- synthesis 先解析成结构化对象，再做 citation/evidence ownership、完整性、预算与安全字段校验；最终用户文本只能从已接受对象渲染。
- `ToolReturn.metadata` 在 v1.93.0 明确是应用侧 metadata，不会发送给模型；用它携带 `Evidence` 是合理的，但它在 specialist 完成后必须被应用代码显式抽取并写入受控结果结构。[S25]

**版本风险。** `pyproject.toml` 当前写 `pydantic-ai>=1.93.0`，但官方已在 2026-06-23 发布稳定 v2，当前 v2 的 retry/output API 与 v1 不同。[S26] 当前 lock 固定 1.93.0 只保护已锁环境；新解析环境可能跨 major。应改为精确 pin 或 `>=1.93,<2`，并把升级作为显式迁移。

### 6. Retry、branch isolation 与 pending writes

**官方保证。** `RetryPolicy.max_attempts` 包含第一次执行；在 `langgraph==1.1.10` 源码中默认值是 3，即最多两次 retry。[S9] 并行 superstep 中若某节点抛异常，该 superstep 不会把 partial updates 提交到 graph state；如果启用了 checkpointer，成功 task 的 pending writes 会被保存，恢复时成功 task 不必重跑。[S1][S8]

**对 spec 的判断。** worker 捕获“预期失败”并返回 typed `TaskFailed`，而 invariant/bug 继续抛出，是正确的 failure isolation：否则一个可预期的搜索失败会中止整个 fan-out。技术性重试和“研究不足后的新 follow-up task”分开计数也正确。

**语义空白。**

- “two eligible retries”要明确是最多 **3 次总尝试**。若本意是最多两次总尝试，则 `RetryPolicy.max_attempts=2`。
- LangGraph node retry、PydanticAI tool/output retry、HTTP/provider SDK retry 可能叠乘。必须指定唯一的 attempt ledger，并在每次外部调用开始前原子占用预算。
- PydanticAI v1.93 `UsageLimits.request_limit` 在发起模型 request 前检查；token 数在收到 response 后才可累计；`tool_calls_limit` 只统计成功 tool calls，不能单独实现 spec 的“所有模型/工具尝试都计数”。[S23]
- LangGraph 1.1.10 没有当前 1.2 文档中的 node `TimeoutPolicy` API；POC 的 60 秒 node timeout 应由应用内 `asyncio.timeout`/provider timeout 实现，并以锁定版本集成测试。不要从较新的文档反推 1.1.10 已提供该能力。[S10]

**fatal branch 与 pending writes。** checkpointer 可能已经保存同一失败 superstep 中完成 sibling 的 pending writes。[S1] 因此必须测试：一个 branch 完成、另一个发生 fatal failure 后，未经过 barrier 的 staged write 不会成为 canonical accepted state，新请求初始化也不会吸收前一请求的 Run-local pending write。

### 7. Recursion 与 work limits

**官方保证。** `recursion_limit` 是 graph 允许的最大 superstep 数，超过后抛 `GraphRecursionError`；它是顶层 config key。官方也给出 `RemainingSteps` 用于在达到硬限制前主动终止的模式。[S3]

**偏离。** spec 有 `max_tasks`、`max_rounds` 和 model/tool attempts，但没有写明 graph 的 `recursion_limit`。这些领域限制并不能阻止 wiring bug 或不消耗 task budget 的循环。

**建议。** 根据静态图最坏路径计算一个显式上限，并留少量 cleanup/finalize 余量；正常的预算耗尽必须在到达此上限前产生 typed `Finish(incomplete)`，`GraphRecursionError` 只表示 invariant failure。测试一个意外自环，验证达到框架上限后不再发生 model/tool call。

### 8. Postgres persistence 与 checkpointer 运维

**官方保证。** checkpointer 在每个 superstep 保存 checkpoint；以 `thread_id` 组织 thread，使 memory、time travel、HITL 与 fault recovery 成为可能。[S1] 官方 Postgres saver 要求首次运行 `setup()`；手动连接必须使用 `autocommit=True` 和 `dict_row`，连接字符串 helper 会配置这些参数。[S15]

**对 spec 的判断。** 复用 Linear Core 的 PostgreSQL、稳定派生 `thread_id`、避免再造 run/event 表，是合理的最小 POC 选择。但“Postgres 中存在 checkpoint”不自动等于以下应用保证：

- 同 thread 的跨实例互斥；
- 对该 thread 的租户授权；
- SSE event 重放或 exactly-once delivery；
- 已开始的任意 Python/HTTP side effect 自动恢复；
- state schema/graph code 变更后的任意旧 checkpoint 都可恢复。

**必须加入的运维契约。**

- migration/setup 的 owner、幂等启动流程和 readiness check；
- DB 连接池、超时、事务和 tenant/thread authorization；
- checkpoint retention、PII 删除、单 thread 大小与读取延迟预算；
- state schema 与 graph 版本兼容测试。官方说明 checkpoint 会由当前部署代码执行；重命名/删除节点、修改 state 可能破坏暂停或恢复中的 thread。[S17]
- serializer 采用 strict msgpack/最小 allowlist，不启用 `pickle_fallback`；官方 serializer 源码警告 checkpoint 数据库可写者可能利用反序列化执行代码，官方 threat model 也把非 strict 默认与 pickle fallback 标为风险。[S24][S27] 当前 `langgraph-checkpoint==4.2.0` 已高于官方修复 pickle cache 默认值的 4.0.0，但 strict allowlist 仍需应用显式选择。[S28]

**体积。** 当前默认 checkpoint 保存每个 superstep 的完整 channel value。最新文档中的增量 `DeltaChannel` 是 1.2 beta，仓库的 `langgraph==1.1.10` 不能把它当作现成功能。[S1][S11] 因此 state 中只留轻量 ID 是正确方向，但长期 `conversation_messages` 仍会增长；“prompt 投影有界”不等于“持久 state 有界”。需要对历史裁剪/摘要、checkpoint retention 和大 thread load time 做验收。

### 9. 多轮会话、跨实例恢复与 recovery 边界

**官方事实。** 相同 `thread_id` 能让新的 graph 实例从共享 checkpointer 读取该 thread 的 state；这支持完成回合后的多轮 conversation continuity。[S1] 对一次失败 superstep，pending writes 允许恢复时跳过已成功的 task。[S1]

**合理推断。** 当前 spec 只在 checkpoint 中保存 Evidence ID，Evidence 正文存在请求/进程内 cache；因此新进程无法重建 active run 的 synthesis 输入。spec 已明确“正文缺失时 fail closed”以及透明 crash recovery 不在 POC 范围内，这个选择内部一致，但“checkpoints are durable execution authority”表述过强。

建议改成：

> Checkpoints are the durable authority for conversation state and control state committed at completed supersteps. Cross-process continuation of an active agent request is unsupported in the POC because evidence bodies are request-local; missing bodies fail closed. Completed-turn multi-session conversation continuity remains supported.

若未来承诺 active-run recovery，应在受租户保护且有 TTL 的持久 store 保存 Evidence body 或可确定重取的 locator/hash，并为工具副作用定义幂等键。否则不要把 checkpointer 的 framework capability直接宣传成端到端 recovery guarantee。

### 10. 同 thread 并发 / double texting

**官方边界。** LangSmith/LangGraph Agent Server 文档提供 enqueue、reject、interrupt、rollback 等 double-texting 策略，同时明确这些能力属于 Agent Server，**OSS LangGraph 不包含这些运行管理策略**。[S16] Postgres saver 源码中的 async lock 是 saver 实例内部锁；它不能构成跨进程、跨实例的同-thread 分布式 single-flight。[S29]

**阻断偏离。** spec 的 `thread_id` 从 tenant/conversation/mode 稳定派生，并要求跨实例连续性，但没有规定两个实例同时处理同一 conversation 时怎么办。两个请求可能读取相同 parent checkpoint、并发写分支 checkpoint，最终状态和对外回复不再有确定性。

**必须选择并写入一种语义。** POC 建议默认 **reject**（已有 active request 时返回明确 conflict/retry-after），实现成本最低；也可使用 DB advisory lock/租约实现 enqueue。无论选择哪种，都要以 derived `thread_id` 为键、跨进程生效、处理 owner 崩溃/租约过期，并做两实例并发测试。不要依赖进程内 `asyncio.Lock`。

### 11. Streaming 与 publication semantics

**官方保证。** OSS LangGraph 可输出 `values`、`updates`、`messages`、`custom`、`checkpoints`、`tasks`、`debug` 等 stream mode；`custom` 适合从任意节点或非 LangChain LLM 集成（如 PydanticAI）发送进度。[S12] Pregel 执行采用 plan/execute/update：同一 superstep 的 writes 在所有 task 完成前对其他节点不可见。[S13]

**官方不保证。** 这些 transport events 本身不是 checkpoint journal，也不承诺 SSE 断线重放、exactly once 或“客户端看到的每个 token 已持久化”。Agent Server 的 join/rejoin 是单独的 server 能力，传统 request streaming 断开会失去该 stream，不能隐含归因给 OSS graph。[S30]

**对 spec 的判断。** buffered output、校验失败时零答案 token、确定性 progress event、validation 后才答复，是非常好的 publication policy。现有共享 stream adapter 已经使用 `durability="sync"` 并延迟 terminal frame；但它只在收到 `checkpoint_terminal` 后开始延迟后续 frame。若 Agent publication 按正常顺序先发 token/citations、最后发 done，前面的答案内容仍可能早于 terminal checkpoint 对外可见。要兑现“每条发布的 assistant Message 有 exact checkpoint”，建议把图分为：

```text
validate/gate -> finalize_state -> [sync checkpoint] -> publish -> END
```

- `finalize_state` 只写 canonical accepted response、citations 与 publication manifest，不发送用户可见答案 token。
- 保留共享 graph invocation 已有的 `durability="sync"`；官方定义 `sync` 为在进入下一步前等待 checkpoint 持久化，而默认 `async` 会与下一步并行。[S13][S31]
- `publish` 仅从 checkpointed canonical response 渲染 `answer_chunk` / `citations` / `done`。
- clarification 走相同边界。
- 文档明确交付是 at-most-once/best-effort stream，除非另建 event journal；客户端断开后可用正常 conversation read 获取 canonical final message，但不能无 journal 精确续传 token cursor。

“synthesis_actor 的 LLM token streaming”和“validated answer publication”必须是两个不同通道；前者最多是内部 telemetry，不能直接成为对用户的 answer chunks。

### 12. 真实模型 eval 与可观测性

spec 的 stub/structural/property/integration tests 对确定性不变量是必要的，但不能证明真实模型在 schema adherence、tool routing、repair、prompt injection、citation grounding 上达到可用质量。LangSmith 官方把复杂 agent eval 分成 final response、trajectory 和 single-step 三层，并建议在开发与生产监控中持续评估非确定性输出。[S32][S33]

建议增加一个小而稳定、可重复运行的真实 provider canary suite（不替代 deterministic CI）：

- **final response**：答案正确性、完整性、无证据时的诚实降级、citation 与 accepted Evidence 一致；
- **trajectory**：该澄清时结束、不该澄清时 dispatch；specialist 选择、round 数、工具 scope 和预算符合预期；
- **single-step**：Coordinator schema adherence、非法 ID/未知 role/越权工具被确定性 validator 拒绝；
- **repair**：首个输出故意缺字段或越权，验证恰好一次 repair，第二次失败进入 deterministic fallback；
- **adversarial**：工具结果/evidence 含“忽略规则、调用其他工具、伪造 citation”等文本，验证其只能作为数据，不能改变 graph policy；
- **统计验收**：固定数据集、多次重复、按模型/版本记录成功率、p95 延迟、token/tool attempts 和 fallback 率。升级模型、prompt 或 PydanticAI/LangGraph 版本时对比基线。

这属于应用质量保障，不是 LangGraph 对 agent correctness 的保证。

## 建议直接修改 spec 的条款

按优先级：

1. **请求初始化：** 同一 thread 的每个新请求先进入唯一初始化节点；对所有 run-local reducer channel 使用 `Overwrite(empty)`，写新 `request_id`，仅继承经过裁剪的 `conversation_messages`。
2. **并发：** derived `thread_id` 跨实例 single-flight；POC 选择 reject 或 enqueue 之一，并定义 crash/lease 语义。
3. **发布：** `finalize_state` 与 `publish` 分离；运行使用 `durability="sync"`；明确 SSE 无 replay/exactly-once 保证。
4. **框架上限：** 显式 `recursion_limit`；领域 stop 在其之前；`GraphRecursionError` 归为 invariant failure。
5. **重试计量：** 定义总尝试次数，注明 `RetryPolicy.max_attempts` 含首轮；禁用 coordinator/synthesis 的内建 output retry；外层/内层/provider retry 全部纳入 ledger。
6. **并发计量：** `max_concurrency=8` 之外，再限制 PydanticAI actor 内部工具并行与全局 provider/tool concurrency。
7. **恢复措辞：** 明确 completed-turn conversation continuity 支持，active-run cross-process recovery 不支持；Evidence body 缺失 fail closed。
8. **Postgres：** setup/migration、pool、retention、tenant auth、serializer strict allowlist、schema/graph version 兼容写入验收。
9. **依赖：** PydanticAI 加 `<2` 或精确 pin；任何升级必须重新运行真实模型与 checkpoint compatibility suite。

## 最小新增验收矩阵

| 场景 | 必须断言 |
|---|---|
| 同 thread 连续两请求 | 第二请求 state、prompt、outcomes、artifacts 不含第一请求 ID；messages 按策略继承 |
| 两实例同 thread 同时请求 | 一个被 reject/排队；不存在双 final reply 或分叉 canonical state |
| 并行分支一成一败 | 成功 branch pending write 行为符合声明；恢复/作废不会重复副作用或污染新请求 |
| reducer 更新全排列 | 最终 map 相同；同 ID 异值在任意顺序下稳定冲突 |
| 8 个 Send + actor 内并发工具 | graph、provider、tool 三层峰值均不超过各自上限 |
| retry 叠加 | 总 model/tool/provider attempt 不超过统一 ledger；统计包含失败调用 |
| 意外 graph loop | `recursion_limit` 生效；越界后无更多外部调用 |
| finalize checkpoint 写失败 | 客户端收到零 answer token/clarification text |
| publish 中途断线 | canonical final message 已持久化；文档不声称 token replay/exactly once |
| 新进程 active-run continuation | Evidence body 缺失时确定性 fail closed，不生成无依据答案 |
| serializer 恶意/未知类型 | pickle 禁用；非 allowlist 类型被阻断；DB tenant 不能跨界读 thread |
| 真实模型 canary | final/trajectory/single-step 指标达到版本化阈值，prompt injection 不改变权限 |

## 一手资料索引

以下均为项目官方文档、官方源码/示例或官方发布记录；访问日期均为 **2026-09-04**。

- **[S1] LangGraph 官方文档 — Persistence**（官方保证：checkpoint、thread、reducer、pending writes；版本敏感的 DeltaChannel）：https://docs.langchain.com/oss/python/langgraph/persistence — 访问 2026-09-04。
- **[S2] LangGraph 官方文档 — Interrupts**（官方保证：`Command(resume)`、same thread、节点从头重放）：https://docs.langchain.com/oss/python/langgraph/interrupts — 访问 2026-09-04。
- **[S3] LangGraph 官方文档 — Graph API**（官方保证：`Send`、`Command`、reducers、`recursion_limit`）：https://docs.langchain.com/oss/python/langgraph/graph-api — 访问 2026-09-04。
- **[S4] LangGraph 官方文档 — Workflows and agents**（官方示例：orchestrator-worker、parallelization）：https://docs.langchain.com/oss/python/langgraph/workflows-agents — 访问 2026-09-04。
- **[S5] LangGraph 官方错误文档 — INVALID_CONCURRENT_GRAPH_UPDATE**（官方保证：并行同 key 更新需要 reducer）：https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE — 访问 2026-09-04。
- **[S6] LangChain 官方文档 — Subagents**（官方模式：centralized routing、stateless context isolation、parallel calls）：https://docs.langchain.com/oss/python/langchain/multi-agent/subagents — 访问 2026-09-04。
- **[S7] LangChain 官方文档 — Custom workflow**（官方建议：复杂路由、deterministic + agentic 混合使用 LangGraph）：https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow — 访问 2026-09-04。
- **[S8] LangGraph 官方文档 — Use the graph API**（官方保证：parallel superstep、fan-in、并行顺序未定义、error/retry/max concurrency）：https://docs.langchain.com/oss/python/langgraph/use-graph-api — 访问 2026-09-04。
- **[S9] LangGraph 1.1.10 官方源码 — `RetryPolicy`**（版本锁定保证：`max_attempts` 包含首轮，默认 3）：https://github.com/langchain-ai/langgraph/blob/1.1.10/libs/langgraph/langgraph/types.py#L404-L423 — 访问 2026-09-04。
- **[S10] LangGraph 官方文档 — Fault tolerance**（当前版本文档：retry、node attempt、timeout；1.2 新功能需与 1.1.10 区分）：https://docs.langchain.com/oss/python/langgraph/fault-tolerance — 访问 2026-09-04。
- **[S11] LangGraph 官方文档 — Pregel**（官方保证：BSP 执行；当前版本 DeltaChannel reducer 约束）：https://docs.langchain.com/oss/python/langgraph/pregel — 访问 2026-09-04。
- **[S12] LangGraph 官方文档 — Streaming**（官方保证：stream modes、custom streaming）：https://docs.langchain.com/oss/python/langgraph/streaming — 访问 2026-09-04。
- **[S13] LangGraph 官方文档 — Thinking in LangGraph**（官方说明：checkpoint durability 默认 async，可选 sync）：https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph — 访问 2026-09-04。
- **[S14] LangSmith 官方文档 — Double texting**（官方 server 能力及“not in OSS LangGraph”边界）：https://docs.langchain.com/langsmith/double-texting — 访问 2026-09-04。
- **[S15] LangGraph 官方仓库 — Postgres checkpointer README**（官方要求：生产使用、`setup()`、连接配置、安全提示）：https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/README.md — 访问 2026-09-04。
- **[S16] LangGraph 官方文档 — Double texting concepts**（官方说明 enqueue/reject/interrupt/rollback 的 server 语义）：https://docs.langchain.com/langsmith/double-texting — 访问 2026-09-04。
- **[S17] LangGraph 官方文档 — Backwards compatibility**（官方边界：部署代码、节点/state 修改与 checkpoint 兼容性）：https://docs.langchain.com/oss/python/langgraph/backward-compatibility — 访问 2026-09-04。
- **[S18] LangChain 官方 Open Deep Research 示例 — clarification node**（官方示例：clarification 后 `goto=END`）：https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py — 访问 2026-09-04。
- **[S19] LangGraph 1.1.10 官方源码 — `Overwrite` / `Send` / `Command` / `interrupt`**（版本锁定 API）：https://github.com/langchain-ai/langgraph/blob/1.1.10/libs/langgraph/langgraph/types.py#L574-L868 — 访问 2026-09-04。
- **[S20] PydanticAI 官方文档源码 — Output**（当前 v2 的 structured output/validator/retry 设计，版本敏感）：https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md — 访问 2026-09-04。
- **[S21] PydanticAI v1.93.0 官方源码 — Agent**（版本锁定：`retries`、`tool_retries`、`output_retries`）：https://github.com/pydantic/pydantic-ai/blob/v1.93.0/pydantic_ai_slim/pydantic_ai/agent/__init__.py — 访问 2026-09-04。
- **[S22] PydanticAI 官方文档源码 — Parallel tool calls**（当前行为与并发控制，版本敏感）：https://github.com/pydantic/pydantic-ai/blob/main/docs/tools-advanced.md#parallel-tool-calls-concurrency — 访问 2026-09-04。
- **[S23] PydanticAI v1.93.0 官方源码 — Usage limits**（版本锁定计量时点与成功 tool call 计数）：https://github.com/pydantic/pydantic-ai/blob/v1.93.0/pydantic_ai_slim/pydantic_ai/usage.py#L262-L281 — 访问 2026-09-04。
- **[S24] LangGraph 官方 threat model**（官方安全边界：checkpoint 反序列化、tool/node 信任、项目责任）：https://github.com/langchain-ai/langgraph/blob/main/.github/THREAT_MODEL.md — 访问 2026-09-04。
- **[S25] PydanticAI v1.93.0 官方源码 — `ToolReturn.metadata`**（版本锁定保证：metadata 不发送给模型）：https://github.com/pydantic/pydantic-ai/blob/v1.93.0/pydantic_ai_slim/pydantic_ai/messages.py#L859-L882 — 访问 2026-09-04。
- **[S26] PydanticAI 官方 Version Policy**（官方发布信息：v2 stable 于 2026-06-23）：https://github.com/pydantic/pydantic-ai/blob/main/docs/version-policy.md — 访问 2026-09-04。
- **[S27] LangGraph 官方源码 — `JsonPlusSerializer`**（当前 serializer strict msgpack/allowlist/pickle 安全说明，版本敏感）：https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py — 访问 2026-09-04。
- **[S28] LangGraph 官方安全公告 — checkpoint cache pickle fallback**（官方修复版本 `langgraph-checkpoint>=4.0.0`）：https://github.com/langchain-ai/langgraph/security/advisories/GHSA-mhr3-j7m5-c7c9 — 访问 2026-09-04。
- **[S29] LangGraph 官方源码 — `AsyncPostgresSaver`**（源码事实：saver 实例的 async lock 与连接配置）：https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py — 访问 2026-09-04。
- **[S30] LangChain 官方文档 — Join and rejoin streams**（官方 server 能力与传统 request stream 断线边界）：https://docs.langchain.com/oss/python/langchain/frontend/join-rejoin — 访问 2026-09-04。
- **[S31] LangGraph 官方源码 — `Durability`**（官方 API：`sync`、`async`、`exit`）：https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py — 访问 2026-09-04。
- **[S32] LangSmith 官方文档 — Evaluate a complex agent**（官方实践：final response、trajectory、single step）：https://docs.langchain.com/langsmith/evaluate-complex-agent — 访问 2026-09-04。
- **[S33] LangSmith 官方文档 — Evaluation concepts**（官方实践：非确定性质量的 pre-deployment 与 production evaluation）：https://docs.langchain.com/langsmith/evaluation-concepts — 访问 2026-09-04。

## 研究限制

- 本文只核对架构/spec，没有对尚未实现的 graph 做端到端运行或真实模型 benchmark。
- “官方示例”证明模式受支持，不证明其满足本项目的权限、预算、交付或恢复 SLA；这些都需要本项目测试。
- LangGraph 主站文档在研究日已包含 1.2-era 功能，而仓库 pin 在 1.1.10；本文凡涉及 `TimeoutPolicy`、`DeltaChannel`、较新的 event streaming 均按版本敏感处理，没有将其视为本仓库现成功能。
