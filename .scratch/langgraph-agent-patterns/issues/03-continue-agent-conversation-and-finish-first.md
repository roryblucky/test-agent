# 03: 安全继续 Agent Conversation 并处理 `Finish`-first

**What to build:** 让已消歧的 Agent 请求安全进入受信任 Scope 与 Coordinator，并支持第一轮直接 `Finish`、没有 Evidence 的保守终态。

**Blocked by:** 02: 发布已提交的 Agent clarification Run

**Status:** done

- [x] 新 Request 只读取完整历史消息对，并清空全部 run-local accepted、staging、answer、error、diagnostic 与 counter 状态。
- [x] Query Understanding 产出非空 standalone query 与受信任 Intent；未知 Intent 在任何数据访问前 fatal。
- [x] 系统从 Intent 构造不可变 Research Scope；模型不得命名或授予 tool、skill、source、filter。
- [x] Coordinator 只接收 standalone query、Intent 和经 Scope 过滤的描述符，不接收 Conversation 历史。
- [x] Coordinator 调用没有 tools，关闭 Tool retry，并显式配置一次 output retry；同一 actor invocation 最多 2 次 model requests，每次请求超时 60 秒，`max_tokens` 为 1500。
- [x] 第一条合法 Coordinator 决策可以是无 payload 的 `Finish`。
- [x] 没有 Evidence 时跳过 Synthesis，生成 `IncompleteResearch`，`termination_reason` 为 `insufficient_evidence`，并且固定披露文本恰好出现一次。
- [x] clarification 与 pre-moderation 不适用 research completion；带有 Evidence 的权威 negative/empty result 可以完整完成，不自动视为不足。

## Comments

- 2026-09-06：Sol-medium 确认三 seam：HTTP/SSE+checkpoint、公开 Scope resolver、Coordinator PydanticAI adapter；拒绝提前实现 Specialist dispatch、batch/staging、rolling rounds、Synthesis、Team/DSL 或 Run repository。
- 2026-09-06：可信 Tenant `agentResearchConfig` 提供 Intent policy 与 Query Understanding/Coordinator model profile。Query Understanding 只读完整 Conversation pairs；Scope 仅从受信任 policy 派生，忽略模型 Intent metadata 中的 tool、skill、source 与 filter。
- 2026-09-06：首轮 Coordinator 仅见 standalone query、selected Intent、Scope-filtered Specialist descriptors；无 tools、每请求 60 秒、`max_tokens=1500`。2026-09-07 后续重构将 output retry 改为显式一次并接入 code-owned output validator；timeout 通过 `ModelSettings` 逐 model request 生效，不包围整个 `Agent.run`；Tool retry 仍为零。仅接受必含 `kind=finish` 的无业务 payload 决策。
- 2026-09-06：首轮 Finish 无 Evidence 时不调用 Synthesis；构造 application-only `IncompleteResearch`，以唯一固定 disclosure 发布 `completion_status=incomplete` 与 `termination_reason=insufficient_evidence`。clarification 与 pre-moderation 保持既有非 research-completion 路径；Evidence-producing negative/empty path 属后续 Evidence ticket，未改写为不足。
- 2026-09-06：Sol-high 初审 finding 均已处置：真实 PydanticAI `TestModel` actor 测试替代 registry spy；token cap 从 factory 移至 invocation，消除重复 `model_settings`；Intent policy/model profile 改由 Tenant config；`Finish.kind` 改必填并测试拒绝空对象。复审 Standards/Spec 通过；未解决 review comments：0。
- 验证：focused 15 pass；`pytest tests`：268 passed、1 skipped；`ruff check app tests`、`pyright --pythonpath .venv/bin/python`、`git diff --check` 均通过。PostgreSQL integration 使用本机专用 test DB。
