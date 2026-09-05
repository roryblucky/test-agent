# 08: 按封闭 V1 映射重试 Specialist

**What to build:** 按锁定的 PydanticAI V1 行为对 Specialist 做可证明、有界的外层 retry，同时隔离 abandoned attempt 的消息、Evidence、Data Gap 与 usage。

**Blocked by:** 07: 将预期 Tool 不可用转换为保守 Data Gap

**Status:** ready-for-agent

- [ ] 测试与启动检查锁定 `pydantic-ai==1.93.0`；版本偏移不得静默改变分类。
- [ ] retry 分类依据当前 V1 的精确 exception type、origin 与 status 组合；仅明确列入允许集的失败可重试。
- [ ] HTTP 401、408、600、bare/wrong-boundary provider error、business tool error、`ContentFilter` 以及未知类型或来源均 fatal。
- [ ] 同一 Task 最多 3 个 attempts；累计上限为 12 次 model requests 与 8 次 completed tool calls，单次 model request 超时 60 秒、单次 tool 调用超时 20 秒，`max_tokens` 为 2000；每个数值均测试恰好上限与多 1。
- [ ] 每个 attempt 独立 capture；成功只合并该 attempt 的 `result.new_messages`，失败消息保存在专用诊断结构中，abandoned attempt 的消息、Evidence 与 Data Gap 不得进入 accepted contribution。
- [ ] Coordinator、Specialist 与 Synthesis 都有完整 PydanticAI adapter contract tests：actor 注册、output schema、metadata 隔离、usage、精确调用轨迹、`TestModel`/`FunctionModel`、override、`ALLOW_MODEL_REQUESTS=False`、关闭 builtin retry、end strategy、timeouts 与 token caps；后续 Ticket 可扩展 repair 轨迹。
- [ ] 每个 Evidence cache item body 上限 16 KiB、Run 总 cache 上限 8 MiB；覆盖恰好上限与多 1。overflow write 被拒绝，产生不可重试 `TaskFailed` 且不携带 Evidence ID。
- [ ] cache 保留 orphan 以便诊断，但其永不具备发布资格；相同 body 并发写幂等，冲突内容 fatal，sibling 的有效 Evidence 不受失败 attempt 影响。
- [ ] 仅当 request/tool count limits 已明确配置且全部 PydanticAI token-usage limits 未配置时，`UsageLimitExceeded` 才按相应 count origin 分类；模型 `max_tokens` 与 usage limit 是不同概念，token-limit 或未知 origin fatal。
- [ ] retry exhaustion 生成 `TaskFailed`，usage 单独累计，不塞入 `TaskOutcome`。
- [ ] application exception 到 retry/fatal 的映射与 canonical spec 的封闭表逐项一致并有负例。

