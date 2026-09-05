# 05: 发布首份 Scope-bound Evidence-backed Report

**What to build:** 让一个受 Scope 限制的 Specialist 调用真实 tool binding，写入 request cache，经过基础 Synthesis 与确定性 gate，发布首份带 citation 的报告。

**Blocked by:** 04: 接受首个无 Tool 的 Specialist Task

**Status:** ready-for-agent

- [ ] effective tool set 在 Run 前冻结为 registered、Tenant、Scope 与 Specialist allowlist 的交集。
- [ ] 直接请求无效 Specialist、Tool、source，或尝试放宽约束，均在数据访问前失败；tool arguments 只能进一步收窄范围。
- [ ] `ToolReturn` 将有界的 model-visible value 与 app-only typed Evidence metadata 分离，metadata 对模型不可见。
- [ ] 只有已校验成功且被 `Finding` 引用的 body 才进入 request cache；missing、cross-Tenant、cross-Run、cross-Task 或冲突的 success provenance 均被拒绝。
- [ ] checkpoint、Send、reducers 与 PostgreSQL pending writes 不包含 Evidence body 或 raw provider payload。
- [ ] 相同 Evidence body 的重复写入幂等；同 ID 不同内容 fatal；orphan Evidence 不具备发布资格。
- [ ] Specialist Result 最多引用 16 个 Evidence ID；测试覆盖 16 与 17，超限在 contribution 前作为 structured-output-invalid 拒绝，外层 retry 留给 Ticket 08。
- [ ] 基础 Synthesis 只接收有界 excerpt，输出带 marker 的 Markdown；代码校验 marker 语法与 Evidence eligibility，并由代码推导 citations。
- [ ] Synthesis 首次调用没有 Tools、关闭隐藏 retry、超时 120 秒，`max_tokens` 为 4000。
- [ ] Tool audit 以及 planning、Task、Tool 的增量 SSE 是旁路 telemetry，不进入 `TaskOutcome`；Specialist 内部 tool calls 不是顶层 Task。
- [ ] Tool 对权威 negative/empty 查询仍产出 Evidence，不映射成 `ToolUnavailable`，并允许完整完成。
- [ ] 覆盖 cache 未超限路径；Evidence cache 溢出语义由 Ticket 08 完成。

