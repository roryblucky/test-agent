# 13: 证明 research 与 clarification 终态遵守 committed-state publication

**What to build:** 统一所有可发布终态，使 clarification、完整 research 与有界 incomplete research 都先同步提交 canonical state，再由唯一 publication owner 投影 SSE；取消与失败不得制造未提交答案。

**Blocked by:** 12: 在冻结的 Prepared Synthesis 上 repair 并 fail closed

**Status:** ready-for-agent

- [ ] 只有专用 publication owner 能发出 answer token、citations 与 `done`；所有前置节点只写 canonical state 或发送无答案内容的 progress。
- [ ] clarification、完整 research、partial results、execution limit、两者并存以及 insufficient Evidence 的终态都复用同一 checkpoint-first 规则；pre-moderation 明确排除在 research completion 外。
- [ ] 受控 barrier/fake-saver 测试覆盖 prepare、Synthesis、gate、finalize commit 前、commit 后但 publish 前以及 publish 进行中的 cancellation，而不依赖 timing race。
- [ ] commit 前 cancellation 不留下 final assistant Message 或 final frames；commit 后 cancellation 不回滚 canonical Message，可发送零个或部分 final frames且没有 `done`；已提交 Message 不重复。
- [ ] `IncompleteResearch` 只含稳定 failed Task IDs、accepted Data Gaps、封闭 structural-reason allowlist 与 `insufficient_evidence`，并保持 immutable、monotonic 和 checkpoint round-trip 稳定。
- [ ] structural reasons 只允许 `task_limit`、`coordination_limit`、`coordinator_context_limit`、`calculation_state_limit`、`prepared_synthesis_limit`、`final_markdown_limit`；原因与 disclosure 按稳定 Round、Task、unavailable-outcome 顺序排列。
- [ ] disclosure 只显示经 Markdown escaping 的有界 canonical failed-task objectives、Data Gap coverage labels、固定 insufficient-Evidence statement 与标准 structural-limit statement，不泄露 tool identity、arguments、provider details、exceptions、retries 或 counters。
- [ ] `completion_status=incomplete` 在任何 incompleteness signal 存在时设置；`termination_reason` 严格按 `partial_results`、`execution_limit`、`partial_results_and_execution_limit`、`insufficient_evidence` 的优先级与组合规则产生。
- [ ] insufficient-Evidence statement 在 canonical checkpoint Message、拼接 token stream 与 `done.answer` 中一致且最多一次；Synthesis 不能删掉或覆盖 code-owned disclosure。
- [ ] 没有 incompleteness signal 时不添加 block，代码也不宣称全局完备；带 Evidence 的权威 negative/empty result 可以完整完成，缺失 Evidence body 则 unsupported-recovery fatal。
- [ ] planning、Task、Tool、gate、warning、citation 与 final progress 使用现有 additive SSE 类型；不暴露 chain-of-thought、provider deltas 或内部 diagnostics。
- [ ] fatal、authorization、reducer、corrupted-state、programmer、checkpoint 与 recursion failures 不得转成 bounded-completion `done`。
- [ ] canonical final Markdown 在 publication-ready 阶段继续满足 192 KiB cap；overflow fallback 仍只生成一次且保持 disclosure 规则。
- [ ] failed-task disclosure 使用 Ticket 10 保存的同一规范化 objective，只做规定的 Markdown escaping；checkpoint round-trip 后文本逐字稳定。

