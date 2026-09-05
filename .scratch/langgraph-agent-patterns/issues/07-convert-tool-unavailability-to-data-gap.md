# 07: 将预期 Tool 不可用转换为保守 Data Gap

**What to build:** 把已注册的预期 tool 不可用与 binding 自身超时转换为有界、可审计的 Data Gap，使部分成功只能保守地收紧最终结论。

**Blocked by:** 05: 发布首份 Scope-bound Evidence-backed Report

**Status:** ready-for-agent

- [ ] binding 只捕获已注册的预期 inability 与自身 20 秒 timeout，并映射到封闭的 reason-code 集合；其他异常不被吞掉。
- [ ] model-visible fallback return 上限为 4 KiB；测试覆盖恰好上限与多 1 byte，无法安全投影时使用 `response_unusable`。
- [ ] app-only metadata 保留完整 binding 字段，但不得包含 secrets、raw provider payload 或其他不可信大对象。
- [ ] 支持 fallback、multi-hop 与 Specialist 内部 fan-out；单个预期 unavailable 不取消 sibling，也不触发外层 Specialist retry。
- [ ] accepted attempt 中每个 accepted unavailable 与一个 Data Gap 一一对应；完全相同的重复项幂等。
- [ ] missing、stale、cross-Run、cross-Task、cross-attempt 或相互冲突的 Data Gap provenance 均 fatal。
- [ ] failed 或 abandoned attempt 的 Data Gap 不得被接受，并为 Ticket 08 的跨 attempt 隔离提供测试基础。
- [ ] 每个贡献最多 8 个 Data Gap；coverage 最多 256 个 UTF-8 bytes，identifier 最多 64 个 ASCII characters，source 最多 256 个 UTF-8 bytes；每项覆盖恰好上限与多 1。
- [ ] `DataGapView` 隐藏内部 ID、tool 与 source 信息。
- [ ] accepted partial success 单调地设置 incomplete；fallback 成功不能清除此信号。
- [ ] 权威 negative/empty Evidence 仍可完整完成，不自动成为 Data Gap。

