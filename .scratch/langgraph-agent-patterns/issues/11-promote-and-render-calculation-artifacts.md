# 11: 推广并由代码渲染 Calculation Artifacts

**What to build:** 让 Specialist 从受信任 Evidence refs 产生有限、可复算的 Calculation Artifacts，由代码而非模型决定最终数值展示。

**Blocked by:** 09: 原子接受单 Run 内并发 mixed Dispatch Batch

**Status:** ready-for-agent

- [ ] 只支持 canonical spec 指定的 3 类 versioned calculation。
- [ ] 模型只能提交受信任 refs；executor 解析 series，直接提交 raw series 的模型输出被拒绝。
- [ ] calculation metadata 只从 accepted attempt 推广，并扩展 Ticket 08 的 attempt isolation 测试。
- [ ] `SpecialistResult` 不包含 Artifact IDs，不新增 artifact repository 或公开 artifact array；内部 calculation 调用不计作顶层 Task。
- [ ] Artifact IDs 与内容映射具备幂等与冲突检测，并在并行完成顺序变化时保持稳定。
- [ ] Synthesis 使用稳定 alias；模型不得提供 canonical value，最终值完全由代码渲染。
- [ ] 覆盖恰好上限与多 1：每个 contribution 最多 8 个 calculations、每项最多 4 KiB、每 Run 含 framing 最多 1 MiB、提供给模型的最多 32 个 calculation projections 且每项最多 2 KiB。
- [ ] active batch calculation overflow 时整个 batch 不 promote 并清空 staging；终态原因由此前已接受状态确定。
- [ ] internal reproducibility schema 包含 method、precision、unit/currency、period/as-of、assumptions、normalized input、Evidence refs/hashes、audit inputs、execution record、Run/Task/attempt provenance；round-trip 与 integrity 测试能重现格式化值，该 schema 仅内部可见。

