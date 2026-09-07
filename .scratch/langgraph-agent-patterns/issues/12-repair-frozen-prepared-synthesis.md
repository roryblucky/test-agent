# 12: 在冻结的 Prepared Synthesis 上 repair 并 fail closed

**What to build:** 从冻结、可审计的 Prepared Synthesis 输入生成最终报告；格式错误只允许一次同输入 repair，所有 marker、citation 与 calculation gate 均由代码 fail closed。

**Blocked by:** 10: 执行有界 rolling Coordination Rounds; 11: 推广并由代码渲染 Calculation Artifacts

**Status:** done

- [x] Prepared Synthesis 只包含 canonical spec 明确允许的字段，不含 Conversation history、runtime state、raw payload 或其他隐式上下文。
- [x] repair 复用完全相同的冻结 input value，只额外提供 aliases 与确定性 validation errors；不得用 digest 或重新 materialize 的近似输入替代。
- [x] Synthesis 最多 2 次 invocations，每次恰好 1 个 model request，无 tools、无隐藏 retries，超时 120 秒，`max_tokens` 为 4000，并断言精确调用轨迹。
- [x] 代码执行严格 marker matrix 校验，拒绝 missing、duplicate、unknown、malformed 或不匹配的 Evidence/calculation marker。
- [x] recovery 时缺少所需 Evidence body 直接 fatal。
- [x] citation eligibility 按 frozen accepted provenance 与当前 Scope 的交集判定。
- [x] 没有 Evidence 时继续跳过 Synthesis。
- [x] Prepared Synthesis 最多包含 64 个 Evidence excerpts 和 32 个 calculation projections；测试覆盖恰好上限与多 1，禁止静默截断。
- [x] gate 不做语义 entailment 或事实蕴含判断。

## Comments

- Scope/test-seam review（gpt-5.6-sol medium）：以 `synthesize_report(actor, prepared)` 为公共 orchestration seam；它独占一次初稿、同一对象的一次 repair 与 publication gate。Calculation marker 可选引用；Evidence marker 须完整且唯一。
- Code review（gpt-5.6-sol high）：Prepared 三个模型的 `extra="forbid"` 与显式 `repair` protocol 已在 `5e3dd63` 关闭。Spec 复审为 0 actionable findings（定向 40 passed）。
- Dismissed review finding：256 KiB Prepared、192 KiB final/fallback byte caps 已从本票当前 scope 删除，现票唯一的大小约束为 64 Evidence / 32 calculation count；故不将旧需求回灌为本票实现。
- Verification：Ruff、Pyright 均通过；完整 PostgreSQL suite 为 430 passed、1 skipped。
- Unresolved review comments：0。
