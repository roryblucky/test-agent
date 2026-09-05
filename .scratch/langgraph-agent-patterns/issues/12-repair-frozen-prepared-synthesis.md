# 12: 在冻结的 Prepared Synthesis 上 repair 并 fail closed

**What to build:** 从冻结、有界、可审计的 Prepared Synthesis 输入生成最终报告；格式错误只允许一次同输入 repair，所有 marker、citation、calculation 与大小 gate 均由代码 fail closed。

**Blocked by:** 10: 执行有界 rolling Coordination Rounds; 11: 推广并由代码渲染 Calculation Artifacts

**Status:** ready-for-agent

- [ ] Prepared Synthesis 只包含 canonical spec 明确允许的字段，不含 Conversation history、runtime state、raw payload 或其他隐式上下文。
- [ ] repair 复用完全相同的冻结 input value，只额外提供 aliases 与确定性 validation errors；不得用 digest 或重新 materialize 的近似输入替代。
- [ ] Synthesis 最多 2 次 invocations，每次恰好 1 个 model request，无 tools、无隐藏 retries，超时 120 秒，`max_tokens` 为 4000，并断言精确调用轨迹。
- [ ] 代码执行严格 marker matrix 校验，拒绝 missing、duplicate、unknown、malformed 或不匹配的 Evidence/calculation marker。
- [ ] recovery 时缺少所需 Evidence body 直接 fatal。
- [ ] citation eligibility 按 frozen accepted provenance 与当前 Scope 的交集判定。
- [ ] 没有 Evidence 时继续跳过 Synthesis。
- [ ] 覆盖恰好上限与多 1：Prepared Synthesis 256 KiB、最多 64 个 Evidence excerpts 且每项 4 KiB、最多 32 个 calculation projections 且每项 2 KiB、最终发布内容 192 KiB。
- [ ] Prepared 或 final overflow 使用各自确定性的 termination reason，禁止静默 truncate。
- [ ] 192 KiB final cap 在 marker replacement、calculation rendering、escaping 与 incomplete disclosure 都完成后的 publication-ready Markdown 上计算。
- [ ] 测试 final 恰好 192 KiB 与多 1；超限只生成一次有界、确定性的 fallback，禁止递归 fallback 或重复 disclosure。
- [ ] gate 不做语义 entailment 或事实蕴含判断。
