# 06: 渐进激活 Specialist-owned Skills

**What to build:** 在不扩大 Coordinator 或模型权限的前提下，让 Specialist 从受信任、受 Scope 限制的摘要中选择并固定一个可复现的 Skill 版本。

**Blocked by:** 05: 发布首份 Scope-bound Evidence-backed Report

**Status:** done

- [x] effective skill summaries 只包含 shared、scoped、Tenant/Scope eligible 的 Skill，按受信任顺序最多取前 20 个；测试覆盖恰好 20 与 21。
- [x] Coordinator 永远看不到 Skill catalog。
- [x] Specialist 只能按目标选择 Skill；完整 instructions 只在当前 invocation 内可见。
- [x] 激活时校验 eligibility，并固定 Skill 的 name、version 与 content hash。
- [x] ineligible、cross-Specialist 或只存在于 cache 的 Skill 引用均 fatal。
- [x] Skill 声明的 required tools 必须已在冻结的 effective tool set 中。
- [x] Skill 的 allowed/required names 不得授予新权限、重新绑定实现或放宽 Scope 约束。
- [x] Skill 附带的 scripts/assets 在本 POC 中不得执行。

## Comments

- 实现：`72ff7f3`、`5767b04`；full definition hash 覆盖 instructions、Tool 声明与 references；refs 仅 activation invocation 可见。
- 验证：focused unit 34 passed；首批 HTTP/SSE/PostgreSQL integration 4 passed；全量 `scripts/run-pytest -q tests` 为 316 passed、1 skipped。
- 复审：gpt-5.6-sol high Standards、Spec 均无未决项；未决 review comments：0。
- Spec 所述并发、多轮、计算与完整外部 golden path，明定归 Ticket 14（该票亦以 06/13 为前置）；非本票遗漏。
