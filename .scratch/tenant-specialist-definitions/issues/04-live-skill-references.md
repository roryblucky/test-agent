# 04: Activated Skill 按需读取实时 References

**What to build:** Specialist 模型通过 `load_reference` 获取已激活 Skill 的全部最新辅助材料；同一次 invocation 中修改文件后，下一次调用看到新内容。

**Blocked by:** 03 — Markdown Skills 渐进式多激活。

**Status:** resolved

- [x] `load_reference` 只读取当前 Tenant、当前 invocation 已激活的 Skill。
- [x] 每次调用通过 Loader 读取 `references/` 的直属普通文件并按文件名排序；不缓存列表或内容。
- [x] startup 和 activation 不预读 references；references 不附加到 Tier 2 Skill Definition。
- [x] 目录缺失或为空时成功返回空集；列举或任一文件读取失败时返回有界失败且不返回部分内容。
- [x] Specialist 与 FlowEngine context adapters 共用一个 reference contract 和结果模型，不重复实现加载逻辑。
- [x] reference 修改不改变冻结的业务 Tool bindings或 Research Scope。
- [x] Local/GCS 共享 Adapter contract 覆盖顺序、实时内容、空目录、读取失败和 Tenant 隔离。
- [x] 不增加 reference cache、版本保留、Pins、自动重试或 bundled script 执行。

## Answer

`activate_skill` 与 `load_reference` 保持不同职责：前者通过共同 Registry 激活 Tier 2 instructions；后者校验 invocation-local activation 后，每次读取最新 Tier 3 references。两条 runtime path 共享同一底层 reference tool contract，不存在第二份 definition 或 reference cache。

## Comments

- 2026-09-17：确定性模型覆盖 activation、两次实时读取、业务 Tool 和最终结果；未解决 review comments：0。
