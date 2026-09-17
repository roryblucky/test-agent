# 06: 重启后使用当前 Catalog 恢复 Task

**What to build:** Run 从 checkpoint 恢复或 Task 重试时，使用当前进程的 Tenant Specialist Catalog，同时保留原 Run 的数据权限。已删除 Specialist 对应的合法 Task 独立失败，同批有效 Tasks 可以继续完成。

**Blocked by:** 02 — 本地 Markdown Specialist 从启动加载到完成 Task。

**Status:** ready-for-agent

依据：已批准的 Tenant-authored Specialist Definitions 规格及 ADR 0008。

- [ ] 用无 Skills 的 Local definitions 构建两个独立 startups；恢复后的 Coordinator 使用当前 catalog 全部有效 descriptors，新增 Specialist 可见，删除 Specialist 不可选，同 ID 修改采用当前 description 和 instructions。
- [ ] 保留原 Run 的 Tool、source、query、freshness 及其他数据约束；旧 checkpoint descriptors 不成为当前 catalog 的授权来源，不持久化或求交历史 Specialist ID allowlist。
- [ ] 将 Batch 完整性与当前 Specialist 可用性分开校验。身份、objective、context、已接受 Coordinator Decision 与 manifest 一致性继续由确定性 validation 保证。
- [ ] 已证明合法派发的 Task 若在当前 catalog 缺失其 Specialist Definition，不调用模型、不替换 Specialist，直接产生该 Task 的失败 outcome。
- [ ] 恢复包含两个 Tasks 的已接受 Batch，其中一个 Specialist 已删除：缺失项独立失败，有效 sibling 正常运行，barrier 收集两个 outcomes，后续流程遵循现有部分结果合同。
- [ ] 伪造 Specialist ID、Task 与已接受 decision 不匹配、损坏 manifest 等情况仍触发 invariant failure，不能降格为普通缺失 definition 的 Task failure。
- [ ] 同 ID definition 修改后，恢复或重试执行使用当前内容，不读取或恢复历史 Markdown body。
- [ ] 通过既有 PostgreSQL checkpoint 和 HTTP/SSE Agent 集成设施验证上述行为；测试无真实模型网络调用，并保留其他 resume、retry、Batch acceptance 的回归保障。
- [ ] 本票无需 Skill 或 GCS 路径。Skill definitions 两次 startup 语义由 03 承担，GCS 合同由 05 承担。
- [ ] 聚焦确定性恢复测试及受影响回归测试通过；全特性完成时执行规格要求的完整测试套件，若 05 先完成则由本票收尾执行。

## Comments

- 拆分及验收范围经独立 GPT-6 Astra agent 复审为 PASS；06 只依赖 02，02 完成后可与 03 分别推进。
