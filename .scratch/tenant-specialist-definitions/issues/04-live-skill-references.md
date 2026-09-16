# 04: Activated Skill 按需读取实时 References

**What to build:** Specialist 模型可通过 load_reference 获取已激活 Skill 的全部最新辅助材料。同一次 invocation 中修改 reference 文件后，下一次调用能够看到新内容。

**Blocked by:** 03 — Markdown Skills 渐进式多激活与记录。

**Status:** ready-for-agent

依据：已批准的 Tenant-authored Specialist Definitions 规格及 ADR 0008。

- [ ] load_reference 通过当前 invocation 的 Activated Skill identity 解析可信 storage identity，只允许读取当前 Tenant、当前 invocation 已激活的 Skill；其他 invocation 的 activation 不能提供权限。
- [ ] 每次调用都通过 Loader 读取 reference 目录中的所有普通文件，并按确定性文件名顺序返回；不依赖会过期的内容或目录列表缓存。
- [ ] 启动和 Skill activation 均不预读 reference 内容；reference 内容不附加到进程缓存 Skill Definition，也不构成历史版本或 pin。
- [ ] 目录不存在或为空时成功返回空结果；列举失败或任一文件读取失败时返回明确、有界的失败信息，不把异常伪装为空目录，也不返回部分文档作为成功结果。
- [ ] 通过确定性模型完成 activation、load_reference、业务执行和最终结果；用真实 Local fixture 在两次读取之间修改内容，证明第二次取得最新内容。
- [ ] reference 修改不改变 Specialist Definition Pin、Skill Pin、有效业务 Tool bindings 或 Research Scope。
- [ ] 建立 05 可复用的 Loader reference 合同，覆盖读取顺序、当前内容、空目录、列举失败、单文件失败和 Tenant 隔离。
- [ ] 不增加 references 缓存、版本保留、自动重试机制或 bundled script 执行；聚焦测试和受影响回归测试通过。

## Comments

- 拆分及验收范围经独立 GPT-6 Astra agent 复审为 PASS；本票交付的读取合同是 GCS Adapter 的前置条件。
