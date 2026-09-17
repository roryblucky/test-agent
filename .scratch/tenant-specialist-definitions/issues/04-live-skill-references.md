# 04: Activated Skill 按需读取实时 References

**What to build:** Specialist 模型可通过 load_reference 获取已激活 Skill 的全部最新辅助材料。同一次 invocation 中修改 reference 文件后，下一次调用能够看到新内容。

**Blocked by:** 03 — Markdown Skills 渐进式多激活与记录。

**Status:** resolved

依据：已批准的 Tenant-authored Specialist Definitions 规格及 ADR 0008。

- [x] load_reference 通过当前 invocation 的 Activated Skill identity 解析可信 storage identity，只允许读取当前 Tenant、当前 invocation 已激活的 Skill；其他 invocation 的 activation 不能提供权限。
- [x] 每次调用都通过 Loader 读取 reference 目录中的所有普通文件，并按确定性文件名顺序返回；不依赖会过期的内容或目录列表缓存。
- [x] 启动和 Skill activation 均不预读 reference 内容；reference 内容不附加到进程缓存 Skill Definition，也不构成历史版本或 pin。
- [x] 目录不存在或为空时成功返回空结果；列举失败或任一文件读取失败时返回明确、有界的失败信息，不把异常伪装为空目录，也不返回部分文档作为成功结果。
- [x] 通过确定性模型完成 activation、load_reference、业务执行和最终结果；用真实 Local fixture 在两次读取之间修改内容，证明第二次取得最新内容。
- [x] reference 修改不改变 Specialist Definition Pin、Skill Pin、有效业务 Tool bindings 或 Research Scope。
- [x] 建立 05 可复用的 Loader reference 合同，覆盖读取顺序、当前内容、空目录、列举失败、单文件失败和 Tenant 隔离。
- [x] 不增加 references 缓存、版本保留、自动重试机制或 bundled script 执行；聚焦测试和受影响回归测试通过。

## Answer

Specialist 现在在有 Eligible Skills 时同时获得 `activate_skill` 与 `load_reference`。invocation 只记录本次实际激活的 Skill 名称和 pins；`load_reference` 先检查当前 invocation 的 activation，再由共同 `TenantSkillRegistry` 通过进程级 Tier 2 cache 解析可信 Tenant-scoped storage identity，并让 Local 或 GCS Loader 每次实时读取 Tier 3 references，不建立第二份完整 definition 或 reference cache。

Tool 以结构化结果明确区分 loaded、not-activated 和 load-failed。Local/GCS 的共同 contract 统一为只读取 `references/` 的直属普通文件并按文件名排序；目录缺失或为空返回成功空集，列举、symlink 状态变化或任一文件读取失败均返回有界失败且不暴露部分内容。References 不参与 Specialist/Skill pins，也不改变冻结的业务 Tool 或 Research Scope。

## Comments

- 拆分及验收范围经独立 GPT-6 Astra agent 复审为 PASS；本票交付的读取合同是 GCS Adapter 的前置条件。
- 2026-09-17：确定性 FunctionModel 已覆盖 `activate_skill → load_reference → 修改 Local reference → 再次 load_reference → 业务 Tool → 最终结果`，第二次调用读取到最新文件，pins 与有效业务 Tool set 保持不变。
- 2026-09-17：Local/GCS 共享 Adapter contract 覆盖确定性顺序、实时内容、只读直属普通文件、缺失/空目录、列举失败、单文件失败不返回部分内容和 Tenant 隔离；invocation 另覆盖跨 invocation activation 不授权和跨 Tenant definition 拒绝。
- 2026-09-17：unit 与 eval 共 384 项通过；Ruff、受影响文件严格 Pyright、`git diff --check` 均通过。
- 2026-09-17：使用 `scripts/run-pytest` 在 sandbox 外执行 3 项受影响 PostgreSQL HTTP/graph 集成测试时，本机 `localhost:5432` 未运行，全部停在 session fixture 建连阶段 `connection refused`，未进入测试断言。
- 2026-09-17：最终全仓 pytest 按要求执行一次，但被仓库根目录既有 `test_stream_union.py` 与 `test_union2.py` 的 collection-time PydanticAI 错误提前中止；两文件不属于本票变更。
- 2026-09-17：双轴 code review 首轮发现异常边界过宽、runtime protocol 命名和 Local/GCS 普通文件语义三项问题，均通过收窄 boundary catch、共享 Registry seam 与参数化 Adapter contract 解决；Standards 与 Spec 复查最终均 PASS，未解决 review comments：0。
