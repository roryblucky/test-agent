# 01: 统一 code-defined Specialist Catalog 与 Tool 权限

**What to build:** 当前 Tenant 的 code-defined Specialists 通过共同 Specialist Catalog 向 Coordinator 提供描述并执行 Task。不同 Business Intent 可以选择相同的 Tenant Specialists，但业务 Tool 调用仍受各自 Research Scope 约束。为后续 Markdown Adapter 建立共同执行合同。

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

依据：已批准的 Tenant-authored Specialist Definitions 规格及 ADR 0008。

- [ ] code-defined registration 提供 Specialist ID、description 和构建 actor 所需的可信定义信息；Coordinator descriptors 来自当前 Tenant Catalog，不再来自 Intent-specific descriptor 配置。
- [ ] 两个 Business Intents 获得同一 Tenant 的完整有效 Specialist Descriptors，同时保留各自 Tool、source、query 和 freshness 约束；Intent Result 和 Intent Catalog item 结构不变。
- [ ] 不同 Tenant 的 catalog 和执行上下文隔离；Coordinator 选择未知或跨 Tenant Specialist ID 时被确定性拒绝。
- [ ] Agent Graph 所用 Evidence 和 Calculation Tools 通过共同的平台 registry/binding seam 解析。有效 Tools 为全局已注册 Tools、Tenant Tool policy 和 Research Scope 的交集，不再依赖 Specialist Tool allowlist。
- [ ] 构建 actor 前冻结有效 Tool bindings；现有 Evidence、Calculation、expected unavailability 和 Data Gap 的代码校验继续有效。
- [ ] 保留现有 SpecialistTaskInput、SpecialistAttempt、SpecialistResult 和 Task Outcome 合同，建立可供两种 Adapter 复用的 code Adapter 合同测试基线。
- [ ] 通过现有 HTTP/SSE Agent 集成入口演示 code-defined Specialist 被选择、调用允许的 Tool 并完成 Task；拒绝 scope 外的 Tool。
- [ ] 只调整 Agent Graph 所需的 registry/binding 和调用点，不顺带迁移整个 legacy BuiltInToolRegistry、重设计 Tool schemas 或改变 FlowEngine 行为。
- [ ] 聚焦确定性测试及受影响回归测试通过；PostgreSQL-backed tests 使用项目标准测试 runner。

## Comments

- 拆分及验收范围经独立 GPT-6 Astra agent 复审为 PASS；本记录针对规划，不代表实现已通过 review。
