# 01: 统一 code-defined Specialist Catalog 与 Tool 权限

**What to build:** 当前 Tenant 的 code-defined Specialists 通过共同 Specialist Catalog 向 Coordinator 提供描述并执行 Task。不同 Business Intent 可以选择相同的 Tenant Specialists，但业务 Tool 调用仍受各自 Research Scope 约束。为后续 Markdown Adapter 建立共同执行合同。

**Blocked by:** None (can start immediately).

**Status:** resolved

依据：已批准的 Tenant-authored Specialist Definitions 规格及 ADR 0008。

- [x] code-defined registration 提供 Specialist ID、description 和构建 actor 所需的可信定义信息；Coordinator descriptors 来自当前 Tenant Catalog，不再来自 Intent-specific descriptor 配置。
- [x] 两个 Business Intents 获得同一 Tenant 的完整有效 Specialist Descriptors，同时保留各自 Tool、source、query 和 freshness 约束；Intent Result 和 Intent Catalog item 结构不变。
- [x] 不同 Tenant 的 catalog 和执行上下文隔离；Coordinator 选择未知或跨 Tenant Specialist ID 时被确定性拒绝。
- [x] Agent Graph 所用 Evidence 和 Calculation Tools 通过共同的平台 registry/binding seam 解析。有效 Tools 为全局已注册 Tools、Tenant Tool policy 和 Research Scope 的交集，不再依赖 Specialist Tool allowlist。
- [x] 构建 actor 前冻结有效 Tool bindings；现有 Evidence、Calculation、expected unavailability 和 Data Gap 的代码校验继续有效。
- [x] 保留现有 SpecialistTaskInput、SpecialistAttempt、SpecialistResult 和 Task Outcome 合同，建立可供两种 Adapter 复用的 code Adapter 合同测试基线。
- [x] 通过现有 HTTP/SSE Agent 集成入口演示 code-defined Specialist 被选择、调用允许的 Tool 并完成 Task；拒绝 scope 外的 Tool。
- [x] 只调整 Agent Graph 所需的 registry/binding 和调用点，不顺带迁移整个 legacy BuiltInToolRegistry、重设计 Tool schemas 或改变 FlowEngine 行为。
- [x] 聚焦确定性测试及受影响回归测试通过；PostgreSQL-backed tests 使用项目标准测试 runner。

## Answer

已实现 Tenant-scoped `SpecialistCatalog`、共同 `AgentToolRegistry` 与 code Adapter 合同。Coordinator 始终读取当前 Tenant Catalog；执行时只接受 Catalog 成员，并在构建 actor 前冻结全局注册、Tenant policy 与 Research Scope 的 Tool 交集。`SpecialistRegistration` 必须且只能提供 `actor` 或 `actor_factory`；直接测试 actor 不接收 Tool/Skill bindings，factory actor 使用冻结后的 bindings。

## Comments

- 拆分及验收范围经独立 GPT-6 Astra agent 复审为 PASS；本记录针对规划，不代表实现已通过 review。
- 2026-09-16：确定性 unit/eval 回归共 345 项通过；Ruff 与受影响文件 Pyright 通过；24 项 PostgreSQL 集成测试收集成功。
- 2026-09-16：已使用 `scripts/run-pytest` 执行 PostgreSQL-backed tests，但本机 `localhost:5432` 未运行，fixture 在任何测试断言前因 connection refused 停止。
- 2026-09-16：实现完成两轴 code review；Standards 与 Spec 最终均 PASS，未解决 review comments：0。
