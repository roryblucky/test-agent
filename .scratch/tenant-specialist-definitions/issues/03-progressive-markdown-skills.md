# 03: Markdown Skills 渐进式多激活与记录

**What to build:** Tenant 管理员为 Markdown Specialist 声明 Skills；模型先看到简要 summaries，再按需激活零个或多个 Skills，并与业务 Tool 调用交错。每次已接受执行记录实际使用的 Skill Pins。

**Blocked by:** 02 — 本地 Markdown Specialist 从启动加载到完成 Task。

**Status:** resolved

依据：已批准的 Tenant-authored Specialist Definitions 规格及 ADR 0008。

- [x] 复用或深化现有 Agent Skills schema、parser 和 Local storage 读取能力；启动时加载完整 Skill definitions，references 内容不预读。
- [x] 将共同 Skill Catalog 适配为 invocation-local state，替换 Agent Graph 平行 SkillRegistration 及内嵌缓存 references 语义，不创建第二套持久化 Skill schema/parser，不重构 legacy Agent Handler 的行为。
- [x] 候选 Skills 仅来自 Specialist Definition 显式声明和当前 Tenant Skill Catalog；移除 Intent allowed_skill_names 的筛选作用以及 shared_skill_names 的隐式候选注入。
- [x] required-tools 是 eligibility 硬依赖；allowed-tools 仅是使用指导，不授予、筛选或扩大有效 Tools。任一字段引用全局未知 Tool 时，Skill 启动校验失败并被跳过。
- [x] required-tools 中 Tool 被 Tenant policy 或 Research Scope 排除时，Skill 不进入 summaries；allowed-tools 中已注册但当前不可用的 Tool 不影响 eligibility。
- [x] Specialist 引用不存在或无效的 Skill 时，跳过该 Specialist 并记录原因；多个 Specialists 可以共享同一 Skill definition。
- [x] 按 Specialist 声明顺序过滤后最多暴露前二十条 summaries。invocation 开始时确定固定 Eligible Skill set；未知、未暴露和不合格的 Skill 均不能靠猜名激活。
- [x] 初始模型上下文只有 summaries，没有完整 Skill instructions；activate_skill 才披露已缓存的 instructions。允许零个或多个激活，并允许业务 Tool 调用发生在两次激活之间。
- [x] 同一 Skill 重复 activation 幂等，Activated Skills 持续至本次 invocation 结束，pins 按首次激活顺序去重；activation 不改变有效 Tool bindings。
- [x] Skill Pin 包含 name、必填 content hash 和可选 metadata.version。无 version 的有效 Skill 可以完成 activation、acceptance、checkpoint 持久化及 telemetry；references 不参与 hash。
- [x] 进程内 Skill 文件变更不改变缓存内容；两个独立 startups 与后续 retry 验证新进程使用新 Skill 内容和新 pin，无 version 时同样成立。此验收由本票负责，不依赖 06。
- [x] 扩展 loaded/skipped 计数、内容不泄漏日志、accepted telemetry 和向后兼容 checkpoint 合同到 Skills。
- [x] 通过现有 HTTP/SSE 集成和确定性模型消息 sentinels 验证完整链路、渐进披露、交错调用与固定 Tool set；复用两种 Specialist Adapter 的共同合同。
- [x] 不增加 required-skills、max_activated_skills、Skill precedence、语义冲突检测或自动 fallback；不执行 bundled scripts。
- [x] 聚焦确定性测试及受影响回归测试通过。

## Answer

已将现有 Agent Skills schema 与 parser 深化为 Tenant Skill Catalog，并适配到 Specialist invocation-local discovery、零到多次渐进激活和首次使用顺序的 Skill Pins。启动期只缓存完整 Skill definition，不读取 references；required-tools 决定 eligibility，allowed-tools 只提供指导，任何 Tool 权限仍由平台 registry、Tenant policy 与 Research Scope 的交集决定。无版本 Skill 使用 content hash 完成 acceptance、checkpoint 和 telemetry，重复激活不会重复披露 instructions 或 pin。

## Comments

- 拆分及验收范围经独立 GPT-6 Astra agent 复审为 PASS；Skill startup、restart 和 pin 变更测试归本票，避免为 06 增加无关依赖。
- 2026-09-17：Ruff、受影响文件 Pyright、`git diff --check` 均通过；unit 与 financial eval 共 361 项通过；4 项关键 PostgreSQL 集成测试成功收集。
- 2026-09-17：使用 `scripts/run-pytest` 执行 4 项 PostgreSQL-backed 场景时，本机 `localhost:5432` 未运行，解除 sandbox 后仍在 fixture 建立连接阶段 `connection refused`，未进入测试断言。
- 2026-09-17：最终全套测试被仓库根目录既有 `test_stream_union.py` 与 `test_union2.py` 的 collection-time PydanticAI 错误提前中止；这两个文件不属于本票变更。
- 2026-09-17：review 的重复 source/loader 结构与 Specialist-only 命名已通过共同 Tenant definition loader、明确的 Skill 日志和模块命名解决。重复 activation 泄漏 instructions/pin、缺少第二次 startup retry pin 验证、HTTP 未覆盖多 Skill 与业务 Tool 交错、以及 startup registry 隐式 fallback 均已通过代码和测试解决。
- 2026-09-17：关于“默认生产 registry 必须非空”的 review 要求经证据复核后修正：仓库当前没有满足 Evidence provenance 或 Calculation trusted-series 合同的生产 registrations，legacy FlowEngine callables 不能冒充。composition 现改为显式注入 typed registry，Local loader 强制接收 registry，默认 deployment 明确使用空 registry，未知 Tool 继续 fail closed，不注册占位能力。
- 2026-09-17：完成两轴 code review；Standards 与 Spec 最终均 PASS，未解决 review comments：0。
