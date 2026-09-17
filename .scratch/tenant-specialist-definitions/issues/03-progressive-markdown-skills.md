# 03: Markdown Skills 渐进式多激活

**What to build:** Tenant 管理员为 Markdown Specialist 声明 Skills；模型先看到 summaries，再按需激活零个或多个 Skills，并与业务 Tool 调用交错。

**Blocked by:** 02 — 本地 Markdown Specialist 从启动加载到完成 Task。

**Status:** resolved

- [x] 复用现有 Agent Skills schema、parser、`TenantSkillRegistry` 和 Local 三层读取能力，不创建第二套 Graph Skill loader。
- [x] startup discovery 只保留 summaries；activation 才加载完整 `SKILL.md`；references 不预读。
- [x] Eligible Skills 只来自 Specialist 显式声明和当前 Tenant Skill Catalog；Intent 不筛选 Skills。
- [x] `allowed-tools` 是 Specialist Tool ceiling：仅保留 Global Registry、Tenant policy、Research Scope 和声明 Skills `allowed-tools` 的交集。
- [x] Graph 不使用 `required-tools` 筛选 Skills；该 legacy 字段继续服务 FlowEngine。
- [x] Specialist 引用不存在或无效 Skill 时被跳过并记录原因；多个 Specialists 可以共享 Skill。
- [x] summaries 按声明顺序暴露全部有效项；未知或未声明 Skill 不能猜名激活。
- [x] 允许零个或多个 activation，允许业务 Tool 调用出现在 activation 之间；重复 activation 幂等。
- [x] 不保存 Skill Pins、完整 Skill instructions 或 activation transcript 到 checkpoint。
- [x] 不增加 required-skills、max_activated_skills、Skill precedence、冲突检测或自动 fallback；不执行 bundled scripts。

## Answer

Specialist 直接复用现有 Agent Skills 三层加载：共同 Registry 在启动时 discovery summaries，首次 activation 加载 Tier 2 instructions，Tier 3 references 由独立 live tool 读取。invocation-local 层只保存 eligibility 和 activation 状态。`allowed-tools` 真正缩小业务 Tool bindings；Pins 和 Graph `required-tools` eligibility 已删除。

## Comments

- 2026-09-17：修正为直接复用 agentskills.io 物理三层加载，删除重复 definitions/reference cache。
- 2026-09-17：架构复查进一步删除 Pins、Intent Skill filter 与 `required-tools` Graph policy；未解决 review comments：0。
