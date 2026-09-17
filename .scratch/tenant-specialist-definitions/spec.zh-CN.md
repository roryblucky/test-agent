# Tenant 自定义 Specialist Definition

Status: ready-for-agent

本文是同目录英文规格的简体中文版本，两者描述相同的实现边界。

## 问题陈述

Tenant 管理员目前不能在不修改 Python 注册代码的情况下新增或调整
Specialist。Specialist 的身份、描述、instructions、model profile 和 Skills
本质上是声明式内容，不应依赖应用代码发布。

目标是借鉴 VS Code 的 Markdown Agent authoring 方式，让 Tenant 管理员通过
`.agent.md` 定义 Specialist，同时复用仓库现有的 Agent Skills 物理三层加载。
实现必须保持现有 Agent Graph 的 Task、Result、Evidence、Calculation、SSE 和
Tenant 隔离合同，不修改模型生成的 Intent Result 或 Intent Catalog 结构。

## 解决方案

应用启动时从 Tenant-scoped definition tree 加载 definitions。开发环境使用本地
目录，生产环境使用同一相对结构的 GCS prefix：

```text
tenants/{kmsAppId}/
├── agents/
│   └── {specialist-id}.agent.md
└── skills/
    └── {skill-name}/
        ├── SKILL.md
        └── references/
```

`kmsAppId` 就是 Tenant ID。Tenant 身份来自可信路径和 request context，而不是
Markdown。Local 与 GCS Loader 产出相同的 catalog 合同。

`.agent.md` frontmatter 只有：

- `id`
- `description`
- `model-profile`
- `skills`

Markdown body 是 Specialist instructions。不增加 role、Intent allowlist、直接 Tool
allowlist、自定义输入或输出 schema。`model-profile` 必须属于平台批准的 Model
Registry；instructions 最长 30,000 字符。

Coordinator 只看到当前 Tenant Catalog 中全部有效 Specialist 的 `id` 和
`description`。Intent 继续定义可信的数据 scope，不增加
`allowed_specialist_ids`，也不筛选 Skills。

## Agent Skills 三层加载

Specialist 直接使用现有 Agent Skills loader 和 `TenantSkillRegistry`：

1. 启动 discovery 只加载 Skill metadata summaries；
2. 模型调用 `activate_skill` 时，Registry 按需加载完整 `SKILL.md` instructions，
   Tier 2 definition cache 保持现有进程级语义；
3. 模型调用 `load_reference` 时，每次实时读取 `references/` 的直属普通文件，
   不缓存 reference 列表或内容。

一个 Specialist invocation 的 Eligible Skills 只由 `.agent.md` 中声明的名称和当前
Tenant Skill Catalog membership 决定，按声明顺序暴露全部有效 summaries。
invocation 可以激活零个或多个 Skills，并把 activation 与业务 Tool 调用交错。
重复 activation 幂等；Activated Skill 在本次 invocation 结束前持续有效。

不增加 `required-skills`、`max_activated_skills`、Skill 优先级、语义冲突检测或自动
fallback。Skill 冲突属于 Tenant 内容质量问题，通过发布 review 和 eval 发现。

`load_reference` 只允许访问当前 Tenant、当前 invocation 已激活的 Skill。目录缺失
或为空时返回成功空集；列举失败或任一文件读取失败时返回明确的有界失败，不返回
部分内容。reference 修改在下一次调用立即可见。

## Tool 权限

Tool definitions 只来自平台全局 Registry。Markdown Specialist 的有效业务 Tools 为：

```text
Global Tool Registry
∩ Tenant Tool Policy
∩ Research Scope
∩ Specialist 声明的有效 Skills 的 allowed-tools 并集
```

`allowed-tools` 是限制上限，不是授权来源。任何名称必须在平台 Registry 中存在，
否则该 Skill 在启动时被跳过。Skill 未列出的 Tool 不会绑定给该 Specialist，即使
更宽的平台 policy 允许它。

Graph Specialist runtime 不使用 `required-tools` 做第二套 eligibility policy。
该字段属于现有 FlowEngine Agent Skills 实现，Graph 路径忽略它；不要求 Tenant
管理员删除旧 FlowEngine 内容，也不复制一套 Graph Skill schema。

Skill activation 和 reference loading 不重新绑定业务 Tools。Skill package 中的
scripts、binaries 或 assets 不执行。

## Instructions 优先级

固定顺序是：平台 Specialist instructions、平台 security guards、Tenant Specialist
instructions；Activated Skill instructions 之后通过 activation result 进入 invocation。

文字顺序不是安全边界。Tenant identity、Research Scope、Tool bindings、Evidence 和
Calculation validation、coordination limits 与 graph routing 始终由代码控制。
Tenant instructions 不能覆盖或扩大这些规则。

## Graph 合同与状态

保留现有共同合同：

- 模型产生 `TaskProposal`；代码校验后生成 `AcceptedTask`。
- Specialist 接收 `SpecialistTaskInput`，返回 `SpecialistAttempt`。
- 确定性 acceptance 生成 `SpecialistResult` 和 `TaskOutcome`。
- batch barrier 收齐精确 manifest 后原子生成 `AcceptedBatch`。

这些是必要的信任边界，不属于 Markdown authoring 抽象。

Accepted Task manifest 只保存在 `CoordinationRound.tasks`。最后一个尚未生成
`AcceptedBatch` 的 dispatch round 就是待处理 batch；不再存在独立的
`ActiveBatch` 或 `CoordinationTask` 模型。

checkpoint 只保存 accepted facts：

- coordination 是否结束由 terminal `CoordinationRound` 或
  `coordination_stop_reason` 推导；
- 公共 response 中仍包含 `completion_status` 和 `termination_reason`，但 finalization
  从 `incomplete_research` 推导，不保存独立 checkpoint channels；
- 不保存 invocation-only failed-attempt diagnostics。

runtime 不生成或持久化 Specialist Definition Pins 或 Skill Pins。Pins 不参与权限
控制，而且在当前 catalog 执行、references 实时读取的设计下也不能提供精确历史重放。
Task outcomes、usage、启动日志和常规 traces 足够满足当前可观测性需求。

PydanticAI 版本由 `pyproject.toml` 和 lockfile 固定；runtime 不再重复实现依赖版本检查。

## 启动、恢复与错误处理

Agent 和 Skill definition 修改在重启后生效；同一进程使用已加载 catalog。References
保持实时。

无效 definition 逐条 skip 并记录 Tenant、source identity、definition kind 和稳定原因；
有效 siblings 继续加载。日志不得包含完整 instructions、references 或 secrets。

Coordinator 恢复后读取当前 Tenant Catalog，不保存历史 Specialist ID 快照。原 Run 的
Tool、source、query、freshness 等数据权限继续保留。已合法派发但当前 definition 已删除
的 Task 应独立产生失败 outcome；有效 sibling 继续。损坏 identity 或 accepted decision
仍属于 invariant failure。

## 验收标准

- Local startup 能加载 Markdown Specialist，通过现有 HTTP/SSE Agent 路径完成
  Coordinator selection、dispatch、Specialist execution 和 publication。
- Local/GCS 共享 loader contract，验证相同相对结构、Tenant prefix 隔离、逐条
  skip/log、实时 references、确定性文件顺序和无部分成功。
- Coordinator 对不同 Intents 看到同一当前 Tenant Specialist descriptors；各 Intent
  仍保留不同的数据 scope。
- Skill discovery 只暴露声明且有效的 summaries；覆盖零个、一个、多个、重复激活、
  未知名称和多个 activation 与业务 Tool 的交错。
- `allowed-tools` 真实缩小 Specialist 绑定的业务 Tool set；不能扩大 Registry、Tenant
  policy 或 Research Scope。
- `required-tools` 不影响 Graph Skill eligibility；现有 FlowEngine tests 保持其原合同。
- references 每次读取最新内容；空目录成功，读取错误有界且无部分内容。
- checkpoint 中没有重复 Task manifest、Pins、derived completion channels 或失败消息
  diagnostics；公共 completion metadata 保持不变。
- code-defined 与 Markdown-defined adapters 在迁移期间共享同一输入、输出、Evidence、
  Calculation 和 batch acceptance 合同。
- 聚焦 unit/eval、受影响 PostgreSQL integration、Ruff、Pyright 和 `git diff --check`
  通过；PostgreSQL tests 使用项目标准 runner 和 fixture 安全规则。

## 不在范围内

- Markdown Coordinator、Query Understanding 或 Synthesis。
- 精确兼容 VS Code custom agent 文件。
- 修改 Intent Result/Catalog schema，或增加 Intent-specific Agent/Skill allowlists。
- `required-skills`、`max_activated_skills`、Skill conflict runtime resolution。
- Agent/Skill hot reload、reference cache、历史 definition snapshot 或精确重放。
- Tenant catalog 的 atomic all-or-nothing loading。
- 执行 Skill package 中的代码。
- Tenant publishing UI、approval workflow 或 eval authoring UI。
- 重构 legacy FlowEngine request path。
