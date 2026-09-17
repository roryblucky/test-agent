# 02: 本地 Markdown Specialist 从启动加载到完成 Task

**What to build:** Tenant 管理员在本地定义 Markdown Specialist，应用启动后即可由 Coordinator 选择并通过现有 HTTP/SSE 路径完成 Task。

**Blocked by:** 01 — 统一 code-defined Specialist Catalog 与 Tool 权限。

**Status:** resolved

- [x] 在接收流量前，从可信 Tenant 路径加载已知 Tenants 的 `.agent.md`。
- [x] frontmatter 只支持 id、description、model-profile 和 skills；body 是 instructions。
- [x] 校验批准的 model profile、必需字段和 30,000 字符 instructions 上限。
- [x] 无效 definition 逐条 skip/log，有效 sibling 继续；日志不泄漏内容。
- [x] Local 与后续 GCS 共用 Loader interface 和 catalog model。
- [x] Markdown Adapter 使用现有 PydanticAI Specialist Actor、共同 Task/Result 合同和固定 instruction precedence。
- [x] 完整 instructions 不进入 checkpoint 或公共 SSE；不增加 definition hash、Pin 或历史 body 持久化。
- [x] 进程内 definition 保持启动快照；新进程读取修改后的内容。
- [x] 通过现有 HTTP/SSE Agent 路径验证加载、Coordinator selection、dispatch 和结果发布。

## Answer

已实现启动期 Local `.agent.md` 加载、严格 frontmatter/body 校验、Tenant 隔离、现有 PydanticAI Specialist Adapter、固定 instruction precedence，以及 code/Markdown 共同执行合同。后续简化删除了不参与权限控制或精确重放的 Definition Pin 链路。

## Comments

- 2026-09-17：实现与聚焦回归完成；未解决 review comments：0。
- 2026-09-17：架构复查确认 Definition Pin 属于非必要审计状态，已从 runtime、checkpoint、telemetry、测试和当前文档合同删除。
