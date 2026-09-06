# 02: 发布已提交的 Agent clarification Run

**What to build:** 让受信任的 Agent 模式通过现有流式查询入口完成首条端到端 clarification Run，并且只从已提交的 checkpoint 状态发布结果。

**Blocked by:** 01: 锁定共享 checkpoint 序列化与 typed runtime seam

**Status:** done

- [x] 受信任的 Tenant 配置通过同一个 `/v2/query/stream` 路由选择 Agent 模式，客户端不得覆盖模式。
- [x] 每个新 Run 只有一个 initializer 首先执行；集合 channel 使用 Overwrite，标量按普通赋值重置。
- [x] Query Understanding 是唯一能读取有界、完整历史消息对的 actor。
- [x] clarification 路径绕过 Intent、Scope、Coordinator、Specialist 与 Synthesis；`done.answer` 返回结构化 clarification，citations 为空。
- [x] clarification 的最终 user/assistant 消息对在 `done` 前同步提交；受控测试覆盖提交前取消与提交后取消。
- [x] pre-moderation flagged 路径不生成 research completion、`IncompleteResearch`、最终 assistant Message 或正常 `done`。
- [x] 用户回答 clarification 时，在同一 Conversation 中使用新的 Request ID，经干净 initializer 开始新的 Run；不得把它当作 interrupt/resume，也不得继承前一 Run 的控制状态。
- [x] clarification follow-up 只有 Query Understanding 能读取 clarification 消息对，随后按普通新 Run 继续；复用同一 Request ID 不得被当作 resume。

## Comments

- 2026-09-06：Sol-medium 确认三 seam：HTTP/SSE+checkpoint、Agent graph builder、受控 checkpoint barrier；不预设 Team、DSL、research DAG 或 Agent state。
- 2026-09-06：实现 clarification-first Agent runtime。唯一 initializer 保留 Conversation Messages 并普通重置当前已有标量；本票尚无 run-local reducer collection，后票引入时依 spec 以 `Overwrite` 清除。Query Understanding 独占完整历史对；无 clarification 的 Scope/Coordinator/Finish continuation 明确由 03 接续。
- 2026-09-06：`finalize_state` 先同步 checkpoint canonical response 与 assistant Message，`publish` 后发 `done`。受控 saver 覆盖 commit 前/后取消：前者仅留 user，后者留完整 pair，皆无 `done`。
- 2026-09-06：Sol-high 初审：删未用 AgentGraph protocol；response 改名 `V2QueryResponse`；补 clarification 1–3 非空 questions、每题最多 4 options 的 public rejection 覆盖。Linear final-response 采用 wire `session_id`，并接受既有 `conversation_id`；此为全套既有验收断言所揭兼容修，非破坏变更。
- 2026-09-06：Sol-high 复审通过。非-clarification continuation 与 publication manifest 分别由 03、13 明确拥有；未解决 review comments：0。
- 验证：`pytest tests`：259 passed、1 skipped；`ruff check app tests`、`pyright --pythonpath .venv/bin/python` 均通过。PostgreSQL integration 使用本机专用 test DB。
