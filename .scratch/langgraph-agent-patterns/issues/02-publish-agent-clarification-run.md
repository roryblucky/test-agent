# 02: 发布已提交的 Agent clarification Run

**What to build:** 让受信任的 Agent 模式通过现有流式查询入口完成首条端到端 clarification Run，并且只从已提交的 checkpoint 状态发布结果。

**Blocked by:** 01: 锁定共享 checkpoint 序列化与 typed runtime seam

**Status:** ready-for-agent

- [ ] 受信任的 Tenant 配置通过同一个 `/v2/query/stream` 路由选择 Agent 模式，客户端不得覆盖模式。
- [ ] 每个新 Run 只有一个 initializer 首先执行；集合 channel 使用 Overwrite，标量按普通赋值重置。
- [ ] Query Understanding 是唯一能读取有界、完整历史消息对的 actor。
- [ ] clarification 路径绕过 Intent、Scope、Coordinator、Specialist 与 Synthesis；`done.answer` 返回结构化 clarification，citations 为空。
- [ ] clarification 的最终 user/assistant 消息对在 `done` 前同步提交；受控测试覆盖提交前取消与提交后取消。
- [ ] pre-moderation flagged 路径不生成 research completion、`IncompleteResearch`、最终 assistant Message 或正常 `done`。
- [ ] 用户回答 clarification 时，在同一 Conversation 中使用新的 Request ID，经干净 initializer 开始新的 Run；不得把它当作 interrupt/resume，也不得继承前一 Run 的控制状态。
- [ ] clarification follow-up 只有 Query Understanding 能读取 clarification 消息对，随后按普通新 Run 继续；复用同一 Request ID 不得被当作 resume。

