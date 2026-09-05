# 03: 安全继续 Agent Conversation 并处理 `Finish`-first

**What to build:** 让已消歧的 Agent 请求安全进入受信任 Scope 与 Coordinator，并支持第一轮直接 `Finish`、没有 Evidence 的保守终态。

**Blocked by:** 02: 发布已提交的 Agent clarification Run

**Status:** ready-for-agent

- [ ] 新 Request 只读取完整历史消息对，并清空全部 run-local accepted、staging、answer、error、diagnostic 与 counter 状态。
- [ ] Query Understanding 产出非空 standalone query 与受信任 Intent；未知 Intent 在任何数据访问前 fatal。
- [ ] 系统从 Intent 构造不可变 Research Scope；模型不得命名或授予 tool、skill、source、filter。
- [ ] Coordinator 只接收 standalone query、Intent 和经 Scope 过滤的描述符，不接收 Conversation 历史。
- [ ] Coordinator 首次调用没有 tools，关闭框架内置 Tool retry 与 output retry，超时 60 秒，`max_tokens` 为 1500。
- [ ] 第一条合法 Coordinator 决策可以是无 payload 的 `Finish`。
- [ ] 没有 Evidence 时跳过 Synthesis，生成 `IncompleteResearch`，`termination_reason` 为 `insufficient_evidence`，并且固定披露文本恰好出现一次。
- [ ] clarification 与 pre-moderation 不适用 research completion；带有 Evidence 的权威 negative/empty result 可以完整完成，不自动视为不足。

