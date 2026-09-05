# 14: 证明完整金融 golden path

**What to build:** 通过真实 HTTP/SSE、LangGraph、PydanticAI adapter 与 PostgreSQL checkpoint 边界，演示一个固定金融 fixture 的完整多轮 fan-out/fan-in、Skill、Tool、Calculation、follow-up 与最终报告路径。

**Blocked by:** 06: 渐进激活 Specialist-owned Skills; 13: 证明 research 与 clarification 终态遵守 committed-state publication

**Status:** ready-for-agent

- [ ] 使用固定 as-of date、synthetic fund ID 与 benchmark ID，全部模型与业务数据由 deterministic fakes 提供，不访问真实 provider。
- [ ] typed registry 只提供 golden path 所需的 market-analysis 与 fund-research Specialists，以及 price-series、fund-holdings、fund-reports、company-news 四个 mock registered business Tools。
- [ ] 第一个 batch 并行派发 market analysis 与 fund holdings/disclosure research，并在真实 LangGraph barrier 上 deterministic collect。
- [ ] market Specialist 在自身 invocation 内通过 registered calculation Tools 执行 period return、annualized volatility 与 maximum drawdown；这些 calculation 不是顶层 Tasks。
- [ ] fixture 至少配置一个 shared Skill 和一个 Specialist-scoped Skill；scripted Specialist 先经真实 progressive activation boundary 选择 eligible Skill，再使用已经授权的 Tools，不能预加载完整 Skill instructions。
- [ ] barrier 后的 company-news follow-up Task 通过 `context_task_ids` 明确选择已接受的 fund-research Result；下一次 Coordinator Decision 为 `Finish` 并进入 Synthesis。
- [ ] 最终报告的 Evidence markers、citations、calculation aliases、code-rendered values、canonical assistant Message、token stream 与 `done.answer` 一致并通过所有 gates。
- [ ] 外部测试确实穿过同一个 `/v2/query/stream`、真实 LangGraph runtime、PydanticAI actors/adapters 与 PostgreSQL checkpointer，而不是用节点直调替代。
- [ ] 使用 alternate holdings outcome 证明 barrier 后的下一轮 Coordinator 决策会改变，但仍在既定 round/task bounds 内终止。
- [ ] 同一路由上的 Linear-configured 与 Agent-configured Tenants 都保持各自行为，request input 不能覆盖模式。
- [ ] 不同 Conversations 可以独立执行；本 ticket 不引入同一 Conversation 的并发 Request 测试或锁协议。
- [ ] 同一个 Agent Graph builder 能通过 dependency injection 装配 fake actors 与 Specialist registry，graph control 不导入金融 actor 实现。

