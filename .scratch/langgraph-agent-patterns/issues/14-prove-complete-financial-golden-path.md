# 14: 证明完整金融 golden path

**What to build:** 通过真实 HTTP/SSE、LangGraph、PydanticAI adapter 与 PostgreSQL checkpoint 边界，演示一个固定金融 fixture 的完整多轮 fan-out/fan-in、Skill、Tool、Calculation、follow-up 与最终报告路径。

**Blocked by:** 06: 渐进激活 Specialist-owned Skills; 13: 证明 research 与 clarification 终态遵守 committed-state publication

**Status:** done

- [x] 使用固定 as-of date、synthetic fund ID 与 benchmark ID，全部模型与业务数据由 deterministic fakes 提供，不访问真实 provider。
- [x] typed registry 只提供 golden path 所需的 market-analysis 与 fund-research Specialists，以及 price-series、fund-holdings、fund-reports、company-news 四个 mock registered business Tools。
- [x] 第一个 batch 并行派发 market analysis 与 fund holdings/disclosure research，并在真实 LangGraph barrier 上 deterministic collect。
- [x] market Specialist 在自身 invocation 内通过 registered calculation Tools 执行 period return、annualized volatility 与 maximum drawdown；这些 calculation 不是顶层 Tasks。
- [x] fixture 至少配置一个 shared Skill 和一个 Specialist-scoped Skill；scripted Specialist 先经真实 progressive activation boundary 选择 eligible Skill，再使用已经授权的 Tools，不能预加载完整 Skill instructions。
- [x] barrier 后的 company-news follow-up Task 通过 `context_task_ids` 明确选择已接受的 fund-research Result；下一次 Coordinator Decision 为 `Finish` 并进入 Synthesis。
- [x] 最终报告的 Evidence markers、citations、calculation aliases、code-rendered values、canonical assistant Message、token stream 与 `done.answer` 一致并通过所有 gates。
- [x] 外部测试确实穿过同一个 `/v2/query/stream`、真实 LangGraph runtime、PydanticAI actors/adapters 与 PostgreSQL checkpointer，而不是用节点直调替代。
- [x] 使用 alternate holdings outcome 证明 barrier 后的下一轮 Coordinator 决策会改变，但仍在既定 round/task bounds 内终止。
- [x] 同一路由上的 Linear-configured 与 Agent-configured Tenants 都保持各自行为，request input 不能覆盖模式。
- [x] 不同 Conversations 可以独立执行；本 ticket 不引入同一 Conversation 的并发 Request 测试或锁协议。
- [x] 同一个 Agent Graph builder 能通过 dependency injection 装配 fake actors 与 Specialist registry，graph control 不导入金融 actor 实现。

## Comments

- Scope/test-seam review（gpt-5.6-sol medium）：确认范围只新增确定性金融 golden fixture；公开 seam 为 `/v2/query/stream` → Agent tenant → LangGraph → PydanticAI `FunctionModel` → registered Tools → PostgreSQL checkpoint。四个业务 Evidence Tools 与三个 Specialist-internal Calculation methods 分别注册；不扩展生产金融 provider、图控制流或同 Conversation 并发协议。
- Mode checklist evidence：已有 `tests/integration/test_langgraph_v2_runtime_mode.py::test_agent_tenant_query_ignores_client_mode_override` 覆盖同路由 Agent tenant 忽略 client `mode=linear`；本票主 fixture 使用该相同服务端路由，完整 suite 已复跑。
- Code review（gpt-5.6-sol high，Standards）：两项 finding 已关闭：补齐三种 Calculation 的 method/unit/currency/period/assumptions code-rendered disclosure 断言，并逐字段断言四条 citation 的排序与 Evidence identity/binding；终审为 0 actionable findings。
- Code review（gpt-5.6-sol high，Spec）：补齐成功终态 `completion_status=complete`、canonical `termination_reason=evidence_backed`、无 `Incomplete research:` disclosure 与 checkpoint `incomplete_research is None` 的断言；终审为 0 actionable findings。审阅中“无 termination_reason”的建议以 production canonical contract 为准处置为 `evidence_backed`。
- Verification：`uv run ruff check app tests alembic/versions/0018_drop_conversation_registry.py`、`uv run pyright app tests`、`scripts/run-pytest tests -q` 通过（443 passed、1 skipped）。
- Unresolved review comments：0。
