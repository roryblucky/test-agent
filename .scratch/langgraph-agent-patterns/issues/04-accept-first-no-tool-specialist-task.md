# 04: 接受首个无 Tool 的 Specialist Task

**What to build:** 让 Coordinator 派发一个无 Tool 的 Specialist Task，经 typed registry、immutable contribution 与 barrier 原子接受，并在零 Evidence 时保守结束。

**Blocked by:** 03: 安全继续 Agent Conversation 并处理 `Finish`-first

**Status:** ready-for-agent

- [ ] 单个合法 `Dispatch` 只能选择已注册且同时满足 Tenant 与 Scope 资格的 Specialist，并包含非空 objective；模型不得提供 tool、skill、limit 或 ID。
- [ ] Graph 根据 Run、Round 与批内顺序生成稳定 Task ID。
- [ ] 通用 `execute_specialist` 通过 typed registry 调用 actor，并产出最小 `TaskSucceeded`；usage、retry 与 diagnostics 不进入 `TaskOutcome`。
- [ ] contribution 不可变；单一 barrier 校验 manifest、identity 与 attempt 后，在一个状态更新中 promote accepted contribution 并清空 staging。
- [ ] accepted success 没有 Evidence 时复用 `insufficient_evidence` 的保守发布语义。
- [ ] 静态 builder 使用 fake Coordinator 与 Specialist registry；graph control 不得导入金融领域实现。
- [ ] Agent builder 与 Linear builder 分离；官方 checkpoint 是唯一 durable authority。不得新增应用自有 Run model/repository/table、重复 checkpoint pointer、transport Event journal、Redis recovery、`AgentTeamDefinition`、Team registry、配置 loader 或 workflow DSL；Run/Task identity 不是产品持久化实体。
- [ ] Specialist 首次调用关闭隐藏 Tool/output retry，使用显式 end strategy，单次模型请求超时 60 秒，`max_tokens` 为 2000。
- [ ] Specialist Result 的 canonical JSON 上限为 16 KiB；测试覆盖恰好 16 KiB 与多 1 byte，超限在生成 contribution 前作为 structured-output-invalid 拒绝。
- [ ] 使用 PydanticAI 的基础 `TestModel`/`FunctionModel`、actor override 与 `ALLOW_MODEL_REQUESTS=False`；每次 invocation 独立 capture，并分别断言 output schema、`new_messages`、usage 和精确的一请求轨迹。

