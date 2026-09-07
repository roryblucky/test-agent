# 17: 建立 8–12 个 Pydantic Evals 金融回归案例

**What to build:** 把本 POC 的关键金融轨迹、最终输出与单步 gate 行为固化为小型、完全确定性的 Pydantic Evals 回归集，作为普通 CI 的长期防线。

**Blocked by:** 14: 证明完整金融 golden path

**Status:** done

- [x] 建立 8–12 个 application-owned financial cases，覆盖 canonical golden path 以及代表性的 alternate outcome、partial result、limit、negative/empty Evidence、invalid marker 与 clarification 行为。
- [x] 提供 deterministic trajectory evaluators，校验 Coordinator decisions、Task/round shape、Skill/Tool selection、accepted batches 与 invocation bounds。
- [x] 提供 final-output evaluators，校验 completion metadata、disclosure、citations、calculation rendering、canonical Message 与 `done.answer`。
- [x] 提供 single-step gate evaluators，校验 Scope/permission、Evidence/calculation provenance、support markers 与 fail-closed publication。
- [x] 普通 CI 全部使用 fake models、deterministic mock Tools 与固定 fixtures，设置严格 per-run limits，禁止网络访问或意外真实 model request。
- [x] evals 不替代 Ticket 08 的 PydanticAI adapter contract suite，也不替代 Ticket 14 的真实 HTTP/LangGraph/PostgreSQL integration path。
- [x] Azure 与 Google real-provider canaries、其 exact model identity、部署与 cost ceiling 留给 operator 后续决策，不作为本 ticket 或普通 PR blocker。
- [x] PydanticAI V2 migration 保持独立后续工作，不在本 eval ticket 中改变锁定的 V1 failure mapping。

## Comments

- Sol-medium seam review 选定「轻量 in-memory Agent Graph eval + 纯 public gate eval」：不反向 import Ticket14 测试，也不以 HTTP/SSE/PostgreSQL E2E 取代其 integration 覆盖。
- 新增显式锁定的 dev dependency `pydantic-evals==1.93.0`，并建立 11 个 cases：canonical、Task failure partial、Tool unavailable partial、authoritative empty Evidence、real coordination-limit、Finish-first、clarification，及 Scope、Evidence provenance、Calculation provenance、invalid support marker 四个 gate。
- Run cases 由编译后的实际 Agent Graph 执行；fixture 只提供 local Coordinator/Specialist/Synthesis actors 与固定 Tool providers。观测由 final graph state 和 custom stream events 派生：accepted rounds/batches、Tool/Skill、每 Task attempts/Tool calls、usage、final response、citations、token、canonical assistant Message、`done.answer`。
- final evaluator 固定比较完整 code-owned incomplete disclosure block、exact period-return code rendering、structured clarification、citations、metadata 与 channel identity；gate evaluator 断言具体 fail-closed type/message。Pydantic Evals 的 false assertion 会被 pytest 显式失败。
- `ALLOW_MODEL_REQUESTS=False` 全局阻断真实模型；全部 providers 是 in-memory fixtures，外层 eval serial，且验证 Task/Tool/round/dispatch 上限。Azure/Google canary、PydanticAI V2 migration 与 Ticket08/Ticket14 覆盖保持 out of scope。
- Sol-high Standards / Spec 初审指出自报轨迹、test-built final output、宽泛 gate catch、usage 计数及 coarse final assertion；均已以 real Graph captures、targeted errors、per-invocation usage、exact output assertions 关闭。终审均为 0 actionable findings。
- 验证：`scripts/run-pytest -q tests/evals/test_langgraph_v2_financial_evals.py`、`uv run ruff check app tests alembic/versions/0018_drop_conversation_registry.py`、`uv run pyright app tests`、`scripts/run-pytest tests -q`、`git diff --check` 通过（450 passed, 1 skipped）。
- Unresolved review comments: 0.
