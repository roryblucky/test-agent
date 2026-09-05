# 17: 建立 8–12 个 Pydantic Evals 金融回归案例

**What to build:** 把本 POC 的关键金融轨迹、最终输出与单步 gate 行为固化为小型、完全确定性的 Pydantic Evals 回归集，作为普通 CI 的长期防线。

**Blocked by:** 14: 证明完整金融 golden path

**Status:** ready-for-agent

- [ ] 建立 8–12 个 application-owned financial cases，覆盖 canonical golden path 以及代表性的 alternate outcome、partial result、limit、negative/empty Evidence、invalid marker 与 clarification 行为。
- [ ] 提供 deterministic trajectory evaluators，校验 Coordinator decisions、Task/round shape、Skill/Tool selection、accepted batches 与 invocation bounds。
- [ ] 提供 final-output evaluators，校验 completion metadata、disclosure、citations、calculation rendering、canonical Message 与 `done.answer`。
- [ ] 提供 single-step gate evaluators，校验 Scope/permission、Evidence/calculation provenance、support markers、size limits 与 fail-closed publication。
- [ ] 普通 CI 全部使用 fake models、deterministic mock Tools 与固定 fixtures，设置严格 per-run limits，禁止网络访问或意外真实 model request。
- [ ] evals 不替代 Ticket 08 的 PydanticAI adapter contract suite，也不替代 Ticket 14 的真实 HTTP/LangGraph/PostgreSQL integration path。
- [ ] Azure 与 Google real-provider canaries、其 exact model identity、部署与 cost ceiling 留给 operator 后续决策，不作为本 ticket 或普通 PR blocker。
- [ ] PydanticAI V2 migration 保持独立后续工作，不在本 eval ticket 中改变锁定的 V1 failure mapping。
