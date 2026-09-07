# 05: 发布首份 Scope-bound Evidence-backed Report

**What to build:** 让一个受 Scope 限制的 Specialist 调用真实 tool binding，写入 request cache，经过基础 Synthesis 与确定性 gate，发布首份带 citation 的报告。

**Blocked by:** 04: 接受首个无 Tool 的 Specialist Task

**Status:** done

- [x] effective tool set 在 Run 前冻结为 registered、Tenant、Scope 与 Specialist allowlist 的交集。
- [x] 直接请求无效 Specialist、Tool、source，或尝试放宽约束，均在数据访问前失败；tool arguments 只能进一步收窄范围。
- [x] `ToolReturn` 将 model-visible value 与 app-only typed Evidence metadata 分离，metadata 对模型不可见。
- [x] 只有已校验成功且被 `Finding` 引用的 body 才进入 request cache；missing、cross-Tenant、cross-Run、cross-Task 或冲突的 success provenance 均被拒绝。
- [x] checkpoint、Send、reducers 与 PostgreSQL pending writes 不包含 Evidence body 或 raw provider payload。
- [x] 相同 Evidence body 的重复写入幂等；同 ID 不同内容 fatal；orphan Evidence 不具备发布资格。
- [x] Specialist Result 最多引用 16 个 Evidence ID；测试覆盖 16 与 17，超限在 contribution 前作为 structured-output-invalid 拒绝，外层 retry 留给 Ticket 08。
- [x] 基础 Synthesis 只接收 eligible excerpt，输出带 marker 的 Markdown；代码校验 marker 语法与 Evidence eligibility，并由代码推导 citations。
- [x] Synthesis 首次调用没有 Tools、关闭隐藏 retry、超时 120 秒，`max_tokens` 为 4000。
- [x] Tool audit 以及 planning、Task、Tool 的增量 SSE 是旁路 telemetry，不进入 `TaskOutcome`；Specialist 内部 tool calls 不是顶层 Task。
- [x] Tool 对权威 negative/empty 查询仍产出 Evidence，不映射成 `ToolUnavailable`，并允许完整完成。

## Comments

- 2026-09-06: Implemented frozen registered ∩ Tenant ∩ Scope ∩ Specialist Tool
  binding, exact Scope-narrowed source/query arguments, request-local Evidence
  acceptance, freshness gate, ToolReturn isolation, Synthesis publication gate, and
  side-channel Task/Tool telemetry with isolated Tool audit.
- 2026-09-06: PostgreSQL fixture migrations are redirected to the approved
  fixture schema; they do not create or drop the application's global schema.
- 2026-09-06: Sol-high final review found zero remaining non-deferred Ticket 05
  issues. Outer retries were deferred to Ticket 08. Unresolved review comments: 0.
- 2026-09-06: Follow-up review findings resolved: Tool-capable Specialist naming
  is current; immutable `EvidenceInvocationContext` owns invocation identity and
  authority; Evidence idempotence excludes noncanonical `raw_provider_payload`;
  and PostgreSQL HTTP E2E proves authoritative empty results still publish an
  Evidence-backed citation.
- 2026-09-06: Standards follow-up removed the whole-Specialist timeout, moved the
  60-second bound to each model request, made Tool execution sequential, disabled
  provider parallel Tool calls, and removed unused freshness parameters from the
  acceptance interface. Binding-owned 20-second Tool timeout/unavailability is
  explicitly deferred to Ticket 07; outer retries remain Ticket 08. The
  fixture-schema concern was dismissed: validation, lifecycle,
  Alembic configuration, SQL translation, and migration behavior each have one
  distinct owner, and the PostgreSQL migration/E2E tests verify the composed seam.
  Final Standards/Spec re-review: zero unresolved Ticket 05 findings. Unresolved
  review comments: 0.
