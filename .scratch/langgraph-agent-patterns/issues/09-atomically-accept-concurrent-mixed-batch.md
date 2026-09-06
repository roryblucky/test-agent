# 09: 原子接受单 Run 内并发 mixed Dispatch Batch

**What to build:** 让一个 Coordinator 决策在同一个 Run 内并行派发至多 8 个独立 Specialist Task，并只在整个 mixed success/failure batch 完整时原子接受结果。

**Blocked by:** 08: 按封闭 V1 映射重试 Specialist

**Status:** done

- [x] 每个 Dispatch Batch 最多 8 个 Task，测试覆盖恰好 8 与 9；执行 `max_concurrency` 为 8，所有校验在创建 Send 前完成。
- [x] 同批 Task 相互独立，不允许引用同批中其他 Task 的结果。
- [x] 每个完成项只写入带稳定 identity 与 attempt provenance 的 immutable staged contribution；staging reducer 具备 associative、commutative/idempotent 合并语义，冲突 fatal。
- [x] barrier 按精确 manifest 校验全部 identity 与 provenance，并在一个 state update 中 promote 整个 mixed batch、清空 staging。
- [x] 反向完成顺序与正向完成顺序得到完全相同的 accepted state。
- [x] fatal、cancel、checkpoint failure 或 manifest 缺项均不能产生 half-accepted batch。
- [x] 并发生成相同 Evidence ID/body 时收敛为一个幂等值；同 ID 不同内容 fatal；orphan Evidence 不具备 eligibility。
- [x] usage 以稳定方式合并；aggregate counts 只用于 telemetry 与阻止未来工作，不是业务预算，也不回滚已完成 work。
- [x] Specialist 内部调用不计作顶层 Task。
- [x] 并发范围只覆盖单个 Run 内的 Tasks；不实现同一 Conversation 上多个 Request 的 admission、锁或并发接受。
- [x] 使用真实 LangGraph runtime 与 pending-write 集成测试覆盖 mixed batch、逆序完成、fatal/cancel 与 checkpoint 边界，不能只用 reducer 单元测试代替。

## Comments

- Test seam / scope review（gpt-5.6-sol medium）：确认只扩展首个 single-Run batch 至 8；不提前实现 Ticket 10 的 rolling Dispatch，也不触及跨 Request admission 或锁。测试 seam 为 Dispatch/accept、同一 barrier、Evidence catalog 和真实 LangGraph/PostgreSQL pending-write。
- Code review（gpt-5.6-sol high）：三项初审问题均已修复：Evidence eligibility 仅由 checkpointed `accepted_batches` 投影；恢复 manifest 限制 1–8；同 ID/body 的 sibling Evidence 采用稳定存档。最终复核 P0/P1/P2 均为 0；恢复 manifest 于 Send 前复验 stable batch/task identity、canonical objective 和 scope/registry eligibility。
- Standards review（gpt-5.6-sol high）：初审的并发集成测试重复及伪公共 reducer 已修复；最终 actionable finding 为 0。
- Verification：`uv run ruff check`、scoped `uv run pyright` 与 `git diff --check` 均通过；焦点 PostgreSQL 测试 92 passed；`scripts/run-pytest tests` 为 395 passed、1 skipped（185 个既有 warning）。
- Unresolved review comments: 0.
