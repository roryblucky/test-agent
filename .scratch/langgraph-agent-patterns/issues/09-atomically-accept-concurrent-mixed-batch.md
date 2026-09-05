# 09: 原子接受单 Run 内并发 mixed Dispatch Batch

**What to build:** 让一个 Coordinator 决策在同一个 Run 内并行派发至多 8 个独立 Specialist Task，并只在整个 mixed success/failure batch 完整时原子接受结果。

**Blocked by:** 08: 按封闭 V1 映射重试 Specialist

**Status:** ready-for-agent

- [ ] 每个 Dispatch Batch 最多 8 个 Task，测试覆盖恰好 8 与 9；执行 `max_concurrency` 为 8，所有校验在创建 Send 前完成。
- [ ] 同批 Task 相互独立，不允许引用同批中其他 Task 的结果。
- [ ] 每个完成项只写入带稳定 identity 与 attempt provenance 的 immutable staged contribution；staging reducer 具备 associative、commutative/idempotent 合并语义，冲突 fatal。
- [ ] barrier 按精确 manifest 校验全部 identity 与 provenance，并在一个 state update 中 promote 整个 mixed batch、清空 staging。
- [ ] 反向完成顺序与正向完成顺序得到完全相同的 accepted state。
- [ ] fatal、cancel、checkpoint failure 或 manifest 缺项均不能产生 half-accepted batch。
- [ ] 并发生成相同 Evidence ID/body 时收敛为一个幂等值；同 ID 不同内容 fatal；orphan Evidence 不具备 eligibility。
- [ ] usage 以稳定方式合并；aggregate counts 只用于 telemetry 与阻止未来工作，不是业务预算，也不回滚已完成 work。
- [ ] Specialist 内部调用不计作顶层 Task。
- [ ] 并发范围只覆盖单个 Run 内的 Tasks；不实现同一 Conversation 上多个 Request 的 admission、锁或并发接受。
- [ ] 使用真实 LangGraph runtime 与 pending-write 集成测试覆盖 mixed batch、逆序完成、fatal/cancel 与 checkpoint 边界，不能只用 reducer 单元测试代替。

