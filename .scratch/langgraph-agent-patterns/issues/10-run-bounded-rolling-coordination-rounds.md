# 10: 执行有界 rolling Coordination Rounds

**What to build:** 让 Coordinator 在已接受结果上做多轮、可审计、严格有界的 rolling 决策，既能改变下一轮执行形状，也能在上限或 TOCTOU 异常时 fail closed。

**Blocked by:** 09: 原子接受单 Run 内并发 mixed Dispatch Batch

**Status:** done

- [x] Coordinator 每轮只接收完整稳定的 prior-result projection 与 `DataGapView`，不接收 Evidence body、raw payload 或 Conversation messages。
- [x] 每次决定只能是 `DispatchBatch` 或 `Finish`，且所有结构、权限与上限校验在创建 Send 前完成。
- [x] 跨轮 context 只能引用更早轮次中已接受的 successful result IDs，并按稳定顺序 materialize。
- [x] 每个 round 最多一次 same-round repair，即最多 2 次 actor invocations；被拒绝的 candidate 不计为 round，repair 的输入保持冻结。
- [x] 覆盖边界与多 1：最多 5 次 Coordinator decisions、4 个 dispatch rounds、32 个 Tasks、每个 Task 8 个 context result IDs。
- [x] 完成第 4 个 dispatch round 后只允许 `Finish`；第 5 个 dispatch candidate 先允许一次 repair，仍非法则以 incomplete 结束且不创建 Send。
- [x] task limit 与 coordination limit 分别产生确定性的 incomplete reason。
- [x] 使用前一轮不同 outcome 的测试证明下一轮决策与执行形状可以改变，同时仍受上限约束。
- [x] 通过真实 graph step 证明 recursion limit 40 足够覆盖最大合法路径；非预期循环 fatal。
- [x] 每个 accepted decision 生成 immutable `CoordinationRound` 并单调增加 revision；被拒绝的 repair candidate 不增加 revision、不生成 round，checkpoint round-trip 后保持一致。
- [x] Task objective 先做确定性的单行规范化，明确处理 CRLF、换行、控制字符与 NFC；保存 canonical value，后续 disclosure 复用同一值。

## Comments

- Test seam / scope review（gpt-5.6-sol medium）：范围限于 Ticket 10 的 rolling coordinator、accepted-round checkpoint、cross-round context 与 bounded repair；不提前实现跨 Request admission/locking 或后续 ticket。测试 seam 为 pure acceptance/materialization、PydanticAI actor、真实 Agent graph/PostgreSQL checkpoint 与 request stream。
- Code review（gpt-5.6-sol high）：初审发现并已修复：Specialist 未接收 materialized context；失败 Task 未投影给下一轮；checkpoint 轮次序列未完整校验；schema-invalid repair 仅注入异常；递归溢出仅测无关图；同 Conversation 第二请求以新 request ID 校验旧 rounds。最终实现以 immutable rounds、`coordination_request_id`、真实 `FunctionModel` schema repair 与 production-stream recursion test 覆盖；终审无 actionable finding。
- Standards review（gpt-5.6-sol high）：复审同 Conversation checkpoint ownership 后，无遗留 actionable finding。
- Verification：`uv run ruff check`、scoped `uv run pyright`、`git diff --check` 通过；焦点 PostgreSQL 测试 75 passed；`scripts/run-pytest tests -q`：423 passed、1 skipped（200 warnings）。
- Unresolved review comments: 0.
