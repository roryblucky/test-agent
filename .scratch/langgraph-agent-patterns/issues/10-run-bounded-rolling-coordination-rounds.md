# 10: 执行有界 rolling Coordination Rounds

**What to build:** 让 Coordinator 在已接受结果上做多轮、可审计、严格有界的 rolling 决策，既能改变下一轮执行形状，也能在上限或 TOCTOU 异常时 fail closed。

**Blocked by:** 09: 原子接受单 Run 内并发 mixed Dispatch Batch

**Status:** ready-for-agent

- [ ] Coordinator 每轮只接收完整稳定的 prior-result projection 与 `DataGapView`，不接收 Evidence body、raw payload 或 Conversation messages。
- [ ] 每次决定只能是 `DispatchBatch` 或 `Finish`，且所有结构、权限与上限校验在创建 Send 前完成。
- [ ] 跨轮 context 只能引用更早轮次中已接受的 successful result IDs，并按稳定顺序 materialize。
- [ ] 每个 round 最多一次 same-round repair，即最多 2 次 actor invocations；被拒绝的 candidate 不计为 round，repair 的输入保持冻结。
- [ ] 覆盖边界与多 1：最多 5 次 Coordinator decisions、4 个 dispatch rounds、32 个 Tasks、每个 Task 8 个 context result IDs、objective 512 UTF-8 bytes、Specialist context 64 KiB、Coordinator 单个 prior-result projection 16 KiB、aggregate projection 128 KiB。
- [ ] 完成第 4 个 dispatch round 后只允许 `Finish`；第 5 个 dispatch candidate 先允许一次 repair，仍非法则以 incomplete 结束且不创建 Send。
- [ ] task limit、coordination limit 与 coordinator-context limit 分别产生确定性的 incomplete reason，禁止静默 truncate。
- [ ] 使用前一轮不同 outcome 的测试证明下一轮决策与执行形状可以改变，同时仍受上限约束。
- [ ] 通过真实 graph step 证明 recursion limit 40 足够覆盖最大合法路径；非预期循环 fatal。
- [ ] 决策校验后、materialization 或 serialization 时出现的 size mismatch 视为 TOCTOU fatal，不触发 repair、retry、`TaskFailed` 或 Specialist 调用。
- [ ] 每个 accepted decision 生成 immutable `CoordinationRound` 并单调增加 revision；被拒绝的 repair candidate 不增加 revision、不生成 round，checkpoint round-trip 后保持一致。
- [ ] Task objective 先做确定性的单行规范化，明确处理 CRLF、换行、控制字符与 NFC，再按 UTF-8 bytes 计数；覆盖规范化后 512 与 513，保存 canonical value，后续 disclosure 复用同一值。

