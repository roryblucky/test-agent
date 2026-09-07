# 15: 证明连续 Agent Runs 可安全改变执行形状

**What to build:** 让同一 Agent Conversation 的两个顺序 Request 只通过完整历史消息对延续语义，同时彻底重置 Run-local 状态，并允许第二次 Run 采用不同的执行形状。

**Blocked by:** 14: 证明完整金融 golden path

**Status:** done

- [x] 同一 Conversation 的两个顺序 Request 使用不同 Request IDs；第一条可走 fan-out/fan-in，第二条可走 combined multi-hop，并分别独立完成 publication。
- [x] follow-up fixture 让 Query Understanding 从有界完整 Conversation pairs 中把代词或隐含 benchmark 解析成 self-contained standalone query。
- [x] Coordinator 与 Synthesis 只收到 standalone query 和 selected Business Intent；Specialists 只收到 Task-specific context，所有下游 actor 都不接收 Conversation history。
- [x] 第二个 initializer 后，每个 reducer-backed Run-local channel 实际为空，每个 scalar 用正确语义重置；任何 actor 都不能在 initializer 完成前运行。
- [x] 前一 Run 的 accepted/staging state、manifest、answer、errors、diagnostics、counters、rounds、attempts、usage、Evidence eligibility 与 calculation aliases 均不得泄漏到后一 Run。
- [x] 同一 Request ID 的幂等重试保持 canonical result，不创建新 Run；同 ID 不同输入冲突 fail closed，不被解释为 resume。
- [x] 两次 Run 的 checkpoint 与 SSE 都证明 publication 只读取各自已提交 canonical state。

## Comments

- Sol-medium seam review 确认使用真实 HTTP → Agent Graph → PostgreSQL seam；同一 Request ID 按 ADR-0003 可重执行，但必须收敛到同一 canonical result，不能把不同输入解释为 resume。
- 实现复用 Ticket14 的 FunctionModel-backed financial fixture。首轮 fan-out 生成 calculation aliases，并让 holdings specialist 受控重试三次后失败；第二轮从完整 history pair 解析代词，独立以 holdings → report 的 multi-hop 完成。`dispatched_task` 已加入 initializer 与 checkpoint reset 断言。
- Sol-high Spec review：先后发现并闭环实际 multi-hop、首轮 failure diagnostics/attempts/usage 覆盖、第二轮的 semantic follow-up 与泄漏断言；最终 0 actionable findings。
- Sol-high Standards review：先后发现并闭环 fixture 的 provenance 直接 mutation 与未使用 failure config；`FinancialFixture.configure_run()` 现拥有 provenance 更新，Run-1-only sentinel 已被实际断言；最终 0 actionable findings。
- 验证：`uv run ruff check app tests alembic/versions/0018_drop_conversation_registry.py`、`uv run pyright app tests`、`scripts/run-pytest tests -q`、`git diff --check` 通过（445 passed, 1 skipped）。相关 Ticket14/15 集成测试 4 passed。
- Unresolved review comments: 0.
