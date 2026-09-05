# 15: 证明连续 Agent Runs 可安全改变执行形状

**What to build:** 让同一 Agent Conversation 的两个顺序 Request 只通过完整历史消息对延续语义，同时彻底重置 Run-local 状态，并允许第二次 Run 采用不同的执行形状。

**Blocked by:** 14: 证明完整金融 golden path

**Status:** ready-for-agent

- [ ] 同一 Conversation 的两个顺序 Request 使用不同 Request IDs；第一条可走 fan-out/fan-in，第二条可走 combined multi-hop，并分别独立完成 publication。
- [ ] follow-up fixture 让 Query Understanding 从有界完整 Conversation pairs 中把代词或隐含 benchmark 解析成 self-contained standalone query。
- [ ] Coordinator 与 Synthesis 只收到 standalone query 和 selected Business Intent；Specialists 只收到 Task-specific context，所有下游 actor 都不接收 Conversation history。
- [ ] 第二个 initializer 后，每个 reducer-backed Run-local channel 实际为空，每个 scalar 用正确语义重置；任何 actor 都不能在 initializer 完成前运行。
- [ ] 前一 Run 的 accepted/staging state、manifest、answer、errors、diagnostics、counters、rounds、attempts、usage、Evidence eligibility 与 calculation aliases 均不得泄漏到后一 Run。
- [ ] 同一 Request ID 的幂等重试保持 canonical result，不创建新 Run；同 ID 不同输入冲突 fail closed，不被解释为 resume。
- [ ] 两次 Run 的 checkpoint 与 SSE 都证明 publication 只读取各自已提交 canonical state。

