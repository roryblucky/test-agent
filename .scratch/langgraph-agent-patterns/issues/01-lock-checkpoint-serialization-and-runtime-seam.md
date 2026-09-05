# 01: 锁定共享 checkpoint 序列化与 typed runtime seam

**What to build:** 在不改变现有 Linear 模式行为的前提下，把共享 checkpoint 边界收紧为可验证的安全序列化边界，并为 Agent 模式提供最小的 typed runtime 扩展 seam。

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] 共享 saver 显式配置 `JsonPlusSerializer`，关闭 pickle fallback，并将 JSON 与 MessagePack 的额外允许模块都设为空；不得依赖默认值或环境变量。
- [ ] 现有应用持久化值保持 JSON-native，且批准的框架 Message 类型能够完整 round-trip。
- [ ] 每个 runtime 的 typed checkpoint adapter 在节点输入和 Conversation 投影前校验其拥有的全部 channel。
- [ ] pickle、未知 constructor、退化为普通字典的框架对象以及携带 constructor arguments 的 payload 均被拒绝。
- [ ] 现有 Linear HTTP 行为与 PostgreSQL checkpoint 集成测试保持通过。
- [ ] 只建立狭窄的 runtime 扩展 seam，不预先猜测或固化未来 Agent state 类型。

