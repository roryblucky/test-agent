# 01: 锁定共享 checkpoint 序列化与 typed runtime seam

**What to build:** 在不改变现有 Linear 模式行为的前提下，把共享 checkpoint 边界收紧为可验证的安全序列化边界，并为 Agent 模式提供最小的 typed runtime 扩展 seam。

**Blocked by:** None (can start immediately)

**Status:** done

- [x] 共享 saver 显式配置 `JsonPlusSerializer`，关闭 pickle fallback，并将 JSON 与 MessagePack 的额外允许模块都设为空；不得依赖默认值或环境变量。
- [x] 现有应用持久化值保持 JSON-native，且批准的框架 Message 类型能够完整 round-trip。
- [x] 每个 runtime 的 typed checkpoint adapter 在节点输入和 Conversation 投影前校验其拥有的全部 channel。
- [x] pickle、未知 constructor、退化为普通字典的框架对象以及携带 constructor arguments 的 payload 均被拒绝。
- [x] 现有 Linear HTTP 行为与 PostgreSQL checkpoint 集成测试保持通过。
- [x] 只建立狭窄的 runtime 扩展 seam，不预先猜测或固化未来 Agent state 类型。

## Comments

- 2026-09-05：实现 strict shared serializer、Linear typed checkpoint adapter 与狭窄 runtime seam；持久化 citations、groundedness、final response 均改为 JSON-native。
- 2026-09-05：Sol high Standards review 发现非有限 JSON number 与 adapter 名称问题；均已修复并覆测。
- 2026-09-05：Sol high Spec review 发现 channel shape 过宽、拒绝 LangGraph framework channels、以及真实 downgrade 测试缺失；均已修复。未解决 review comments：0。
- 验证：`ruff check app tests`、`pyright --pythonpath .venv/bin/python`、64 个 LangGraph v2 unit tests 通过。完整 `pytest tests` 为 191 passed、1 skipped、61 errors；所有 errors 均在 PostgreSQL integration fixture setup，因未设置 `LANGGRAPH_V2_TEST_DATABASE_URL`。故 Linear HTTP/PostgreSQL acceptance criterion 待可丢弃测试库后验证。
- 2026-09-05：已在专用空 PostgreSQL test database 运行 v2 UAT：97 passed。覆盖 migrations、Linear HTTP/SSE、官方 checkpoint、groundedness、post-moderation 与 Uvicorn disconnect；本票完成。
