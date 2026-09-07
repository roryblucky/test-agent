# 16: 强化不可信内容边界

**What to build:** 用对抗性数据证明 Evidence、Tool output、Specialist Result 与 Skill reference 始终只是 typed data，不能把内容中的指令提升为权限、路由或发布事实。

**Blocked by:** 06: 渐进激活 Specialist-owned Skills; 12: 在冻结的 Prepared Synthesis 上 repair 并 fail closed

**Status:** done

- [x] 对 Evidence body、model-visible Tool output、Specialist Result 与 Skill references 分别加入 instruction-like、role-like 与 marker-like adversarial fixtures。
- [x] 不可信内容不能扩大 Research Scope、修改 Tool binding、添加 Specialist/Skill、放宽 source/filter constraints 或改变 Coordinator/Graph routing。
- [x] 不可信内容不能读取或泄露 hidden state、app-only metadata、raw provider payload、secrets、retry diagnostics 或 Conversation history。
- [x] 不可信内容不能伪造 eligible Evidence marker、Calculation marker、citation、Artifact alias、Task identity、Run identity 或 provenance。
- [x] direct structural attempts 已按 Ticket 03、05 与 06 的 pre-data-access validation matrix fail closed；本 ticket 补足内容注入而不另建权限系统。
- [x] Synthesis marker parsing、eligibility binding 与 code-owned calculation rendering 在这些 adversarial fixtures 下仍确定性 fail closed，禁止删除非法 marker 后发布剩余文本。
- [x] 不引入或声称使用生产版 PydanticAI Harness defender；本 POC 依赖 typed trust boundaries 与 adversarial tests。

## Comments

- Sol-medium seam review 确认真实 HTTP → Agent Graph → FunctionModel → PostgreSQL 为主 seam；以 3 类 payload × Evidence body / Tool excerpt / Specialist Result / Skill reference 四面覆盖。Result summary 仍可作为 bounded `PriorResultView` 影响 Coordinator 的既有决策；“不得改变 routing”指不得改变图拓扑、eligible Specialist/Tool/Skill 或 code-owned scope binding。
- Red test 发现 `ToolReturn.metadata` 会进入 PydanticAI FunctionModel message history，不能视为 app-only。Evidence 与 Calculation 现只经现有 invocation-local capture 保存 metadata，`ToolReturn` 仅返回 model-visible safe value；单测锁定该界。
- 对抗集成测试先在同一 Conversation 创建真实 prior exchange 与三次受控 retry diagnostics，再跑当前 Request。Query Understanding 仅按既有设计读取完整 history；下游 Specialist FunctionModel、Prepared Synthesis、当前 SSE/assistant Message 与当前 checkpoint 均不泄漏 history、diagnostics、Evidence body、raw provider secret 或 hidden-state sentinel。
- marker fixture 同时含 eligible/unknown Evidence 与 Calculation markers、fake citation、Artifact alias、Task/Run identity、provenance；断言 citation、Prepared aliases、Task/Run identity、Calculation provenance、provider source/query 与 allowed Tool set 仍由 code-owned typed bindings 决定。非法 marker 经一次 frozen Prepared repair 后仍 fail closed，绝不删 marker 后发布。
- Ticket 03、05、06 的 direct structural pre-data-access validation matrix 已存在，故本票不重复实现权限系统；未加入或声称生产 PydanticAI Harness defender。
- Sol-high Standards review：初见重复 checkpoint helper，已提升为 financial fixture 共享 helper；终审 0 actionable findings。Sol-high Spec review：补齐 prior history/retry/hidden sentinel 与身份伪造 assertions；终审 0 actionable findings。
- 验证：`uv run ruff check app tests alembic/versions/0018_drop_conversation_registry.py`、`uv run pyright app tests`、`scripts/run-pytest tests -q`、`git diff --check` 通过（449 passed, 1 skipped）。
- Unresolved review comments: 0.
