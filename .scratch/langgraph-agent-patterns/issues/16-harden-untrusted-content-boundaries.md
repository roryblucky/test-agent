# 16: 强化不可信内容边界

**What to build:** 用对抗性数据证明 Evidence、Tool output、Specialist Result 与 Skill reference 始终只是 typed data，不能把内容中的指令提升为权限、路由或发布事实。

**Blocked by:** 06: 渐进激活 Specialist-owned Skills; 12: 在冻结的 Prepared Synthesis 上 repair 并 fail closed

**Status:** ready-for-agent

- [ ] 对 Evidence body、model-visible Tool output、Specialist Result 与 Skill references 分别加入 instruction-like、role-like 与 marker-like adversarial fixtures。
- [ ] 不可信内容不能扩大 Research Scope、修改 Tool binding、添加 Specialist/Skill、放宽 source/filter constraints 或改变 Coordinator/Graph routing。
- [ ] 不可信内容不能读取或泄露 hidden state、app-only metadata、raw provider payload、secrets、retry diagnostics 或 Conversation history。
- [ ] 不可信内容不能伪造 eligible Evidence marker、Calculation marker、citation、Artifact alias、Task identity、Run identity 或 provenance。
- [ ] direct structural attempts 已按 Ticket 03、05 与 06 的 pre-data-access validation matrix fail closed；本 ticket 补足内容注入而不另建权限系统。
- [ ] Synthesis marker parsing、eligibility binding 与 code-owned calculation rendering 在这些 adversarial fixtures 下仍确定性 fail closed，禁止删除非法 marker 后发布剩余文本。
- [ ] 不引入或声称使用生产版 PydanticAI Harness defender；本 POC 依赖 typed trust boundaries 与 adversarial tests。

