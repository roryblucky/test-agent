# 05: GCS 生产加载与 Local 行为一致

**What to build:** 生产环境通过配置的 GCS bucket 和 prefix 加载 Tenant Specialists 与 Skills，并完成相同的 Coordinator delegation、Skill activation、Tool 调用和实时 reference 读取；开发环境继续使用 Local Adapter。

**Blocked by:** 04 — Activated Skill 按需读取实时 References。

**Status:** ready-for-agent

依据：已批准的 Tenant-authored Specialist Definitions 规格及 ADR 0008。

- [ ] 环境配置选择 Local 或 GCS Adapter，Markdown 不包含存储选择；两者使用规格约定的相同 Tenant 相对结构，GCS 支持配置 bucket 和 prefix。
- [ ] GCS Adapter 复用共同 Loader interface、parsing、validation 和 catalog 模型，不复制出新的 Markdown 方言或 storage-specific runtime 分支。
- [ ] 应用接收流量前完成已知 Tenant 的完整 Agent/Skill definition 加载；references 不预读；遵守逐条 skip/log 和 loaded/skipped counts 规则。
- [ ] prefix 定位严格隔离 Tenant；存储失败不会借用其他 Tenant 内容或扩大 catalog membership；日志不泄漏完整 instructions 或 references。
- [ ] Local 和 GCS 通过同一 Adapter 合同，覆盖等价 source documents、校验结果、启动快照、实时 references、确定性顺序、空目录、列举或读取失败时无部分成功结果。
- [ ] 通过真实 startup composition 和现有 HTTP/SSE Agent 路径验证 GCS 定义驱动的 Specialist selection、Skill 交错激活、scope-limited Tool execution、结果发布及 pins。模型和 GCS 使用确定性替身，测试不要求真实网络或生产 credentials。
- [ ] 生产 composition 以 Markdown-defined Specialists 为目标，迁移用 code Adapter 继续可用；不重构 legacy FlowEngine。
- [ ] 提供最小 authoring 与环境配置说明，明确 definitions 重启生效、references 每次实时读取、required-tools/allowed-tools 语义和无 version Skill 的使用。
- [ ] 聚焦 Adapter/集成测试和受影响回归测试通过，并完成规格要求的最终确定性测试检查；无需等待 06 才能独立验证本票行为。

## Comments

- 拆分及验收范围经独立 GPT-6 Astra agent 复审为 PASS；05 不依赖 06，GCS 全路径与恢复执行可分别验收。
