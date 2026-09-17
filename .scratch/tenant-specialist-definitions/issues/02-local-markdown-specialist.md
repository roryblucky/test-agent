# 02: 本地 Markdown Specialist 从启动加载到完成 Task

**What to build:** Tenant 管理员在本地定义一个无 Skills 的 Markdown Specialist，应用启动后即可由 Coordinator 选择并执行，通过现有 HTTP/SSE 返回结果，并记录实际使用的 Specialist Definition Pin。

**Blocked by:** 01 — 统一 code-defined Specialist Catalog 与 Tool 权限。

**Status:** claimed

依据：已批准的 Tenant-authored Specialist Definitions 规格及 ADR 0008。

- [ ] 在可信 Tenant 配置、Model Registry 就绪后、接收流量前，通过 Local Loader 加载已知 Tenants 的 Agent definitions。采用规格约定的 Tenant 相对目录结构，Tenant 身份来自拥有该文档的可信路径与 request context。
- [ ] frontmatter 仅支持 id、description、model-profile 和 skills；body 是 instructions。校验必需内容、批准的 model profile 和 30,000 字符 instruction 上限，不引入 role、Tool allowlist、Intent allowlist 或自定义 output schema。
- [ ] 本切片完成空 skills 列表的完整执行。非空列表在尚无对应有效 Skill Catalog 条目时按无效定义处理，不静默忽略；03 扩展到实际 Skill packages。
- [ ] 逐项跳过无效 definitions，有效 siblings 继续加载。日志包含 Tenant、source identity、definition kind、原因及 loaded/skipped 数量，不记录完整内容或秘密信息。
- [ ] Local 与后续 GCS 使用共同 Loader interface 和 catalog 模型；parsing、validation、hashing、indexing 留在 catalog 实现中，Agent Graph 不读取文件或构造路径。
- [ ] Markdown Adapter 使用现有 PydanticAI Specialist Actor；平台 instructions 和固定 guards 在前，Tenant instructions 附加在后；实际 Tool bindings 和 graph routing 仍由代码约束。
- [ ] code-defined 与 Markdown-defined Adapters 通过同一套输入、上下文、Tool scope、attempt acceptance 和结果合同测试；保留 code Adapter。
- [ ] canonical metadata 与 instruction body 的 SHA-256 构成稳定 Definition Pin，排除 storage location、references 和可变运行状态。Pin 由应用负责携带、接受和持久化，不由模型生成。
- [ ] 已接受 execution 的 telemetry 与 PostgreSQL Task/checkpoint 记录实际 pin；状态字段演进兼容现有 checkpoint，不持久化历史 Markdown body，不向模型或公共 SSE 泄漏内部 instructions。
- [ ] 进程内修改 Agent 文件不会改变已加载定义；第二次独立 startup 读取同 ID 的新内容并生成新 pin。跨 checkpoint 恢复的完整行为由 06 负责。
- [ ] 通过真实启动 composition 与现有 HTTP/SSE Agent 路径验证加载、Coordinator selection、dispatch、结果发布及 pin 持久化，模型使用确定性替身。
- [ ] 聚焦确定性测试及受影响回归测试通过；不改变 legacy FlowEngine 请求路径。

## Comments

- 拆分及验收范围经独立 GPT-6 Astra agent 复审为 PASS；本记录针对规划，不代表实现已通过 review。
