# Tenant 自定义 Specialist Definition

Status: ready-for-agent

本文是同目录英文规格的简体中文版本，两者描述同一项功能和相同的实现边界。

## 问题陈述（Problem Statement）

目前，Tenant 管理员无法在不修改 Python 注册代码的情况下新增或调整
Specialist Agent。即使 Specialist 的身份、描述、instructions、model profile 和
Skills 都属于声明式内容，这些 Tenant 自有的内容决策仍然需要应用代码变更和部署。

现有 runtime 还存在两套不同的 Skill 实现。code-defined Specialist 的注册模型中，
Specialist、Skill 和 Tool allowlist 与 Tenant policy、Research Scope 已经拥有的权限
边界相互重叠。这使得“究竟哪一层能够授予执行权限”难以解释，也不利于直接比较
Markdown-defined Specialist 和现有 code-defined 实现的行为。

本功能需要允许 Tenant 管理员通过 Markdown 编写 Specialist Definition 及其 Skills，
同时保留现有 Agent Graph 的执行和结果合同。系统必须通过 `kmsAppId` 隔离 Tenant，
保留平台安全 instructions 的最高优先级，把 Tool 权限留在可信的平台 policy 中，并且
不修改当前由模型生成的 Intent Result 或 Intent Catalog 结构。

## 解决方案（Solution）

应用启动时，从 Tenant 范围内的 definition tree 加载每个已知 Tenant 的 Specialist
Definitions 和 Skills。开发环境读取本地目录；生产环境从配置的 GCS prefix 下读取
相同的相对目录结构。两个存储 Adapter 产出相同的内存 Specialist Catalog 和 Skill
Catalog。

`.agent.md` 文档提供 Specialist 的身份、描述、经过批准的 model profile、Skill 名称
以及 instruction body。Tenant ID 由文档所在路径提供。标准 `SKILL.md` 文档提供 Skill
的 discovery metadata、instructions 和 Tool dependencies。完整的 Agent 和 Skill
definition 在启动时完成校验并缓存于当前进程；Skill reference 文件保持实时，只在
Activated Skill 调用 `load_reference` 时读取。

Coordinator Agent 可以看到当前 Tenant Specialist Catalog 中的全部有效 Specialist
Descriptor，并根据 ID 和 description 做选择。Business Intent 继续选择可信的 scope
约束，但它不选择或授权 Specialist，也不增加 `allowed_specialist_ids` 字段。执行时，
runtime 将 Markdown-defined 和 code-defined Specialist 适配到现有 Specialist actor
及 Agent Graph 合同。两种实现接收相同的 Specialist Task Input，产生相同的
Specialist Attempt，并经过同一个确定性的 acceptance 环节，得到相同的 Specialist
Result 和 Task Outcome。

Specialist 可以渐进式激活零个或多个 Eligible Skills，并可以把 Skill 激活与业务 Tool
调用任意交错。Skill 只指导如何使用已经获得授权的 Tool，不授予 Tool 权限。有效 Tool
始终是平台 Tool Registry、Tenant Tool policy 和当前 Research Scope 的交集。平台
instructions 和固定安全 guards 的优先级始终高于 Tenant 编写的 Specialist 和 Skill
instructions。

## 用户故事（User Stories）

1. 作为 Tenant 管理员，我希望使用 Markdown 定义 Specialist Agent，从而无需修改 Python 注册代码就能添加领域能力。
2. 作为 Tenant 管理员，我希望 Specialist Definition 包含 ID、description、经过批准的 model profile、Skill 名称和 instructions，从而能从一个文档理解其 runtime 行为。
3. 作为 Tenant 管理员，我希望使用 Specialist Definition 的 Markdown body 编写 instructions，从而可以采用熟悉的内容格式。
4. 作为 Tenant 管理员，我希望多个 Specialist Definition 可以复用同一个 Skill，从而只需维护一份共享指导内容。
5. 作为 Tenant 管理员，我希望一个 Specialist Agent 可以声明多个 Skills，从而能针对 Task 渐进式组合相关指导。
6. 作为 Tenant 管理员，我希望 Specialist Agent 可以不声明任何 Skill，从而避免简单职责必须创建不必要的 Skill package。
7. 作为 Tenant 管理员，我希望 Skill 冲突被视为发布内容质量问题，从而通过 review 和 eval 解决，而不是依赖隐藏的 runtime 优先级。
8. 作为 Tenant 管理员，我希望格式错误的 definition 被跳过并记录有效日志，从而一个内容错误不会阻止同一 Tenant 的其他有效 Specialist 加载。
9. 作为 Tenant 管理员，我希望 Agent 和 Skill 的变更在重启后生效，从而单个进程使用稳定的 definition snapshot。
10. 作为 Tenant 管理员，我希望 reference 文件的修改在下一次 `load_reference` 调用时可见，从而无需重启 runtime 就能修正辅助材料。
11. 作为 Tenant 管理员，我希望 Coordinator Agent 能看到 Specialist description，从而可以根据声明的职责进行 delegation。
12. 作为 Tenant 管理员，我希望平台拒绝未知的 model profile，从而 Markdown 无法选择未经批准的模型部署。
13. 作为 Tenant 管理员，我希望 Specialist instructions 有明确的长度上限，从而单个 Specialist Definition 不会无边界占用 prompt budget。
14. 作为 Tenant 管理员，我希望每次 Specialist invocation 最多暴露二十个 Eligible Skill summaries，从而让 discovery context 保持有界。
15. 作为 Tenant 管理员，我希望 Skill 声明的每个 Tool 都在启动时根据平台 Tool Registry 校验，从而尽早发现损坏的 Skill package。
16. 作为 Tenant 管理员，我希望 required Tool 位于当前 Research Scope 之外的 Skill 不出现在 discovery 中，从而模型不会被引导去激活无法使用的指导内容。
17. 作为 Tenant 管理员，我希望启动时不读取 Skill reference 内容，从而大体积辅助文档不会增加启动内存和 prompt context。
18. 作为 Coordinator Agent，我希望获得当前进程 Tenant Catalog 的全部有效 Specialist Descriptors，包括从 checkpoint 恢复之后，从而可以不受所选 Business Intent 限制地选择最合适的 Specialist。
19. 作为 Coordinator Agent，我希望选择 Specialist 时只看到 ID 和 description，从而 Specialist instructions 和 Skill 细节不会污染 coordination context。
20. 作为 Coordinator Agent，我希望确定性校验拒绝无效或跨 Tenant 的 Specialist ID，从而模型输出无法逃逸当前 Tenant 的 Specialist Catalog。
21. 作为 Specialist Agent，我希望无论采用哪种定义方式都接收相同的 Specialist Task Input，从而 Markdown authoring 不会引入第二套执行协议。
22. 作为 Specialist Agent，我希望继续产生现有 Specialist Attempt 和 Specialist Result 结构，从而 Agent Graph、Synthesis 和客户端无需增加 Markdown 专用处理。
23. 作为 Specialist Agent，我希望初始阶段只看到 Eligible Skills 的简要 summaries，从而完整 instructions 仅在确实需要时披露。
24. 作为 Specialist Agent，我希望可以激活零个或多个 Eligible Skills，从而根据 Task 选择合适数量的指导内容。
25. 作为 Specialist Agent，我希望在调用业务 Tool 后仍可激活另一个 Eligible Skill，从而 discovery 和 execution 可以自然交错。
26. 作为 Specialist Agent，我希望重复激活同一个 Skill 时保持幂等，从而重试不会重复添加有效 instructions 或 pins。
27. 作为 Specialist Agent，我希望每个 Activated Skill 在本次 invocation 剩余时间内持续有效，从而能一致地遵循组合后的指导。
28. 作为 Specialist Agent，我希望 `load_reference` 返回全部 reference 文件或明确的失败，从而不会把部分内容误认为完整辅助上下文。
29. 作为 Specialist Agent，我希望每次 `load_reference` 都重新读取当前存储，从而不会收到进程缓存中的过期 reference 内容。
30. 作为 Specialist Agent，我希望系统拒绝激活不满足条件或未知的 Skill，从而无法通过猜测名称发现任意 Tenant 内容。
31. 作为 Specialist Agent，我希望 invocation 中只绑定有效的业务 Tools，从而 prompt instructions 无法扩大可执行权限。
32. 作为 Specialist Agent，我希望预期内的 Tool unavailable 继续走现有 typed outcome 路径，从而 Markdown authoring 不改变 Data Gap 行为。
33. 作为平台安全负责人，我希望 `kmsAppId` 同时标识 Tenant 及其 definition-tree prefix，从而一个 Tenant 的 Agent 无法加载另一个 Tenant 的 definitions。
34. 作为平台安全负责人，我希望 Tenant 身份来自可信 request 和 storage context，而不是 Markdown frontmatter，从而内容无法自行指定 Tenant。
35. 作为平台安全负责人，我希望 Specialist 和 Skill instructions 被附加在固定平台 instructions 之后，从而 Tenant 内容无法替换安全 guards。
36. 作为平台安全负责人，我希望 Agent 和 Skill 内容无法扩大 Research Scope、Tool bindings、Tenant authority 或 Agent Graph routing，从而 instructions 始终只是指导而不是 policy。
37. 作为平台安全负责人，我希望所有可执行能力只能来自 Tool Registry，从而 Skill package 无法执行其附带的 scripts。
38. 作为平台安全负责人，我希望不设置 Specialist 专属 Tool allowlist，从而 Tool 权限只由 Tenant Tool policy 和 Research Scope 两个明确来源决定。
39. 作为平台安全负责人，我希望 Business Intent 不包含 `allowed_specialist_ids`，从而 Business Intent 只分类业务目的，而不充当 Agent 权限列表。
40. 作为平台运维人员，我希望 Local 和 GCS storage Adapters 使用同一个 definition-loading interface，从而环境选择不会改变 runtime 行为。
41. 作为平台运维人员，我希望应用接收流量前完成所有 Agent 和 Skill definition 的加载，从而进程不会使用未完成初始化的 catalog 提供服务。
42. 作为平台运维人员，我希望无效条目的日志包含 Tenant、source、definition kind 和 validation reason，从而内容负责人可以修复问题，同时不暴露文件内容或秘密信息。
43. 作为平台运维人员，我希望看到每个 Tenant 加载和跳过的 Specialist、Skill 数量，从而可以观测启动健康状态。
44. 作为平台运维人员，我希望 storage 或 validation failure 对受影响的 definition 执行 fail closed，从而 runtime 不会借用其他 Tenant 的 definition，也不会静默扩大行为权限。
45. 作为审计人员，我希望已接受的 Specialist execution 记录其实际使用的 Specialist Definition Pin，从而可以把 Task Outcome 与当时观察到的配置关联起来。
46. 作为审计人员，我希望 Activated Skill Pins 按第一次激活顺序记录，并包含可选的管理员声明版本及必填 content hash，从而无版本 Skill 仍能准确标识指导已接受 attempt 的缓存 definition。
47. 作为审计人员，我希望 definition pins 不包含实时 reference 内容，从而 pin 不会被错误理解为精确历史重放的保证。
48. 作为应用开发者，我希望 code-defined 和 Markdown-defined Specialist Adapters 通过同一套合同测试，从而可以直接比较迁移行为。
49. 作为应用开发者，我希望迁移期间保留现有 code-defined Adapter，从而可以引入 Markdown 路径而不需要一次性替换全部实现。
50. 作为应用开发者，我希望 Markdown-defined Specialist 成为生产目标，从而过渡性的 code Adapter 不会变成永久存在的第二套 authoring system。
51. 作为应用开发者，我希望一个 Specialist Catalog interface 隐藏 parsing、validation、storage 和 indexing 细节，从而 Agent Graph 不需要了解存储或 Markdown 行为。
52. 作为应用开发者，我希望在语义一致的部分复用现有 Agent Skills schema 和 loader 行为，从而 Specialist runtime 不会发明第二种 `SKILL.md` 方言。
53. 作为应用开发者，我希望现有模型生成的 Intent Result 和 Intent Catalog item schema 保持不变，从而 query understanding 向后兼容。
54. 作为应用开发者，我希望恢复或重试的 Task 通过当前进程 catalog 解析，从而 runtime recovery 不需要持久化完整的历史 Markdown。
55. 作为 eval 负责人，我希望拥有 description 歧义、多 Skill 激活和 Tool-scope 限制的确定性场景，从而在发布前衡量 Tenant 内容质量。
56. 作为平台运维人员，我希望 Specialist Definition 已被删除的合法已派发 Task 独立失败，从而同一 Batch 中其他有效 Tasks 仍能完成。

## 实现决策（Implementation Decisions）

### Authoring 与存储合同

- Tenant 由 `kmsAppId` 标识。平台从可信配置枚举已知 Tenants，只加载各自对应的
  definition prefix；不会根据存储内容发现或注册新的 Tenant。
- 存储合同采用相对目录结构：Agent 位于
  `tenants/{kmsAppId}/agents/{specialist-id}.agent.md`，Skill 位于
  `tenants/{kmsAppId}/skills/{skill-name}/SKILL.md`，每个 Skill 可以在自己的
  `references` 目录下包含可选文件。
- 开发环境选择以配置目录为根的 Local Adapter；生产环境选择以配置 bucket 和 prefix
  为根的 GCS Adapter。两者实现同一个 Loader interface，并向 catalog
  implementation 返回相同的 source documents 和 source identities。
- 存储选择属于环境配置，不是 Tenant-authored Markdown 的字段。runtime modules 只消费
  catalogs，不根据 Local 或 GCS 分支执行。
- Specialist Definition 的 YAML frontmatter 只包含 `id`、`description`、
  `model-profile` 和 `skills`。Markdown body 是 instruction content。Definition 中没有
  `role`、Tenant ID、Intent allowlist、Tool allowlist、output schema 或
  `required-skills` 字段。
- `model-profile` 必须通过当前 Tenant 的平台 Model Registry 解析为经过批准的 profile。
  无法解析 profile 的 Specialist Definition 无效。
- Specialist instruction body 最长 30,000 个字符。空内容或结构无效的 definition
  会被跳过并记录日志。
- Skills 继续使用仓库现有的 Agent Skills 兼容 frontmatter 和 Markdown 格式。
  `required-tools` 声明用于判断 invocation eligibility 的硬依赖。`allowed-tools` 是可移植
  的 Tool 使用指导，描述 Skill 在 Tool 通过其他规则可用时可以涉及哪些 Tool；它既不
  授予或筛选 Specialist 的有效 Tools，也不会让某个 Tool 成为必要条件。两个字段中的
  每个 Tool 名称都必须能在平台 Tool Registry 中解析。
- 系统不加载或执行 Skill package 中的 scripts。References 是由 `load_reference`
  读取的数据，不是可执行能力。

### Catalog module 与启动生命周期

- 构建一个深层 catalog module。其 interface 返回当前 Tenant 的 Specialist Catalog、
  Skill Catalog、简要 Specialist Descriptors，以及 invocation-specific Eligible Skill
  summaries。Parsing、validation、hashing、indexing 和无效条目日志属于该 module
  的 implementation details。
- 应用启动时，先建立可信 Tenant 配置和 Model Registries，再加载每个已知 Tenant，且
  必须在接收 request traffic 前完成。启动 composition 把 Tenant-scoped catalogs
  安装到 Agent runtime 可以根据可信 request context 解析的位置。
- 完整的 `.agent.md` 和 `SKILL.md` definitions 在启动时读取、解析、校验并缓存。
  Progressive disclosure 只描述模型能够看到什么，不表示 Skill definitions 从存储中
  lazy load。
- 启动时既不读取也不缓存 reference 文件内容。catalog 只保留足够可信的 source
  identity，以便后续定位 Activated Skill 的 reference directory。
- 一个无效 Specialist 或 Skill 不会回滚同一 Tenant 的其他有效 definitions。错误日志
  只包含有界的定位 metadata，不包含完整 instructions 或 reference 内容。
- `required-tools` 或 `allowed-tools` 中存在平台 Tool Registry 无法解析的 Tool 名称时，
  该 Skill 无效并从 Skill Catalog 中排除。如果 Specialist Definition 指向的 Skill
  不在成功加载的 Skill Catalog 中，则该
  Specialist Definition 无效并从 Specialist Catalog 排除；runtime 不会静默改写管理员
  声明的 Skill 列表。
- Catalog membership 在进程生命周期内不可变。Agent 或 Skill 修改需要重启。存储失败
  不得复用另一个 Tenant 的 catalog，也不得扩大 catalog membership。
- 现有 Agent Skills parsing 以及 Local/GCS storage implementations 是可复用的先例，
  在合同一致时应复用或深化。Agent Graph 不应维护第二套持久化 Skill schema 或 parser。
  当前 FlowEngine Agent Handler 在本阶段仍是另一个调用者，但不能要求 Specialist
  invocation 采用与新设计冲突的语义。

### Intent、Research Scope 与 Coordinator 选择

- 模型生成的 Intent Result 和 Intent Catalog item 结构保持不变。不得向模型输出增加
  `allowed_specialist_ids`、Skill IDs、Tool IDs 或 Agent routing 字段。
- Intent Policy 仍然是 Tools、sources、search constraints、Evidence freshness 及其他
  非 Agent Research Scope 上限的可信 Tenant 配置。它不筛选 Specialist Catalog 或
  Skill Catalog。
- 当前作为 Intent-specific policy 数据携带的 Specialist descriptors 不再是权限来源。
  Coordinator 每次运行或 dispatch Task 时，包括从 checkpoint 恢复以后，runtime 都从
  当前进程 Tenant Specialist Catalog 投影全部有效 descriptors 和可执行 definitions。
  因此，新增 Specialist 对恢复后的 Run 可见，已删除 Specialist 则不再存在。
- 从 checkpoint 恢复时，保留原 Run 的 Tool、source、query、freshness 及其他数据约束。
  checkpoint 中持久化的 Specialist descriptors 只是过期 projection，不能限制或扩大当前
  catalog。如果同一 ID 的内容已修改，则使用当前 description 和 definition，并在 attempt
  中记录当前 definition pin。
- Intent-specific `allowed_skill_names` 同样不再筛选 Skill discovery。Skill eligibility
  只来自 Specialist Definition、有效的 Skill Catalog membership，以及 `required-tools`
  在当前 invocation 中是否可用。
- Coordinator prompt 只接收简要 Specialist Descriptors。选择时不接收 Specialist
  instruction body、model profile 或 Skill list。
- 现有 Coordinator Decision 的确定性 validation 继续拒绝不属于当前 Tenant Specialist
  Catalog 的 ID。模型选择不能绕过 registry 和 Tenant checks。

### 共同的 Specialist runtime 合同

- 保留现有 Specialist execution seam。Specialist Actor 接收 `SpecialistTaskInput` 并返回
  `SpecialistAttempt`；Agent Graph 的确定性 acceptance 环节生成现有
  `SpecialistResult`，并放入 `TaskOutcome`。不增加 Markdown 专用的 Task、attempt、
  result 或 SSE schema。
- runtime catalog 将所选 Specialist 解析为 descriptor、批准的 model profile、
  instructions、Skill names、definition pin 以及构建一次 actor invocation 所需的 factory
  inputs。Agent Graph 不解析文件，也不构造存储路径。
- Markdown Adapter 使用解析后的 Tenant Model Registry、平台 Specialist instructions、
  固定 security guards、Tenant-authored Specialist instructions、冻结的 Tool bindings 和
  invocation-local Skill activation interface，创建现有 PydanticAI Specialist Actor。
- 迁移期间保留 code-defined Adapter。它从可信代码提供相同的 runtime 信息和 actor
  interface。其 registrations 必须直接提供 prompt-visible description 和稳定的 definition
  identity，而不能依赖每个 Intent 中的 descriptors。
- 通过 code Adapter 继续支持直接用于测试的 actors，但生产 composition 的目标是
  Markdown definitions。本功能不为 Coordinator 或 Synthesis actor 创建通用框架。
- Specialist Definition Pin 是已解析 Specialist metadata 和 instruction body 的规范化表示
  所对应的 SHA-256 content hash。它不包含 storage location、reference 内容和可变 runtime
  state。
- 每个 Specialist attempt 携带创建该 actor 时使用的 pin。Acceptance 会把 pin 复制到已
  接受的 Specialist Result 或 Task Outcome，使其随 Task record 持久化。系统不存储完整
  历史 Specialist 内容。
- 恢复或重试的 Task 通过当前进程 catalog 解析，不会恢复更早进程的 Definition body；
  ID 仍存在时，definition pin 使内容变化可观测。
- 将 Batch 完整性和当前 Specialist 可用性分开处理。Batch-level validation 继续拒绝无效
  Batch 或 Task identity、格式错误的 objective 或 context、伪造的 Specialist ID，以及
  与已接受 Coordinator Decision 不匹配的 Task。Task 被证明是合法派发后，只解析该 Task
  对应的 Specialist；如果其 definition 已删除，则不调用模型，直接产生失败 Task Outcome，
  并允许同一 Batch 的其他有效 Tasks 继续执行。不得替换为另一个 Specialist。

### Skill discovery、activation 与 references

- 在一次 Specialist invocation 中，一个 Eligible Skill 必须同时满足：被该 Specialist
  Definition 声明、存在于 Tenant Skill Catalog 中、其所有 `required-tools` dependencies
  都存在于本次 invocation 的有效 Tool set 中。
- 如果 `required-tools` dependency 已在全局注册，但被 Tenant Tool policy 或 Research
  Scope 排除，则该 Skill 不得出现在本次 invocation 的 Eligible Skill summaries 中。
  `allowed-tools` 中某个 Tool 当前不可用不会使 Skill 失去 eligibility。正常流程不应先
  暴露硬依赖未满足的 Skill，再依赖 activation-time failure 拒绝。
- 投影 Eligible Skill summaries 时保留管理员声明顺序，并最多暴露前二十个。初始
  Specialist prompt 不包含完整 Skill instructions。
- Specialist invocation 开始时计算并冻结 Eligible Skill set。移除当前只能激活一个
  Skill 的限制；invocation 可以在任意业务 Tool 调用之前或之后，渐进式激活该固定集合
  中的零个或多个 Skills。
- 重复激活同一个 Skill 保持幂等。Activated Skill Pins 按第一次激活的顺序保存且不重复，
  每个 Activated Skill 在当前 Specialist invocation 结束前持续有效。
- Skill Pin 包含 Skill name、可选 version，以及缓存 Skill Definition 的必填 content hash。
  `metadata.version` 存在时保留，不存在时省略；version 不是管理员必填字段，reference 内容
  不参与 hash。
- Skill definitions 互为平级。runtime 不设置优先级、不重排 instructions、不检测语义
  冲突、不降级 Skill，也不提供自动 fallback。Tenant 发布 review 和 eval 负责该内容
  质量控制。
- `load_reference` 只在具备 Skills 的 Specialist invocation 中提供。它接收一个 Activated
  Skill identity，并按稳定的文件名顺序读取该 Skill `references` 目录下的全部普通文件。
- 每次 `load_reference` 都通过选定的 storage Adapter 读取并返回最新内容，绕过进程级和
  invocation 级 reference cache。目录不存在或为空时成功返回空结果；列举目录失败或
  任一文件读取失败时，返回有界且明确的 Tool failure，不返回部分内容。Reference 内容
  不附加到缓存的 Skill Definition，也不参与 Specialist 或 Skill pin。
- Skill activation 和 reference loading 都不会注册业务 Tools，也不会改变冻结的有效
  Tool set。它们对模型可见的结果只包含 instructions 或 reference content，不包含新权限。

### Tool 权限与 instruction 优先级

- 平台 Tool Registry 是可执行 Tool definitions 的唯一来源。现有 Evidence 和 Calculation
  Tool registrations 必须通过这一共同 registry seam 提供，而不是复制到 Specialist
  自己拥有的 registry 中。
- 有效业务 Tools 是已经注册、同时被 Tenant Tool policy 和当前 Research Scope 允许的
  Tools。有效 Tool 计算中移除 Specialist registration 的 Tool allowlist。
- `required-tools` 属于 validation 和 eligibility metadata；`allowed-tools` 只属于
  validation 和 usage guidance。两个字段都不能注册、授予、重新绑定、筛选或扩大 Tool；
  Specialist instruction 同样不能。
- 构建 Specialist Actor 之前解析并冻结有效 Tools。Skill activation 和实时 reference
  修改不能改变该集合。
- 平台 Specialist instructions 和固定 security guards 最先组合。Tenant-authored
  Specialist instructions 附加在其后。Activated Skill instructions 随后通过模型可见的
  activation result 进入 invocation。
- Instruction 顺序不能替代确定性 enforcement。Tenant identity、Research Scope、Tool
  bindings、Evidence validation、Calculation validation、coordination limits 和 graph
  routing 继续由代码拥有。
- 不增加语义 prompt-conflict detector。要求覆盖平台规则的内容没有权限，因为可执行
  interfaces 和 graph transitions 仍受代码约束。

### 迁移与可观测性

- code-defined 与 Markdown-defined Specialist Adapters 只在行为比较和迁移期间并存。
  两者必须通过同一套合同测试；生产配置以 Markdown catalog 为目标。
- 避免 Agent Graph 中长期存在两套并行 Skill runtime models。把进程生命周期的 Skill
  Catalog 适配为 invocation-local discovery 和 activation state，不再延续当前独立
  `SkillRegistration` 中缓存 references 的语义。
- 启动日志包含每个 Tenant 成功加载和跳过的 Specialist、Skill 数量。definition-level
  error 包含 source identity 和稳定 reason，供 Tenant 管理员修复。
- 已接受 execution 的 telemetry 和持久化 Task 数据暴露 Specialist Definition Pin 与
  Activated Skill Pins，但不暴露完整 instructions、reference 内容、Tool-call history 或
  model messages。
- 不为保留历史 Markdown body 单独增加数据库迁移。如果现有序列化 Specialist Result 或
  Task Outcome 合同需要新增 pin 字段，应使用仓库现有的向后兼容 checkpoint/state 演进
  规则，并保证该字段有界。

## 测试决策（Testing Decisions）

- 良好的测试只断言 module interface 上可观察的行为：catalog membership、Coordinator
  可见的 descriptors、接受或拒绝 dispatch、绑定的 Tool names、模型可见的 Skill
  activation results、pins、Task Outcomes、日志和 SSE completion。测试不应断言私有
  dictionaries、parser helper 调用、storage implementation details 或完整 prompt 的精确
  字符串。
- 主要 seam 是现有 Specialist Actor 和 Agent Graph 合同。对 code-defined 和
  Markdown-defined Adapters 运行同一套合同测试。在给定等价可信 definitions 和确定性
  actors 时，两者必须接受相同 Specialist Task Input、保留相同 context、暴露相同 Tool
  set、产生相同 Specialist Attempt shape、通过相同 acceptance rules，并生成相同
  Specialist Result shape。不同 Adapter 的 pin 值可以不同，但都必须提供有效且稳定的 pin。
- 最高层集成 seam 是现有 `/v2/query/stream` HTTP/SSE Agent 路径，测试使用真实 Agent
  Graph、request Tenant headers、确定性 PydanticAI models 和 PostgreSQL checkpointing。
  使用本地 Tenant definition tree 证明启动加载、不变的 Intent Result handling、完整
  catalog 的 Coordinator 选择、Specialist dispatch、多个 Skill 交错激活、scope-limited
  Tool execution、accepted result publication 和持久化 pins。
- 扩展现有 Agent-first Specialist integration suite，不创建另一套端到端 harness。复用
  它已有的确定性 Coordinator、Specialist、Tool、SSE、concurrency、retry 和 checkpoint
  测试先例。
- 扩展现有 Specialist PydanticAI runtime tests，通过 sentinels 捕获 model messages。
  证明 activation 前只出现简要 summaries，可以按首次使用顺序激活多个不同 Skills，
  重复 activation 保持幂等，业务 Tool calls 可以出现在两次 activation 之间，且平台
  instructions 始终位于 Tenant instructions 之前。权限断言基于实际可用 Tool bindings，
  不能只相信 prompt 文本。
- 用多 Skill activation 行为替换当前 Specialist Skill tests 中只能激活一个 Skill 的
  assertions。覆盖零个、一个和多个 Skills、二十条 summary 上限、未知名称、缺失的
  `required-tools`、可选的 `allowed-tools`，以及被 Research Scope 排除的 Tools。不增加
  `max_activated_skills` 测试，因为该配置不存在。
- Local 和 GCS implementations 使共同 Loader 成为真实 seam。用共享 Adapter contract
  fixture 证明两者针对同一个 Tenant 相对目录返回等价的 Agent 和 Skill source
  documents、隔离 Tenant prefix、启动时读取完整 Agent 和 Skill definitions、启动时不
  读取 references，并且每次调用都返回最新 reference 内容。GCS 使用内存 fake；确定性
  测试不得依赖网络。
- 通过 catalog 的 public interface 测试有效和无效 definitions 混合存在的情况。验证无效
  条目不在 catalog 中、有效 sibling 保留、未知 model profile 以及 `required-tools` 或
  `allowed-tools` 中的未知 Tool 名称被拒绝、引用缺失 Skill 的 Specialist 被拒绝、
  instruction 长度有界，且日志能够定位 Tenant 和 source 但不会输出内容。
- 联合测试 Research Scope resolution 和 Coordinator projection：两个具有不同 Tool/
  source policies 的 Business Intents 必须得到同一份完整 Tenant Specialist Descriptor
  set，同时保留各自不同的非 Agent constraints。这项 regression test 防止 Intent Policy
  再次成为隐式 Specialist 或 Skill allowlist。
- 使用变化后的当前 catalog 恢复 checkpoint，同时保留原 Run 的 Tool、source、query 和
  freshness constraints。验证修改后的 descriptions 和 definitions 会刷新，新增
  Specialists 对 Coordinator 可见，删除 Specialists 不再出现在 discovery 中。
- 删除其中一个 Task 的 Specialist Definition 后，恢复包含两个 Tasks 的已接受 Batch。
  被删除 Specialist 对应的合法 Task 必须在不调用模型的情况下产生失败 Task Outcome，
  有效 sibling 正常运行，barrier 收集两个 outcomes。另行验证伪造 Specialist ID 或损坏
  Batch manifest 仍然触发整批 invariant validation failure，而不是普通 Task failure。
- 在现有 Specialist Catalog/Tool binding seam 测试 Tool authority。一个 Tool 只有在全局已
  注册、Tenant 允许且 Research Scope 允许时才能被调用。改变 Specialist 或 Skill
  metadata 永远不能增加 Tool。现有 Evidence、expected unavailability、Calculation、
  Data Gap 和 telemetry tests 继续作为 Tool-result 语义的权威测试。
- 激活一个 Skill 并读取 references，然后在不重建 catalog 的情况下修改 Local Adapter
  reference fixture，再次读取。第二次结果必须包含新内容，而 Agent Definition Pin 和
  Skill Pin 保持不变。共享 Local/GCS Adapter contract 还必须证明：空目录成功返回，而
  列举或读取失败不返回部分文档，并且能与空 reference set 明确区分。
- 分别测试包含和不包含 `metadata.version` 的 Skill Pin。两种形式都必须通过 validation、
  保持第一次 activation 的顺序，并携带缓存 Skill Definition 的 content hash；即使缺少
  version，definition 内容变化也必须改变 hash。
- 使用两个独立构建的 catalogs 测试重启语义，不实现 hot reload。第一个 catalog 继续
  使用已缓存的 Agent 和 Skill definitions；第二次启动能够看到修改。第二个 runtime 中
  重试的 Task 记录第二个 pin，并且不请求历史内容。
- 实现过程中先运行聚焦的确定性 unit 和 integration suites，再运行完整测试套件。
  PostgreSQL-backed tests 使用仓库标准 PostgreSQL test runner 和 fixture safety rules。

## 不在范围内（Out of Scope）

- 使用 Markdown 定义 Coordinator Agent、Query Understanding Agent 或 Synthesis Agent。
- 精确兼容 VS Code custom agent，或承诺任意 VS Code Agent 文件无需修改即可运行。
- 修改模型生成的 Intent Result 或 Intent Catalog item schema，包括增加
  `allowed_specialist_ids`。
- Intent-specific Specialist 或 Skill allowlists。
- Specialist-specific Tool allowlist，或通过 Agent/Skill Markdown 授予任何 Tool。
- `max_activated_skills` 设置、强制只激活一个 Skill，或 Skills 之间的
  `required-skills` dependency mechanism。
- 在 runtime 中检测、排序、合并冲突的 Skill instructions，或自动从冲突中恢复。
- `.agent.md` 或 `SKILL.md` 的 hot reload。
- reference 文件内容的缓存、pin、version 或历史保留。
- 为精确历史重放而持久化完整 Specialist 或 Skill definitions。
- 为恢复后的 Run 持久化历史 Specialist ID snapshot，或将其与当前 catalog 求交集。
- 对整个 Tenant catalog 执行 atomic all-or-nothing loading。
- 执行 Local 或 GCS definition storage 中的 Skill scripts、binaries、assets 或任意代码。
- Tenant 管理员 UI、publishing workflow、approval workflow 或 eval authoring interface。
- 在同一个变更中移除 code-defined Specialist Adapter；完成行为一致性验证后再进行移除。
- 除复用本功能需要的 Agent Skills parsing 和 storage 行为外，重构 legacy FlowEngine
  request path 或其 Agent Handler。

## 补充说明（Further Notes）

- 本规格落实已经接受的决策：借鉴 VS Code Agent Markdown 和开放 Agent Skills 格式中
  有用的 authoring conventions，但不完整采用任一方的 runtime。
- `description` 是 Specialist 和 Skill discovery 的 routing contract。其内容质量对运行
  结果很重要，应该通过 Tenant publishing review 和 eval 覆盖；runtime 本身只进行结构
  validation。
- Definition pins 用于可观测性，不保证精确重放。Agent 和 Skill definitions 只在单个
  进程内保持稳定，重试使用当前进程 catalog，references 则有意保持实时。
- 该设计有三个可测试 seams，但只有两个 runtime-facing interfaces：catalog/Loader 一侧，
  以及现有 Specialist Actor/Agent Graph 一侧。Parsing、hashing、validation 和 prompt
  projection 应隐藏在这些 interfaces 后面，避免 callers 逐渐了解 storage 或 Markdown
  细节。
