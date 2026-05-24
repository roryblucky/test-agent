# RAG Citation Attribution V4.2 代码变更细节

本文记录 RAG 引用溯源 V4.2 的实际代码变更：每个模块改了什么、为什么这样改，以及它满足的设计目标。

## 1. 总体目标

本次改造让最终回答支持 `[1][2]` 行内引用，并在 API / SSE 中返回结构化 `citations`，包含来源、引用原文、可高亮内容和 UTF-16 offset。

核心目标：

- 后端生成稳定 citation index。
- LLM answer 按 evidence 中给定的 `[n]` 输出引用标记。
- answer 生成完成后，后处理构建 citation metadata。
- 对推理型引用，用 fast LLM 提取支撑原文，再由后端验证定位。
- 对 LLM quote 失败或无法定位的情况，用 deterministic fallback scorer 找支撑窗口。
- API 顶层返回 `citations`，`metadata` 不重复携带同一份数据。
- SSE 统一为实时 `token` 流式，再发送 `citations`，最后 `done`。
- 彻底删除旧的 `llm:compliance_review` / `approved_answer_only` / `answer_delta` 延迟释出流程。

## 2. 数据模型变更

### `app/models/workflow.py`

改了什么：

- `AggregatedEvidence` 新增：
  - `citation_index`
  - `source_type`
  - `page_number`
  - `section`
- 新增 `CitationReference` 模型，包含：
  - citation 基础字段：`index`、`evidence_id`、`source`、`source_type`
  - 来源展示字段：`title`、`url`、`snippet`、`page_number`、`section`、`published_at`
  - 高亮字段：`quoted_text`、`quoted_passages`、`highlight_content`、`highlight_spans`
  - 协议字段：`offset_encoding="utf-16"`、`attribution_status`
  - 扩展字段：`metadata`
- 删除 `ComplianceReviewResult`。

为什么这样做：

- `AggregatedEvidence.citation_index` 用于把聚合证据和 answer 里的 `[n]` 稳定绑定。
- `source_type/page_number/section` 让 citation 能覆盖 Discovery Engine、MCP API、NL-to-SQL 等不同来源。
- `CitationReference` 是 API / SSE 的统一 citation contract，满足前端或其他客户端直接消费高亮信息的需求。
- 删除 `ComplianceReviewResult` 是为了满足 V4.2 统一实时 token streaming 的协议，不再保留旧 delayed compliance release 模型。

### `app/models/domain.py`

改了什么：

- `Document` 新增：
  - `source_url`
  - `source_type`
  - `page_number`
  - `section_title`

为什么这样做：

- classic RAG 路径没有 `AggregatedEvidence`，需要从 `Document` 包装出 citation evidence。
- 这些字段让 classic RAG 也能返回 URL、页码、章节等来源信息，满足 API contract 对 citation metadata 的要求。

### `app/services/flow_context.py`

改了什么：

- 删除 `draft_answer`。
- 删除 `compliance_review`。
- 删除 `ComplianceReviewResult` 导入。

为什么这样做：

- `draft_answer` 和 `compliance_review` 只服务于旧的 approved-answer buffering 流程。
- V4.2 不再允许“已流式输出的 answer 被后置合规步骤替换”，因此这些状态必须删除，避免协议语义混乱。

## 3. API 与 SSE 协议变更

### `app/api/schemas.py`

改了什么：

- `QueryResponse` 新增顶层字段：
  - `citations: list[CitationReference]`
- `from_flow_context()` 从 `ctx.metadata` 的 copy 中 `pop("citations", [])`，构建顶层 `citations`。
- 不 mutate 原始 `ctx.metadata`。

为什么这样做：

- 顶层 `citations` 是新的 API contract，调用方无需从 metadata 中解析业务字段。
- 用 copy 后 `pop` 避免同一份 citation payload 同时出现在 `metadata` 和顶层字段。
- 不修改 `ctx.metadata`，避免影响后续 handler、审计或测试观察上下文。

### `app/services/events.py`

改了什么：

- 新增 `EventType.CITATIONS = "citations"`。
- 新增 `EventEmitter.emit_citations()`。
- 删除 `EventType.ANSWER_DELTA`。
- 删除 `emit_answer_delta()`。
- 更新模块文档字符串，移除 high-compliance buffering / `answer_delta` 说明。

为什么这样做：

- SSE 协议改为 `token -> citations -> done`。
- `citations` 是 answer 完成后的后处理结果，必须作为独立事件发送。
- 删除 `answer_delta` 是为了彻底移除旧的延迟释出协议，避免客户端同时支持两套互相冲突的流式语义。

## 4. 证据聚合与 Prompt 注入

### `app/services/handlers/aggregation.py`

改了什么：

- `_select_evidence()` 在最终 selected evidence 上分配：
  - `citation_index=index`
  - `source_type=evidence.metadata.get("source_type") or evidence.source`
  - `page_number` 兼容 `page_number` / `pageNumber`
  - `section=evidence.title`

为什么这样做：

- 生成 answer 前必须有稳定的 citation index，才能在 prompt 中要求模型引用 `[n]`。
- `source_type/page_number/section` 在聚合阶段就落到 evidence 上，后续 citation extraction 不需要猜来源字段。
- 不引入 `record_source_map`，避免使用不存在的隐式映射。

### `app/services/handlers/llm.py`

改了什么：

- `_build_answer_runtime_prompt()` 的 `<runtime_rules>` 增加 citation guardrails：
  - 必须用 `[n]` 引用事实性 claim。
  - 只能使用 evidence tags 中出现的 citation index。
  - 多来源 claim 使用 `[1][3]`。
  - 不得编造 citation number。
  - 不得引用 unsupported claims。
- `_format_aggregated_evidence()`：
  - 有整型 `citation_index` 时渲染为 `[Evidence [n] | ...]` 并加 `Cite as: [n]`。
  - 没有整型 `citation_index` 时渲染为 `[Evidence {evidence_id} | ...]`，不加 `Cite as`。
- classic RAG fallback document 格式改为 `[Document [n] | id=...]` 并加 `Cite as: [n]`。
- `_llm_answer()` 统一实时 `emit_token()`。
- answer 完成后调用 `build_citations()`。
- 将 citation extraction usage 累加到 `ctx.total_usage`。
- 将 citations 写入 `ctx.metadata["citations"]`。
- 若存在 SSE emitter，先发送 `emit_citations()`，再发送 `emit_step_completed("llm:answer")`。
- 删除全部 `compliance_review` 相关 import、factory、handler、coerce、buffer/release helper。

为什么这样做：

- Citation guardrails 放在 runtime prompt 中，不改变 tenant identity 和静态 layered prompt 缓存语义。
- `Cite as: [n]` 给 answer model 明确可用 citation marker。
- `citation_index=None` 时不渲染 `[Evidence [non-number]]`，避免奇怪 bracket nesting，也避免模型输出无法解析的 marker。
- citation extraction 只能基于最终 `ctx.llm_response`，所以放在 answer streaming 完成后执行。
- `emit_citations()` 在 `llm:answer` step completed 前发送，满足事件顺序：tokens 已经结束，但 answer step 仍拥有 citation 后处理。
- 删除 compliance 路径是为了防止“客户端已看到 A，最终 response 被替换成 B”的协议级不一致。

## 5. 新增 Citation Extraction 服务

### `app/services/citation_extractor.py`

改了什么：

- 新增 `ClaimExtraction` 和 `LocatedSpan` dataclass。
- 新增 `QuoteExtractionItem` / `QuoteExtractionResult` 结构化 LLM 输出模型。
- 新增 `extract_claims()`。
- 新增 `extract_quotes_with_llm()`。
- 新增 quote locator chain：
  - exact substring
  - relaxed whitespace
  - punctuation-normalized projection
  - multi-sentence split
- 新增 `validate_and_locate_quotes()`，对 located spans 排序、去重、合并，并从原文重新切片生成 text。
- 新增 fallback scorer：
  - sentence windows
  - line-based rows
  - JSON/table-like row blocks
  - 数字、时间实体、单位、关键词、fuzzy similarity 多信号评分
- 新增 UTF-16 offset 转换：
  - `py_index_to_utf16_offset()`
  - `span_to_utf16()`
- 新增 `safe_parse_page_number()`。
- 新增 `build_evidence_index()`。
- 新增 `build_citations()` orchestrator。

为什么这样做：

- `extract_claims()` 用中英文句子切分正则识别 `[n]` 所属 claim，满足直接引用、改写引用、推理引用的统一入口。
- Fast LLM quote extraction 用于解决推理型 citation：answer 里的结论可能不是原文 substring，必须让 LLM 找支撑原始数据句。
- LLM 返回的 quote 不能直接信任，必须由后端定位验证，满足“只高亮 evidence 中真实存在的文字”。
- relaxed whitespace 使用 `re.sub(r"(?:\\\s)+", lambda _: r"\s+", re.escape(passage))`，避免 Python 3.12+ replacement escape 问题。
- punctuation projection 处理中英文标点、连字符、引号差异。
- multi-sentence split 允许 LLM 返回多个句子的组合 quote。
- fallback scorer 不是 claim 对全文 fuzzy，而是找 evidence 支撑窗口，满足推理/计算型引用在 LLM quote 失败时仍能降级。
- UTF-16 offset 满足 Web / JavaScript 字符串 offset 语义，避免 emoji/surrogate pair 错位。
- `safe_parse_page_number()` 处理 `"3.0"`、`3.7`、`"3"` 等常见 metadata 格式，避免 citation 构建因页码格式崩溃。
- `build_citations()` 首先检查是否存在 claims；没有 `[n]` 时直接返回，避免无意义 fast LLM 调用。
- answer 中有 `[n]` 但缺失对应 evidence 时跳过，不抛异常，满足模型误输出 marker 的容错要求。

## 6. 删除旧 Compliance Review 流程

### `app/agents/compliance_review.py`

改了什么：

- 删除整个 compliance review agent 文件。

为什么这样做：

- 该 agent 只服务于 `llm:compliance_review` delayed release 流程。
- V4.2 已决定不保留该流程，物理删除可以避免后续误用。

### `app/services/flow_engine.py`

改了什么：

- `self.streaming_policy` 固定为 `"token"`。
- 删除 `_has_compliance_review_step()`。
- 更新文档字符串，移除 high-compliance buffering 说明。

为什么这样做：

- 流式策略不再由是否存在 `llm:compliance_review` 决定。
- 所有 answer 流统一为实时 token streaming。

### `app/services/handlers/analysis.py`

改了什么：

- 删除 `ctx.compliance_review` 读取。
- 删除 `compliance_passed`。
- 删除 `compliance_violation_count`。

为什么这样做：

- `FlowContext` 不再有 compliance review 状态。
- analysis metadata 必须反映当前真实执行模型，避免继续输出已删除流程的指标。

## 7. 文档变更

### `docs/rag_citation_attribution_v4_2_zh.md`

改了什么：

- 新增 RAG Citation Attribution V4.2 设计方案文档。
- 包含 API contract、模型变更、核心算法、LLM handler 接入、删除项、verification plan 和实现 agent prompt。

为什么这样做：

- 为后续维护者和 coding agent 提供单一设计基线。
- 避免每次实现或 review 都重新推导 quote extraction、fallback scorer、UTF-16 offset 等决策。

### `docs/agentic_workflow_platform_design_zh.md`

改了什么：

- 在文档开头增加 V4.2 废弃公告。
- 对 `compliance_review`、`approved_answer_only`、`answer_delta` 等旧设计增加 `[DEPRECATED by RAG Citation Attribution V4.2]` 批注。

为什么这样做：

- 该文档仍包含旧平台演进设想。
- 不直接大规模重写历史设计，而是标注废弃上下文，避免后续 agent 或工程师误以为 compliance buffering 仍是当前目标架构。

## 8. 测试变更

### 新增 `tests/unit/test_citation_extractor.py`

覆盖内容：

- decimal sentence splitting。
- `[n]` 句首归属前一句。
- relaxed whitespace regex。
- UTF-16 offset。
- span 合并并从原文重新切片。
- exact locator。
- relaxed whitespace locator。
- punctuation-normalized locator。
- multi-sentence split locator。
- numeric/unit fallback scoring。
- 缺失 evidence 容错。
- unlocated fallback。
- mocked fast LLM quote extraction。
- tolerant page number parsing。

为什么这样做：

- citation extraction 是本次最核心、最容易出现 offset/regex/fallback bug 的模块，必须以单测锁住行为。
- 特别覆盖 Python regex、UTF-16 offset 和 `"3.0"` page number 这些历史 review 中指出的边界。

### 更新 workflow / LLM / streaming 测试

改了什么：

- 删除 `tests/unit/test_llm_compliance_review.py`。
- 移除测试中对 `approved_answer_only`、`answer_buffering`、`emit_answer_delta` 的断言。
- 将 planner aggregation 集成测试改为 citation workflow：
  - answer token streaming。
  - citation extraction。
  - `emit_citations()` 在 `llm:answer` completed 前发送。
  - `QueryResponse.citations` 顶层存在。
- 更新 analysis / flow engine / workflow contract tests，匹配删除 compliance 状态后的模型。

为什么这样做：

- 测试必须表达新的协议：实时 token + citations 后处理。
- 旧 compliance tests 如果保留，会持续鼓励恢复已删除的 delayed release 语义。

## 9. 验证结果

已验证：

```bash
git diff --check
PYTHONPATH=. uv run ruff check app/services/citation_extractor.py tests/unit/test_citation_extractor.py tests/integration/test_workflow_flows.py app/models/domain.py app/services/flow_engine.py app/services/handlers/llm.py
PYTHONPATH=. uv run pytest tests/ -k "citation" -v
PYTHONPATH=. uv run pytest tests/unit/test_llm_answer_aggregation.py tests/unit/test_streaming_events.py -v
PYTHONPATH=. uv run pytest tests/ -v
```

结果：

- changed-files ruff check 通过。
- citation tests 通过。
- targeted LLM / streaming tests 通过。
- full test suite 通过。

## 10. 行为总结

最终运行时行为：

1. aggregation 为 evidence 分配 `citation_index`。
2. answer prompt 中 evidence 标注 `[n]` 和 `Cite as: [n]`。
3. LLM 实时输出 answer token，包含 `[n]`。
4. answer 完成后调用 citation extractor。
5. extractor 解析 claims。
6. fast LLM 提取支撑 quote。
7. 后端验证 quote 并计算 UTF-16 spans。
8. LLM quote 无法定位时，fallback scorer 找支撑窗口。
9. citations 写入 `ctx.metadata["citations"]`。
10. SSE 发送 `citations`。
11. `QueryResponse.from_flow_context()` 将 citations 移到顶层返回。

这套实现满足 V4.2 的核心约束：**LLM 负责理解引用，后端负责验证定位，fallback 负责找证据窗口，API 返回 Web-safe UTF-16 spans。**
