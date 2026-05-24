# RAG Citation Attribution V4.2 设计方案

本文档定义 RAG 引用溯源能力的 backend/API/SSE 实现方案。目标是在最终回答中保留行内引用标记 `[1][2]`，并通过 API 返回每个引用对应的来源、支撑原文和可高亮 span。

本方案只覆盖后端协议和实现，不包含 React 或其他前端渲染方案。

## 1. 目标和范围

目标：

- 最终 answer 保留 `[1][2]` 行内引用。
- API response 顶层返回 `citations`。
- SSE 在 answer token 之后、`done` 之前发送 `citations` 事件。
- 每个 citation 返回来源 metadata、`quoted_passages`、`quoted_text`、`highlight_content`、`highlight_spans`。
- `highlight_spans` 使用 `offset_encoding="utf-16"`，便于 Web/JavaScript 消费。
- 推理型引用由 fast LLM 识别支撑原文，后端负责验证和定位。
- LLM quote 无法验证时，用 deterministic fallback window scorer 找支撑证据窗口。

不在范围内：

- 不实现前端 hover/click/preview card。
- 不设计 React hook 或浏览器 EventSource/fetch 细节。
- 不保留 `approved_answer_only` / delayed compliance release 相关流程。
- 不在本方案中实现 GCP Discovery Engine provider；如果另行实现 provider，再单独引入 `google-cloud-discoveryengine` 依赖。

## 2. 核心原则

- LLM 负责理解哪段原文支撑引用。
- 后端负责验证 LLM 返回的 passage 是否真的存在于对应 evidence。
- 后端只对验证成功的 passage 生成高亮 span。
- 某个 citation 的 LLM passage 全部无法定位时，再启用 fallback scorer。
- fallback scorer 找的是 evidence 中的支撑窗口，不是简单拿 claim 对全文做 fuzzy。
- citation extraction 只基于最终 `ctx.llm_response`，避免 draft/released answer 不一致。
- `citations` 只出现在 top-level response，不重复放在 `metadata`。

## 3. API Contract

非流式 response：

```json
{
  "answer": "公司收入环比增长了20%[1]。",
  "citations": [
    {
      "index": 1,
      "evidence_id": "evidence:1:row-123",
      "source": "finance_db",
      "source_type": "mcp_sql",
      "title": "Revenue Database Query",
      "url": null,
      "snippet": "Q3收入12亿美元，Q2收入10亿美元",
      "quoted_text": "Q3收入12亿美元，Q2收入10亿美元",
      "quoted_passages": ["Q3收入12亿美元，Q2收入10亿美元"],
      "page_number": null,
      "section": null,
      "published_at": null,
      "highlight_content": "Q3收入12亿美元，Q2收入10亿美元",
      "highlight_spans": [{"start": 0, "end": 21}],
      "offset_encoding": "utf-16",
      "attribution_status": "located",
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

SSE 事件顺序：

```text
token ... token
citations
done
```

`citations` event 的 `data` 与 response 顶层 `citations` 字段结构一致。

`attribution_status` 取值：

- `located`：fast LLM 返回 passage，且后端验证定位成功。
- `fallback_located`：LLM 失败或无有效 passage，fallback scorer 找到支撑窗口。
- `unlocated`：有 citation 来源，但无法可靠定位具体高亮文本。

## 4. 数据模型变更

### 4.1 `app/models/workflow.py`

`AggregatedEvidence` 增加字段：

```python
citation_index: int | None = None
source_type: str | None = None
page_number: int | None = None
section: str | None = None
```

新增 `CitationReference`：

```python
class CitationReference(BaseModel):
    """Citation metadata for API/SSE rendering with highlight support."""

    index: int
    evidence_id: str
    source: str
    source_type: str | None = None

    title: str | None = None
    url: str | None = None
    snippet: str | None = None

    quoted_text: str | None = None
    quoted_passages: list[str] = Field(default_factory=list)

    page_number: int | None = None
    section: str | None = None
    published_at: datetime | None = None

    highlight_content: str | None = None
    highlight_spans: list[dict[str, int]] = Field(default_factory=list)
    offset_encoding: Literal["utf-16"] = "utf-16"
    attribution_status: Literal[
        "located",
        "fallback_located",
        "unlocated",
    ] = "unlocated"

    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 4.2 `app/models/domain.py`

`Document` 增加 classic RAG fallback 字段：

```python
source_url: str | None = None
source_type: str | None = None
page_number: int | None = None
section_title: str | None = None
```

### 4.3 `app/api/schemas.py`

`QueryResponse` 增加：

```python
citations: list[CitationReference] = Field(default_factory=list)
```

`from_flow_context()` 必须从 metadata copy 中取出 citations，不能 mutate `ctx.metadata`：

```python
final_meta = dict(ctx.metadata) if ctx.metadata else {}
citations_raw = final_meta.pop("citations", [])
citations = [
    c if isinstance(c, CitationReference)
    else CitationReference.model_validate(c)
    for c in citations_raw
]
```

返回时：

```python
return cls(
    ...,
    metadata=final_meta,
    citations=citations,
)
```

## 5. Citation Index 分配

在 aggregation 选中 evidence 后，为每个 selected evidence 分配稳定 citation index。

修改 `app/services/handlers/aggregation.py`：

```python
selected = [
    evidence.model_copy(
        update={
            "evidence_id": f"evidence:{index}:{evidence.metadata['item_id']}",
            "citation_index": index,
            "source_type": evidence.metadata.get("source_type") or evidence.source,
            "page_number": (
                evidence.metadata.get("page_number")
                or evidence.metadata.get("pageNumber")
            ),
            "section": evidence.title,
        }
    )
    for index, evidence in enumerate(selected, start=1)
]
```

不要使用未定义的 `record_source_map`。

## 6. Prompt 和 Evidence 格式

answer prompt 增加 guardrails：

```text
- ALWAYS cite retrieved factual claims using inline markers [n].
- Use only citation indexes shown in evidence tags.
- Cite as [1], [2], or [1][3] for multi-source claims.
- Do NOT invent citation numbers.
- Do NOT cite unsupported claims.
```

aggregated evidence 格式：

```text
[Evidence [1] | source=finance_db | relevance=high]
Cite as: [1]
Title: Revenue Database Query
Q3收入12亿美元，Q2收入10亿美元
```

classic RAG document fallback 格式：

```text
[Document [1] | id=doc-1]
Cite as: [1]
...
```

## 7. 核心模块：`app/services/citation_extractor.py`

新增 `citation_extractor.py`，包含以下职责：

- claim extraction
- evidence index construction
- fast LLM quote extraction
- LLM passage validation and location
- fallback supporting-window scoring
- UTF-16 span conversion
- citation orchestration

### 7.1 Claim Extraction

句子切分正则固定为：

```python
_SENTENCE_PATTERN = re.compile(
    r'[^。！？\n]+?(?:[。！？\n]+|\.(?=\s|$)|$)'
)
```

原因：

- 能正确切分 `"The total count is 10. Next sentence."`。
- 不会误切小数点，例如 `15.2`，因为小数点后面不是空白或字符串结尾。
- 支持中文句号、问号、感叹号和换行。

`extract_claims(answer)` 输出：

```python
@dataclass(frozen=True)
class ClaimExtraction:
    citation_index: int
    claim_text: str
    position_in_answer: int
```

规则：

- 找出 answer 中每个 `[n]`。
- 同一个 index 只处理一次。
- `[n]` 默认归属所在句。
- 如果 `[n]` 出现在句首附近，例如 `"第一句。[1]第二句"`，归属前一句。
- `claim_text` 去除 citation marker。

### 7.2 Fast LLM Quote Extraction

结构化输出：

```python
class QuoteExtractionItem(BaseModel):
    citation_index: int
    quoted_passages: list[str] = Field(default_factory=list)


class QuoteExtractionResult(BaseModel):
    extractions: list[QuoteExtractionItem] = Field(default_factory=list)
```

调用：

```python
agent = registry.create_agent(
    "fast",
    output_type=QuoteExtractionResult,
    instructions=QUOTE_EXTRACTION_INSTRUCTIONS,
)
result = await agent.run(prompt)
```

异常时返回 `{}`，由 fallback 处理。

LLM instructions 必须强调：

- 返回 evidence 中的 exact verbatim substring。
- 改写型 claim 返回被改写的原始句。
- 推理/计算型 claim 返回支撑推理的原始数据句。
- 找不到时返回空 `quoted_passages`。
- 不要返回整段 chunk，尽量只返回相关句或数据行。

### 7.3 LLM Passage 验证定位

LLM 返回的 passage 不能直接信任。每条 passage 必须在对应 evidence 中定位成功才保留。

定位顺序：

1. exact substring
2. relaxed whitespace locator
3. punctuation-normalized locator
4. multi-sentence split locator

定位结果：

```python
@dataclass(frozen=True)
class LocatedSpan:
    start: int
    end: int
    text: str
```

#### 7.3.1 Relaxed Whitespace Locator

宽松空白匹配必须使用 lambda replacement，避免 Python 3.12+ 把 replacement 中的 `\s` 当作非法替换模板解析：

```python
escaped_passage = re.escape(passage)
pattern_str = re.sub(
    r'(?:\\\s)+',
    lambda _: r'\s+',
    escaped_passage,
)
match = re.search(pattern_str, evidence_content)
```

这个 pattern 已用 Python 验证：

```python
escaped = re.escape("hello  world")
assert escaped == r"hello\ \ world"
assert re.sub(r"(?:\\\s)+", lambda _: r"\s+", escaped) == r"hello\s+world"
```

不要改成 `r'(?:\\\\\s)+'`。后者匹配两个 literal backslashes + whitespace，而 `re.escape()` 的空格序列只有一个 literal backslash + whitespace，会导致匹配失败。

#### 7.3.2 Punctuation-normalized Locator

推荐字符投影法：

- 遍历 raw evidence。
- 只保留 `char.isalnum()` 的 lower 字符作为 normalized evidence。
- 保存 normalized index 到 raw index 的映射。
- passage 同样归一化。
- 在 normalized evidence 中查找 normalized passage。
- 找到后通过映射还原 raw `start/end`。

### 7.4 Fallback Supporting Window Scorer

如果某个 citation 没有任何有效 located LLM span，进入 fallback scorer。

候选窗口：

- 单句。
- 连续 2 句。
- 连续 3 句。
- SQL/table-like 行。
- JSON-ish row block。

评分信号：

- 数字重合度：最高权重，适合推理/计算引用。
- 时间/实体重合度：例如 Q2、Q3、年份、公司名、产品名。
- 单位重合度：例如 `%`、`亿`、`美元`、`revenue`、`income`。
- 关键词重合度。
- `SequenceMatcher` fuzzy similarity。

有数字 claim 时，数字/时间/单位权重大于 fuzzy。返回 top 2-3 个非重叠窗口，并按原文顺序排序。

全部失败时：

```json
{
  "quoted_text": null,
  "quoted_passages": [],
  "highlight_spans": [],
  "attribution_status": "unlocated"
}
```

### 7.5 UTF-16 Offset

Python 定位得到的是 codepoint offset，API 返回前必须转换成 UTF-16 code unit offset。

```python
def py_index_to_utf16_offset(text: str, index: int) -> int:
    return len(text[:index].encode("utf-16-le")) // 2


def span_to_utf16(text: str, start: int, end: int) -> dict[str, int]:
    return {
        "start": py_index_to_utf16_offset(text, start),
        "end": py_index_to_utf16_offset(text, end),
    }
```

示例：

```python
text = "A😊B"
idx = text.index("B")
assert idx == 2
assert py_index_to_utf16_offset(text, idx) == 3
```

### 7.6 Orchestrator

`build_citations()` 伪代码：

```python
async def build_citations(
    answer: str,
    evidence_items: list[AggregatedEvidence],
    documents: list[Any] | None = None,
    registry: Any | None = None,
) -> list[CitationReference]:
    claims = extract_claims(answer)
    evidence_by_index = build_evidence_index(evidence_items, documents)

    llm_quotes: dict[int, list[str]] = {}
    if registry:
        evidence_map = {
            idx: ev.content
            for idx, ev in evidence_by_index.items()
        }
        llm_quotes = await extract_quotes_with_llm(answer, evidence_map, registry)

    citations: list[CitationReference] = []
    for claim in claims:
        evidence = evidence_by_index.get(claim.citation_index)
        if evidence is None:
            continue

        located = validate_and_locate_quotes(
            evidence.content,
            llm_quotes.get(claim.citation_index, []),
        )

        status = "located"
        if not located:
            located = fallback_locate_supporting_windows(
                claim.claim_text,
                evidence.content,
            )
            status = "fallback_located" if located else "unlocated"

        spans = [
            span_to_utf16(evidence.content, span.start, span.end)
            for span in located
        ]
        passages = [span.text for span in located]

        citations.append(CitationReference(
            index=claim.citation_index,
            evidence_id=evidence.evidence_id,
            source=evidence.source,
            source_type=evidence.source_type,
            title=evidence.title,
            url=evidence.url,
            snippet=evidence.content[:300],
            quoted_text=" ... ".join(passages) if passages else None,
            quoted_passages=passages,
            page_number=evidence.page_number,
            section=evidence.section,
            published_at=evidence.published_at,
            highlight_content=evidence.content,
            highlight_spans=spans,
            offset_encoding="utf-16",
            attribution_status=status,
            metadata=evidence.metadata,
        ))

    return sorted(citations, key=lambda citation: citation.index)
```

注意：`evidence_map` 必须显式从 `evidence_by_index` 构建，避免 `NameError`。

## 8. LLM Handler 接入

在 `app/services/handlers/llm.py` 的 `_llm_answer()` 中：

1. 正常 streaming answer token。
2. 设置最终 `ctx.llm_response`。
3. 调用 `build_citations()`。
4. 写入 `ctx.metadata["citations"]`。
5. 若有 emitter 且 citations 非空，发送 `citations` 事件。
6. 再发送 `emit_step_completed("llm:answer", ...)`。

示例：

```python
citations = await build_citations(
    answer=ctx.llm_response or "",
    evidence_items=(
        ctx.aggregated_evidence.selected_evidence
        if ctx.aggregated_evidence else []
    ),
    documents=ctx.ranked_documents or ctx.documents,
    registry=self.registry,
)

ctx.metadata["citations"] = [
    citation.model_dump(mode="json")
    for citation in citations
]

if ctx.emitter and citations:
    await ctx.emitter.emit_citations(
        [citation.model_dump(mode="json") for citation in citations]
    )
```

## 9. Events

`app/services/events.py`：

```python
class EventType(StrEnum):
    ...
    CITATIONS = "citations"
```

```python
async def emit_citations(self, citations: list[dict]) -> None:
    """Emit citation metadata with highlight spans."""
    await self.emit(StreamEvent(type=EventType.CITATIONS, data=citations))
```

## 10. 删除或废弃项

删除或废弃：

- `_STREAMING_POLICY_APPROVED_ONLY`
- `_should_buffer_answer`
- `_should_release_after_review`
- `_llm_compliance_review`
- `answer_delta` release 分支
- `approved_answer_only` 相关 tests

保留普通 token streaming：`emit_token(chunk)`。

## 11. Verification Plan

运行：

```bash
uv run pytest tests/ -k "citation" -v
uv run pytest tests/unit/test_llm_answer_aggregation.py tests/unit/test_streaming_events.py -v
```

必须覆盖：

- 直接引用 exact 定位。
- 改写引用：LLM 返回原文后定位成功。
- 推理引用：LLM 返回 Q2/Q3 原始数据后定位成功。
- LLM 返回 paraphrase：验证失败，进入 fallback。
- LLM 返回不存在 passage：验证失败，进入 fallback。
- fallback scorer 通过数字/时间/单位命中支撑窗口。
- 全部失败时 `attribution_status="unlocated"` 且 `highlight_spans=[]`。
- UTF-16 offset：emoji/surrogate pair 不错位。
- relaxed whitespace regex 正确处理 `re.escape("hello  world")`。
- 数字结尾句子 `"The total count is 10. Next sentence."` 正确切分。
- `QueryResponse` 中 `citations` 只在 top-level，metadata 不重复。
- SSE 支持 `citations` 事件。

## 12. 给实现 Agent 的 Prompt

```text
你是 agent-kms 仓库的实现 agent。请实现 RAG Citation Attribution V4.2，遵守仓库 AGENTS.md 和现有代码风格，保持 diff 小而可测。

目标：
在最终 RAG answer 中保留 [1][2] 行内引用，并通过 API/SSE 返回 citations。每个 citation 包含来源 metadata、quoted_passages、quoted_text、highlight_content、UTF-16 highlight_spans、offset_encoding="utf-16"、attribution_status。只做 backend/API/SSE，不做前端。

必须修改：

1. app/models/workflow.py
   - AggregatedEvidence 增加 citation_index/source_type/page_number/section。
   - 新增 CitationReference 模型，字段包括 index/evidence_id/source/source_type/title/url/snippet/quoted_text/quoted_passages/page_number/section/published_at/highlight_content/highlight_spans/offset_encoding/attribution_status/metadata。

2. app/models/domain.py
   - Document 增加 source_url/source_type/page_number/section_title。

3. app/api/schemas.py
   - QueryResponse 增加 citations: list[CitationReference]。
   - from_flow_context() 用 final_meta = dict(ctx.metadata)，从 copy 里 pop("citations", [])，不要 mutate ctx.metadata。
   - citations 只出现在 top-level response，不重复放 metadata。

4. app/services/events.py
   - EventType 增加 CITATIONS = "citations"。
   - EventEmitter 增加 emit_citations(citations: list[dict])。

5. app/services/handlers/aggregation.py
   - selected evidence 分配 citation_index。
   - source_type = evidence.metadata.get("source_type") or evidence.source。
   - page_number 兼容 page_number/pageNumber。
   - section = evidence.title。
   - 不要使用未定义的 record_source_map。

6. app/services/handlers/llm.py
   - evidence prompt 格式注入 [n] 和 Cite as: [n]。
   - classic RAG document fallback 也按 1-based [n] 注入。
   - _llm_answer() 在最终 ctx.llm_response 后调用 build_citations(answer, evidence_items, documents, registry=self.registry)。
   - 保存 ctx.metadata["citations"] = [c.model_dump(mode="json") ...]。
   - 若有 emitter 且 citations 非空，emit_citations 后再 emit_step_completed("llm:answer", ...)。

7. 新增 app/services/citation_extractor.py
   - extract_claims(answer): 用正则 r'[^。！？\n]+?(?:[。！？\n]+|\.(?=\s|$)|$)'，支持中英文句子，不误切 15.2。
   - extract_quotes_with_llm(): registry.create_agent("fast", output_type=QuoteExtractionResult, instructions=...)；异常时返回 {}。
   - validate_and_locate_quotes(): exact substring -> relaxed whitespace -> punctuation-normalized projection -> split multi-sentence。LLM passage 找不到就丢弃。
   - relaxed whitespace 必须使用：
     pattern_str = re.sub(r'(?:\\\s)+', lambda _: r'\s+', re.escape(passage))
   - 不要把 relaxed whitespace pattern 改成 r'(?:\\\\\s)+'；该 pattern 匹配两个 literal backslashes + whitespace，会无法匹配 re.escape() 产生的 "\ " 序列。
   - fallback_locate_supporting_windows(): 数字/时间/单位/关键词/fuzzy scorer，返回 top 2-3 非重叠 LocatedSpan。
   - py_index_to_utf16_offset() 和 span_to_utf16()，所有 API spans 返回 UTF-16 offsets。
   - build_citations(): 显式定义 evidence_map = {idx: ev.content for idx, ev in evidence_by_index.items()}，避免 NameError；按 citation index 排序。

删除/废弃：
- approved_answer_only / delayed compliance release 相关代码路径。
- _STREAMING_POLICY_APPROVED_ONLY、_should_buffer_answer、_should_release_after_review、answer_delta release 分支。
- 更新或删除相关 tests。

测试要求：
新增 citation_extractor 单元测试，覆盖：
- 直接引用 exact 定位。
- 改写引用 LLM 返回原文后定位。
- 推理引用 Q2/Q3 原始数据定位。
- LLM 返回 paraphrase 或不存在 passage -> fallback。
- fallback 数字/时间/单位窗口命中。
- 全部失败 -> unlocated + empty spans。
- UTF-16 emoji/surrogate pair offset 正确。
- relaxed whitespace regex 正确处理 re.escape("hello  world")。
- 数字结尾句子 "The total count is 10. Next sentence." 正确切分。
- QueryResponse citations top-level，metadata 不重复。
- SSE 支持 citations 事件。

验证：
运行 targeted tests：
uv run pytest tests/ -k "citation" -v
uv run pytest tests/unit/test_llm_answer_aggregation.py tests/unit/test_streaming_events.py -v

如果删除 approved flow 导致更多现有测试失败，请同步更新这些测试，使代码库语义变为普通 token streaming + citations 后处理。
```
