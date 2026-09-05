# Final Synthesis output contracts in open-source research agents

Date: 2026-09-03

## Question

Review the proposed final Synthesis output choices without assuming that any of
them is correct:

- **A:** a claim-level typed base (`claims[]`) plus a financial-report subtype;
- **B:** a generic `Envelope[T]` around an arbitrary domain payload; and
- **C:** Markdown followed by citation extraction.

The target is this repository's read-only, evidence-backed information
platform. The result should remain coherent for a human reader, while code must
be able to reject invalid Evidence and Calculation Artifact references.

## Decisive conclusion

The original choices omit the best POC option. Current primary implementations
favor a **coherent prose report containing inline citations**, sometimes inside
a small concrete structured result. They do not commonly make a list of atomic
claims the canonical final report, and none of the inspected systems obtains
useful runtime polymorphism by wrapping an unknown payload in a generic
envelope.

For this repository, use a concrete financial Synthesis output whose report is
coherent Markdown and whose inline markers refer to the bounded support catalog
provided to the Synthesis Agent:

```python
class FinancialResearchReport(BaseModel):
    markdown_report: str = Field(min_length=1)
```

The Graph then parses the inline markers, rejects malformed, unknown,
ineligible, or cross-Run references, binds valid references to the existing
`CitationReference` representation, and only then publishes the Markdown. The
model should not separately repeat the same citation set in a second global
array; that creates two competing sources of truth.

This is a missing **D: concrete typed prose report + inline support markers +
deterministic binding**. It is close to C only if “extraction” means a strict
parser resolving markers already written next to claims. If C means asking a
later LLM to infer or attach sources after writing, reject C.

Do not introduce a common base class, generic payload envelope, output registry,
or domain adapter in the POC. When a second real domain requires additional
machine-consumed fields, define that domain's concrete Pydantic output and then
extract only the common publication interface actually shared by both domains.

## What current primary implementations return

### OpenAI Agents SDK `research_bot`

The Writer returns a concrete Pydantic `ReportData` with `short_summary`,
`markdown_report`, and `follow_up_questions`. The canonical long-form body is a
single Markdown string, not `claims[]`. The manager passes the original query
and collected search summaries to this Writer
([Writer source](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/agents/writer_agent.py),
[manager source](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py)).

This example demonstrates that a structured outer result and coherent Markdown
are compatible. Its prompt does not require citations, so it is evidence about
output shape, not about citation safety.

### OpenAI Agents SDK `financial_research_agent`

The current financial example also returns one concrete
`FinancialReportData(short_summary, markdown_report, follow_up_questions)`.
Its Writer prompt requires inline Markdown URLs for material numeric or
time-sensitive claims; it does not return a canonical claim array
([Writer source](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/agents/writer_agent.py)).

The manager separately stores retrieved evidence with its URLs, passes that
evidence to the Writer, and passes `allowed_source_urls` to a verifier. The
verifier returns typed issues and may trigger one full-report revision
([manager source](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/manager.py),
[verifier source](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/agents/verifier_agent.py)).

Important limitation: URL membership, support, consistency, and freshness are
all judged by the verifier **Agent** in this example. Prompting it to compare URL
strings exactly does not make that comparison deterministic application code.
The example therefore supports coherent prose plus a verification pass, but it
does not establish a deterministic claim-coverage guarantee.

### LangChain Open Deep Research

Open Deep Research stores `final_report: str`. Its final Writer joins compressed
research notes, asks the model for a report with inline Markdown links, and
writes the model's text content directly into `final_report`
([state definitions](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/state.py),
[final report generation](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py),
[final report prompt](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/prompts.py)).

It does not expose a claim array or a generic domain payload. Citation
preservation is prompt-based; the inspected finalization path does not perform a
deterministic source-membership or entailment check.

### STORM

STORM keeps generated article prose and its source registry separate. Generated
sections contain numeric inline markers. Application code parses those markers,
removes out-of-range references, retains cited source records, deduplicates by
URL, and rewrites section-local citation numbers into stable article-wide
indices
([article data and citation binding](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/modules/storm_dataclass.py),
[citation parsing utilities](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/utils.py)).

This is the strongest inspected example of deterministic **referential
integrity** around coherent prose. It still cannot prove that the source entails
the adjacent sentence. Its code silently deletes some invalid references; this
platform should fail the publication gate instead because silent deletion can
leave an unsupported sentence looking valid.

### GPT Researcher

GPT Researcher's recursive deep-research example first models intermediate
learnings as `{insight, sourceUrl}`, then serializes those learnings and URLs
into context for a final Writer. The Writer ultimately returns a report string,
not the learning array
([deep-research implementation](https://github.com/assafelovic/gpt-researcher/blob/main/backend/report_type/deep_research/example.py)).

This is useful evidence for claim-like **intermediate findings**, but not for
making claims the final user-facing contract. A reported empty-context failure
also shows why the Graph should skip Synthesis when no eligible Evidence exists
instead of letting a Writer improvise sources
([GPT Researcher issue #1572](https://github.com/assafelovic/gpt-researcher/issues/1572)).

### PydanticAI

PydanticAI supports both plain text and concrete Pydantic output types. Its
structured output mechanism validates the declared shape, while output
validators or output functions can apply application validation and request a
model retry. It does not prescribe a generic envelope or a research claim
schema
([PydanticAI output documentation](https://pydantic.dev/docs/ai/core-concepts/output/)).

Therefore, PydanticAI supports either a small concrete report model or validated
text. It does not provide a reason to add `Envelope[T]`, and Pydantic shape
validation alone cannot establish citation entailment.

## Critical review of the original options

### A — Claim-level typed base plus financial subtype

**Reject as the POC's canonical final-report shape.**

Claim-level items make these properties mechanically checkable:

- every declared claim has at least one reference;
- every reference resolves to an allowed Evidence or Calculation Artifact; and
- IDs are well formed, unique where required, and authorized for the Run.

But the proposed `claims[]` base does not define the actual long-form report.
There are only two ways to add one, and both introduce a cost:

1. **Generate prose and claims separately.** The same facts exist twice. Code
   cannot deterministically prove that the prose contains only the declared
   claims or that both versions agree.
2. **Render prose deterministically from claims.** Referential coverage becomes
   strong, but narrative ordering, qualifications, comparisons, and cross-claim
   reasoning are reduced to whatever the renderer can express. This is a
   different, machine-first product contract.

The inspected research Writers make prose/sections canonical and keep source
records beside them. Claim arrays are more defensible for intermediate findings,
review queues, or downstream APIs than for the final report itself.

If the product later needs a machine-consumable claim feed, add it as a distinct
artifact with an explicit consumer; do not duplicate it inside the POC report
preemptively.

### B — Generic `Envelope[T]`

**Reject.**

`payload: T` adds no invariant. Pydantic's generic typing can preserve the known
concrete type inside Python, but runtime Core code still cannot validate or
publish an arbitrary future `T` unless it has a real protocol, discriminator,
or registered adapter. If Core already knows `FinancialResearchReport`, the
envelope is only another nesting level.

A transport envelope can become useful when a stable external API genuinely
needs fields such as `kind`, `schema_version`, and a documented payload union.
That is a versioned API decision, not the Synthesis Agent's output contract, and
the current `/v2/query/stream` contract already exposes answer and citations.

### C — Markdown followed by citation extraction

**Split this ambiguous option.**

- **Accept strict parsing/binding:** the Writer emits inline markers at the time
  it writes each statement; deterministic code parses them and binds them to the
  fixed support catalog.
- **Reject post-hoc LLM attribution:** another model reads uncited prose and
  guesses which Evidence supports it. That is probabilistic citation recovery,
  not deterministic validation.

The repository already has the first pattern in
`app/langgraph_v2/answer.py` and `app/services/citation_extractor.py`: numeric
inline markers are mapped to request-local Evidence, and exact quoted passages
can be located. The new pattern should reuse that vocabulary and harden invalid
references into gate failures rather than inventing a parallel claim protocol.

### D — Concrete typed prose report with inline support markers

**Recommend for the POC.**

The model-facing contract is one concrete domain result. Its Markdown field is
the canonical narrative. Inline references are the canonical claim-to-support
links. Platform-produced `CitationReference` values are derived artifacts, not
a second model-authored source list.

If the financial fixture genuinely has a separately consumed executive summary,
add `short_summary` to this concrete type. Otherwise put the executive summary
inside the Markdown and avoid duplicate factual text. Do not add speculative
future-domain fields.

## What the publication gate can honestly guarantee

Given inline markers and a fixed ordered support catalog, deterministic code can
guarantee:

- marker syntax is valid;
- every referenced Evidence or Calculation Artifact exists;
- each reference belongs to the current Tenant and Run;
- each reference was eligible for this Synthesis invocation;
- Evidence freshness and required provenance fields satisfy code-owned policy;
- Calculation Artifact identity, method version, inputs, units, and execution
  record satisfy the registered calculation contract; and
- the published citation list is a deterministic projection of markers present
  in the report.

It cannot generally guarantee:

- that every factual sentence has a citation, because deciding what counts as a
  factual claim is semantic;
- that cited Evidence entails the adjacent statement;
- that the Writer preserved every material qualifier; or
- that the research is globally complete.

Even a model-authored `claims[]` list does not solve the first limitation when a
separate prose field exists: the model may put an undeclared claim in the prose.
Accordingly, the spec should not call universal factual-claim coverage a
deterministic gate unless the final answer is rendered exclusively from the
validated claim objects. For a prose-first POC, use deterministic reference
integrity plus an advisory groundedness/coverage evaluation.

Numeric claims are a narrower case. If the product requires deterministic
calculation provenance, require an inline Calculation Artifact marker next to
the numeric result and validate that marker mechanically. This does not require
turning every prose sentence into a `ResearchClaim` object.

## Recommended decision to present

Choose **D**, not the original A/B/C as written:

```text
Synthesis Agent
  -> concrete FinancialResearchReport(markdown_report)
  -> strict parse of inline Evidence / Calculation markers
  -> deterministic eligibility and referential-integrity gates
  -> derive public CitationReference records
  -> publish Markdown
```

This design proves the core E2E without a generic framework. It follows the
dominant prose-first pattern, retains the strongest STORM-like deterministic
citation hygiene, and leaves a clean seam for a future domain to introduce a
different concrete output only after it has a real consumer.
