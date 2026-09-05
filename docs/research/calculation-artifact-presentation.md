# How open-source research agents present deterministic calculations

Date: 2026-09-03

## Question

Review the proposed ways to present a validated Calculation Artifact to the
user:

- **A:** turn a `[C1]` marker into a calculation footnote containing method,
  version, inputs, as-of time, and unit, while keeping `citations[]` limited to
  Evidence;
- **B:** validate the artifact internally but show no calculation explanation;
- **C:** add `calculation_artifacts[]` to the public response;
- **D:** represent a Calculation Artifact as an ordinary citation.

This review separates observed open-source practice from a recommendation for
this repository. A calculation's provenance is not the same thing as a web
source citation, and the reviewed projects do not establish one universal
contract.

## Conclusion

The common open-source research-agent pattern is much simpler than A: the final
report shows calculated values in prose or tables and may mention the method,
key assumptions, time window, or source nearby. Detailed execution metadata is
usually omitted from the public report. General research agents such as Open
Deep Research and STORM do not model a first-class deterministic Calculation
Artifact at all, so they cannot establish a community standard for exposing
one.

The original options omit the best POC choice:

> **E — concise calculation disclosure in the report, detailed artifact kept
> internal.** A material calculated number refers to a validated Calculation
> Artifact, but the user-facing rendering includes only the dimensions needed
> to interpret that number: a human-readable method label, unit, relevant
> period/as-of date, and material assumptions when applicable. Method version,
> the complete input manifest, hashes, and execution details remain in the
> internal artifact/audit record. Source Evidence is still cited through the
> existing citation mechanism.

For example:

```markdown
Annualized volatility was 18.2% [C1][E2].

[C1] Annualized standard deviation of daily returns, 2025-01-01 to
2025-12-31, 252 trading days/year.
```

`[C1]` is not an Evidence citation. It is a temporary support marker that lets
the publication gate resolve and validate the calculation before rendering the
short disclosure. `[E2]` identifies the eligible price-series Evidence and is
what contributes to the existing public `citations[]` response.

This is a narrower version of A. It preserves interpretability without forcing
every report to dump method versions and all raw inputs, and it avoids changing
the public API before a machine consumer for Calculation Artifacts exists.

## What established research-agent examples do

### OpenAI Agents SDK `financial_research_agent`

The current financial example's final Pydantic type contains a short summary,
a Markdown report, and follow-up questions. Its Writer prompt requires inline
URL citations for material numeric or time-sensitive claims
([Writer source](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/agents/writer_agent.py)).

The manager stores search Evidence as summary, URLs, and retrieval date, then
passes that Evidence to the Writer and a verifier. Specialist financial and
risk agents are exposed to the Writer as tools, but their structured result is
projected down to summary text. The example does not construct or expose a
typed Calculation Artifact, input manifest, method version, or calculation
footnote
([manager source](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/manager.py)).

The Writer is streamed internally, but the manager consumes all stream events,
obtains the final typed result, verifies it, possibly revises it, and only then
prints the full report. This is relevant to publication timing but does not add
calculation provenance.

**Observed practice:** calculated or financial claims live in Markdown and use
ordinary source URLs. Calculation provenance beyond those sources is omitted.

### STORM

STORM is a cited long-form research system rather than a quantitative finance
system. It stores article prose and a source registry separately. Application
code parses numeric citation markers, removes out-of-range markers, deduplicates
source URLs, and rewrites section-local citation numbers to article-wide
indices
([citation processing](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/utils.py),
[article/source data](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/modules/storm_dataclass.py)).
Its demo turns citation numbers into links and presents the source title, URL,
and snippets in a reference panel
([demo rendering](https://github.com/stanford-oval/storm/blob/main/frontend/demo_light/demo_util.py)).

STORM has no separate calculation-reference namespace or structured
calculation-artifact response. Reusing it as evidence for option D would be an
invalid extrapolation: its markers resolve to retrieved sources, not derived
numerical computations.

**Observed practice:** deterministic binding for source references, but no
first-class calculation provenance.

### LangChain Open Deep Research

Open Deep Research asks the final Writer for a long-form report with inline
source citations and stores the result as one report string
([final Writer prompt](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/prompts.py),
[finalization path](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py),
[state definition](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/state.py)).
The inspected final-report contract contains no Calculation Artifact or public
calculation metadata.

**Observed practice:** source-cited prose; calculation lineage is outside the
system's modeled scope.

## What directly relevant financial examples do

The following smaller finance-specific repositories are illustrative rather
than evidence of an industry-wide standard. They are useful because they
actually perform deterministic calculations.

### Sugra Research Agent

Sugra tags each synthesis input block with its API endpoint and request
parameters and instructs the Writer to cite the originating endpoint inline
([input assembly and synthesis prompt](https://github.com/Sugra-Systems/sugra-research-agent/blob/main/sugra_research_agent.py)).
Its sample report presents a DCF value in prose together with the base free cash
flow, growth rate, net debt, discount rate, terminal-growth assumption, and a
sensitivity range. It also names the source endpoint next to the result. It
does not show a method version or complete raw input manifest and does not
publish a separate calculation-artifact array
([sample report](https://github.com/Sugra-Systems/sugra-research-agent/blob/main/examples/sample_report.md)).

This is the clearest inspected example of the missing option E: expose the
assumptions a reader needs to interpret a model-derived value, not every piece
of internal execution metadata.

### `equity-research-agent`

This example delegates financial calculations to deterministic Python and
reserves the report agent for narrative synthesis. Its sample output presents
ratios and DCF values in Markdown tables and adds a general note that the
calculations came from deterministic Python tools. The README documents its
calculator modules and tests, but the sample user report does not include
per-result method versions or an artifact array
([project README and sample output](https://github.com/palak22291/equity-research-agent)).

**Observed practice:** deterministic implementation and human-readable tables,
with auditability documented at the project level rather than encoded as a
public per-calculation object.

## Critical review of A-D

### A — exhaustive generated footnote

**Too broad as written; retain only its useful core.**

No reviewed system routinely puts method version, every input, as-of, and unit
into a footnote for every calculated value. Some of those fields are essential
for internal reproducibility, but a complete input manifest can be large and
can make a report harder to read. It may also reveal internal identifiers or
payload details that are not part of the publication contract.

Keep the marker validation and human-readable disclosure, but render only the
material method, unit, period, and assumptions. Keep the full artifact in the
Run's audit state.

### B — internal validation only

**Matches much open-source output, but is too opaque for a material financial
calculation.**

It is reasonable for trivial arithmetic whose operands are visible. It is not
enough for volatility, Sharpe ratio, drawdown, DCF, or other results whose
meaning changes with time window, frequency, annualization convention,
benchmark, risk-free rate, or model assumptions. A general statement such as
"calculated by deterministic Python" proves neither what was calculated nor
how the reader should interpret it.

### C — public `calculation_artifacts[]`

**Defer until a machine consumer exists.**

A structured array is useful if a UI needs expandable calculation details, an
export workflow needs reproducibility data, or another external client consumes
the artifact. The current POC has no such confirmed consumer. Adding the array
now would version the wire contract and expose internal fields before their
stability is known.

The internal Calculation Artifact should remain typed even though it is not yet
a public API object.

### D — disguise a Calculation Artifact as `CitationReference`

**Reject.**

Evidence answers “where did the input fact come from?” A Calculation Artifact
answers “which deterministic transformation produced this derived value?” A
calculation can depend on one or more Evidence items, so collapsing both into a
single undifferentiated citation loses that relationship and misstates what the
reference represents.

If a future API needs both, introduce an explicit discriminated support union
or a separate artifact resource. Do not make an internal computation pretend to
be a source document.

## Recommended POC boundary

1. Keep a typed, internal Calculation Artifact with the full reproducibility
   contract already required by ADR-0004.
2. Let the Synthesis Agent refer only to an allowlisted artifact through a
   temporary marker such as `[C1]` and to source Evidence through `[E2]`.
3. Have the deterministic publication gate resolve both markers and reject
   unknown, malformed, ineligible, or cross-Run references.
4. Render `[C1]` as a concise calculation disclosure containing the
   human-readable method, unit, relevant period/as-of, and material assumptions.
   Do not render every raw input or the internal method version by default.
5. Derive the existing `citations[]` only from Evidence markers. Keep
   Calculation Artifacts out of `CitationReference`.
6. Do not add `calculation_artifacts[]` to `/v2/query/stream` until a concrete
   external consumer requires machine-readable calculation details.

This boundary is not claimed as a community standard. It combines the common
prose/table presentation used by open-source research agents with the stronger
internal reproducibility contract this repository has already chosen.
