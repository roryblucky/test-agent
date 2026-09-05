# Agent-mode Synthesis output boundary: local code review

Date: 2026-09-03

## Question

What is the smallest Synthesis Agent output that can satisfy the accepted
claim/citation gates while integrating with the existing `/v2/query/stream`
contract? This review compares direct Markdown, a typed claim list, and a
generic payload envelope against the repository's current code and tests.

## Conclusion

Use a **concrete, internal typed claim list for the financial POC**, validate it,
then deterministically render it into the existing `answer: string` and
`citations: CitationReference[]` wire fields.

Do not add a generic envelope or public base class yet. They have no current
consumer. Do not expose the typed claim list on the wire.

The smallest useful POC model is approximately:

```python
class ResearchClaim(BaseModel):
    text: str
    evidence_ids: tuple[str, ...] = ()
    calculation_artifact_ids: tuple[str, ...] = ()


class FinancialResearchDraft(BaseModel):
    claims: tuple[ResearchClaim, ...]
```

The gate must require each publishable claim to have at least one eligible
support reference. Numerical claims that depend on a registered calculation
must also name the validated Calculation Artifact. The renderer owns stable
claim order, Markdown construction, citation numbering, and conversion of
eligible Evidence into the already-existing `CitationReference` records.

This is a corrected version of option A: retain the claim structure, but remove
the speculative `EvidenceBackedOutput` inheritance hierarchy and remove a free
`summary` field that could publish factual prose outside the claim gate.

## Existing boundary and actual consumers

The current model boundary is already partially typed. `AnswerOutput` is
`answer: str` plus structured citation requests; `AnswerResult` adds model usage,
and `BoundAnswerResult` replaces citation requests with bound
`CitationReference` values. Therefore "direct Markdown" is not the current
complete internal contract even though Markdown-like text is the public answer.
See [`app/langgraph_v2/answer.py`](../../app/langgraph_v2/answer.py#L26-L61).

Current deterministic consumers are:

| Value | Deterministic consumer | Current requirement |
|---|---|---|
| `answer: str` | token streaming and stream/final reconciliation | non-empty; streamed deltas must exactly concatenate to the final answer ([source](../../app/langgraph_v2/answer.py#L164-L201), [source](../../app/langgraph_v2/answer.py#L250-L287)) |
| `answer: str` | post-moderation and groundedness | passed as opaque text; neither component parses a typed report ([groundedness](../../app/langgraph_v2/groundedness.py#L120-L139), [post-moderation](../../app/langgraph_v2/post_moderation.py#L45-L85)) |
| `answer: str` | finalization and conversation history | becomes `LinearQueryResponse.answer` and the persisted assistant message ([finalization](../../app/langgraph_v2/finalization.py#L86-L118), [graph](../../app/langgraph_v2/graph.py#L400-L418)) |
| Evidence reference | citation binding | current code accepts an ordered 1-based index and optional exact quote, then produces `CitationReference` ([source](../../app/langgraph_v2/answer.py#L64-L96)) |
| inline `[n]` | citation extraction | recognized indices are mapped to ranked Evidence; out-of-range or duplicate indices are silently ignored ([source](../../app/langgraph_v2/answer.py#L99-L131)) |
| `citations` | groundedness Evidence selection | only Evidence named by bound citations is sent to the advisory evaluator ([source](../../app/langgraph_v2/groundedness.py#L120-L135)) |
| `answer` and `citations` | public response | `answer` must remain `string|null`; `citations` must remain an array ([contract model](../../app/langgraph_v2/contracts.py#L51-L64), [UAT fixture](../../tests/fixtures/langgraph_v2/v2_uat_contract.json)) |

There is currently **no code consumer** for `ResearchClaim`,
`calculation_artifact_ids`, a domain report payload, or a generic
`SynthesisEnvelope`. Calculation Artifact is only a documented concept at this
point; the accepted POC spec requires its implementation, but the application
code contains no model or gate for it yet.

## Option review

### 1. Direct Markdown

Direct Markdown is the least-change integration because the public response and
conversation history already consume a string. It is reasonable only if the
POC defers claim-level deterministic publication gates.

It does **not** satisfy the current accepted spec by itself:

- The inline citation parser proves only that recognized `[n]` values refer to
  an Evidence position. It silently drops malformed or unknown references, as
  the integration test explicitly expects
  ([test](../../tests/integration/test_langgraph_v2_answer_phase.py#L510-L533)).
- It cannot deterministically establish that every publishable factual claim
  has support. Deciding which prose fragments are factual would itself require
  another semantic model or a more constrained authored structure.
- It has nowhere to attach Calculation Artifact identity to a numerical claim.

Verdict: **valid simpler alternative only if the claim/calculation publication
gate is explicitly removed from the POC**. It is not equivalent to the currently
accepted requirements.

### 2. Typed claim list

A typed claim list creates the only current justification for structured output:
the planned gate can consume each claim's support references before any public
answer is emitted. A deterministic renderer can then project the validated
internal draft into the unchanged wire contract.

It should remain concrete and narrow:

- no public base type;
- no payload registry;
- no inheritance hierarchy;
- no independent free-text `summary` outside the gated claims;
- no schema on the public `done` event.

Verdict: **recommended under the accepted spec**. It adds only the structure
required by a named deterministic consumer: the publication gate.

### 3. Generic envelope

`SynthesisEnvelope[T] { payload: T }` adds a wrapper but hides nothing. The gate
and renderer still need to understand the concrete `T` to find claims and
references. If Core does not understand `T`, it cannot enforce the accepted
claim/citation contract. If it does understand `T`, the wrapper is redundant.

There is also only one POC synthesis output, so there is no second implementation
from which to infer a stable common abstraction.

Verdict: **reject for the POC**. Reconsider only when at least two real domain
outputs share a demonstrated runtime operation that can be expressed as a true
protocol or adapter.

## Wire-compatible projection

The Agent-mode boundary should be:

```text
Synthesis Agent
  -> FinancialResearchDraft (internal Pydantic output)
  -> deterministic reference/calculation gates
  -> deterministic Markdown + CitationReference[] renderer
  -> existing LinearQueryResponse / done event
```

`FinancialResearchDraft` is an internal validation artifact. It does not replace
`LinearQueryResponse`, and neither claim objects nor Calculation Artifact IDs
need to be added to `/v2/query/stream` for this POC. Calculation Artifact IDs can
remain internal gate inputs; Evidence IDs are projected through the existing
`CitationReference` sidecar.

The existing Agent runtime test is currently too weak to protect this boundary:
its fake Graph emits `done.data == {"answer": "agent answer"}` rather than the
full `LinearQueryResponse` fields
([test](../../tests/integration/test_langgraph_v2_runtime_mode.py#L27-L50)). The
shared UAT contract requires the complete done shape, so Agent-mode acceptance
tests should assert the same fixture as Linear mode rather than treating an
arbitrary object with an `answer` key as sufficient.

## Contradictions and gaps in the current Agent spec

1. **Generic envelope conflicts with the local consumer boundary.** The spec
   requires a generic typed envelope whose Core does not inspect, while the same
   section requires Core to enforce claim/citation and calculation integrity
   ([spec](../../.scratch/langgraph-agent-patterns/spec.md#L230-L240)). Core must
   either inspect a stable claim protocol or receive a rendered-and-validated
   adapter result. A `payload: T` wrapper does not provide either behavior.

2. **The proposed `summary` bypasses claim coverage.** A domain report with a
   free factual summary plus a gated claim list can publish unsupported facts in
   the summary. Either all publishable prose is rendered from gated claims, or
   every independently authored prose field needs the same reference contract.

3. **The spec's buffered-publication rule differs from current Linear mode.**
   The spec says not to stream answer text until deterministic gates pass
   ([spec](../../.scratch/langgraph-agent-patterns/spec.md#L237-L240)), but the
   current answer path emits token events while the model is still producing
   `AnswerOutput`, before citation binding, groundedness, and post-moderation
   ([source](../../app/langgraph_v2/answer.py#L250-L290)). Agent mode can buffer
   and then emit tokens without changing event types, but this is a deliberate
   behavioral difference and must have an Agent-mode test.

4. **Current groundedness cannot serve as a publication gate.** Tests require a
   low groundedness score to remain advisory and preserve the answer, token
   stream, and terminal response
   ([test](../../tests/integration/test_langgraph_v2_groundedness.py#L105-L142),
   [test](../../tests/integration/test_langgraph_v2_groundedness.py#L157-L180)).
   The new deterministic reference gate must therefore be a separate phase.

5. **Calculation reference visibility remains undecided, but need not block the
   internal model.** The wire `CitationReference` has an `evidence_id` but no
   Calculation Artifact ID. The POC can keep calculation validation internal and
   cite the input Evidence. If clients later need reproducibility metadata, that
   requires a separate additive wire-contract decision rather than a generic
   synthesis envelope.

## Recommended corrected choice

Replace the original A/B/C choice with:

- **D (recommended):** concrete `FinancialResearchDraft(claims=...)` internally;
  validate references; deterministically render to the existing answer and
  citation wire contract. No base class and no generic envelope.
- **C' (simpler but changes scope):** keep the existing answer-style
  `answer + citations` contract and explicitly defer claim-coverage and
  Calculation Artifact gates.

The decision is therefore not "typed vs Markdown" in the public API. The public
API remains Markdown-like text either way. The real decision is whether the POC
keeps the already-accepted deterministic claim/calculation gate; if it does, the
minimal justified internal structure is the concrete claim list.
