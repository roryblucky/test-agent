# Calculation Artifact handoff from Specialist Tools to Synthesis

Date: 2026-09-03

## Question

When a deterministic Tool called inside a Specialist produces a typed
`CalculationArtifact`, how should that artifact become available to the final
Synthesis Agent?

The options under review were:

- **A:** the Graph/runtime automatically collects every validated artifact from
  every successful Task;
- **B:** `SpecialistResult` contains `calculation_artifact_ids` and explicitly
  selects the artifacts to hand off;
- **C:** only put calculated values in the Specialist summary.

The repository has already fixed the POC `SpecialistResult` to
`summary + evidence_ids`, so B would reopen a confirmed contract. This review
therefore also considers whether the same selection semantics can be preserved
without adding another field.

## Conclusion

There is no established open-source contract for a first-class
`CalculationArtifact`. The reviewed systems mostly pass a worker's final
summary or a deliberately projected result to the writer; they do not expose
all internal Tool outputs. Frameworks provide the plumbing for richer handoff,
but leave selection and provenance to the application.

The best fit is a narrowed interpretation of A:

> **A' — run-local metadata followed by success-gated state contribution.**
> A calculation Tool gives the Specialist a concise model-visible result in
> `ToolReturn.return_value` while attaching the typed Artifact as
> application-only `ToolReturn.metadata`. Only after the Specialist returns a
> validated terminal Result does `execute_specialist` contribute that run's
> Artifacts and `TaskSucceeded` in one LangGraph state delta. `TaskFailed`,
> cancellation, exceptions, and abandoned outer retry attempts contribute none.

A' keeps the confirmed two-field `SpecialistResult` and does not ask the LLM to
authorize an artifact. The adapter reads typed application metadata from the
accepted PydanticAI run rather than introducing a collector or Artifact store.
Because the
POC exposes only a small fixed calculation set under hard-coded Tool-call
limits, every validated artifact from an accepted Task may enter the bounded
Synthesis catalog. If autonomous Specialists later perform enough exploratory
calculations to create real catalog pollution, explicit Specialist selection
can then be added as a measured extension.

Conceptually:

```text
Calculation Tool
  -> model-visible concise value
  -> application-only typed ToolReturn metadata

Specialist final output
  -> summary
  -> evidence_ids

execute_specialist adapter
  -> if terminal Result validates: one state delta contributes TaskSucceeded + artifacts
  -> otherwise: discard the run result and contribute no artifacts

Graph fan-in
  -> merge contributed artifacts by stable ID
  -> deterministically order and bound Synthesis catalog
```

This recommendation preserves stronger calculation provenance than common
research-agent examples. It is not presented as an industry standard.

## Evidence from primary sources

### LangGraph: workers write explicit outputs; reducers do not choose relevance

LangGraph's official orchestrator-worker example gives each worker private
state and has every worker write its output to a shared `completed_sections`
state key. An `operator.add` reducer performs the fan-in, and the writer joins
the collected worker results
([orchestrator-worker example](https://docs.langchain.com/oss/python/langgraph/workflows-agents#orchestrator-worker)).
This demonstrates that A is mechanically natural in LangGraph, but the example
collects each worker's declared output, not every intermediate Tool result from
inside the worker.

The Graph API documentation makes two constraints explicit:

1. parallel branches updating one state key need a reducer; and
2. update order from a parallel superstep is not guaranteed, so applications
   that require deterministic order must attach an ordering value and sort the
   results themselves
   ([parallel execution](https://docs.langchain.com/oss/python/langgraph/use-graph-api#run-graph-nodes-in-parallel)).

LangGraph's Pregel documentation additionally requires bulk reducers to be
associative, warns against assigning IDs inside a reducer, and says stable
identity must be attached before the write
([LangGraph runtime reducers](https://docs.langchain.com/oss/python/langgraph/pregel#overview)).

Therefore LangGraph supports a stable-ID artifact map, but the application must
still define validation, success gating, limits, and deterministic projection.

The same Graph API documents the failure boundary: a parallel superstep is
transactional when a branch raises; successful branch updates are not applied
when the superstep fails. It recommends catching expected errors inside nodes
when the workflow needs branch-local failure handling
([parallel exception handling](https://docs.langchain.com/oss/python/langgraph/use-graph-api#run-graph-nodes-in-parallel)).
For this POC, that means the single `execute_specialist` node should stage
artifacts locally and return either the successful outcome plus an atomic
artifact update, or `TaskFailed` without artifacts. It should not stream
artifact writes into parent state before its terminal outcome is known.

### Open Deep Research: Tool transcripts are compressed at the worker boundary

Open Deep Research runs researcher subgraphs concurrently, but each researcher
first compresses its accumulated Tool and AI messages into
`compressed_research`. The supervisor receives that compressed text as the
`ConductResearch` Tool result
([researcher execution and handoff](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L2105-L2137),
[compression](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L2447-L2525)).

When research ends, the supervisor derives `notes` from the research Tool
calls, and final report generation concatenates those notes into `findings`
for the Writer
([note collection](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L2040-L2054),
[Writer input](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L2589-L2659)).

The implementation also retains concatenated raw Tool/AI content as
`raw_notes`, but final report generation reads `notes`, not `raw_notes`. Thus
the model-facing handoff is an explicit compressed finding, while raw execution
material has a separate state path. It supports A's important boundary—only an
explicit worker output is promoted—but loses typed calculation provenance and
therefore is not sufficient as our contract.

### OpenAI Agents SDK: final output or explicit extraction, not automatic promotion

The official financial research example constructs a
`FinancialSearchEvidence` only after a search Agent succeeds. It explicitly
combines the Agent's structured final summary with source URLs extracted from
run items; exceptions and searches without sources return `None`, and only
non-`None` results enter the report input
([financial research manager](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/manager.py#L946-L1035)).
This is an explicit success-gated projection, not automatic forwarding of all
Tool outputs.

Its financial and risk Specialist Agents return a typed `AnalysisSummary`, but
the manager's `custom_output_extractor` deliberately projects that result down
to `summary` text before the Writer sees it
([summary extractor and Agent-as-Tool setup](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/manager.py#L825-L831),
[Writer setup](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/manager.py#L1037-L1067)).
The Writer then authors a typed Markdown report
([Writer contract](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/agents/writer_agent.py#L289-L347)).

The SDK exposes richer mechanisms when an application needs them. `RunResult`
contains both `final_output` and `new_items`; the latter includes Tool calls and
Tool outputs with Agent associations
([result surfaces](https://openai.github.io/openai-agents-python/results/#new-items)).
`Agent.as_tool()` also supports a `custom_output_extractor` that can inspect the
nested run and select or validate a particular Tool output rather than
forwarding everything
([custom output extraction](https://openai.github.io/openai-agents-python/tools/#custom-output-extraction)).
These primitives support application-owned extraction or B's explicit
projection, but do not automatically contribute internal Tool results to a
parent orchestration state.

### PydanticAI: a direct application-owned artifact side channel

PydanticAI Tools may return any JSON-serializable typed value. More
importantly, `ToolReturn` separates:

- `return_value`, which is serialized and sent back to the model;
- optional `content`, which becomes additional model-visible content; and
- `metadata`, which is available to application code but is not sent to the
  model. The official documentation notes that other frameworks call this
  metadata an “artifact”
  ([PydanticAI advanced Tool returns](https://ai.pydantic.dev/tools-advanced/#advanced-tool-returns)).

That separation maps directly to this design: a calculation Tool can give the
Specialist the concise value needed to reason, while preserving the full typed
`CalculationArtifact` on an application-owned path. PydanticAI
does not automatically copy that metadata into the Agent's final structured
output, so promotion to Synthesis remains an application contract.

PydanticAI also records terminal Tool failures distinctly as failed Tool return
parts, letting the Agent decide how to recover
([failed Tool result](https://ai.pydantic.dev/tools-advanced/#tool-failed)).
That is useful inside the Specialist run but is not a reason to publish partial
artifacts from a Specialist that ultimately produces `TaskFailed`.

## Assessment of the original options

### A — promote every artifact from every successful Task

**Mechanically simple, but too broad as the final Synthesis contract.**

Advantages, after narrowing it to A':

- fits LangGraph shared-state/reducer mechanics;
- does not change `SpecialistResult`;
- a success gate prevents failed-Task leakage;
- it preserves the already confirmed two-field result and is sufficient for the
  POC's small, bounded calculation surface.

Limits:

- a successful autonomous Specialist may perform exploratory, superseded, or
  mutually alternative calculations before writing its finding;
- “the Tool ran successfully” does not mean “this artifact supports the final
  Specialist finding”;
- passing every validated artifact becomes noisy if a future Specialist performs
  many exploratory calculations.

A' is recommended for the POC because hard-coded Tool-call and catalog limits
bound that risk. It should not be generalized to an unbounded autonomous
analysis environment. Limits must reject excess production rather than silently
truncate an artifact that may support a finding.

### B — add `calculation_artifact_ids` to `SpecialistResult`

**Semantically clean, but reopens a confirmed POC contract.**

This directly mirrors `evidence_ids`, gives deterministic relevance, and is
easy to validate. It is the clearest choice if calculation artifacts become a
stable machine-consumed part of every Specialist's output. For the present POC,
however, it adds a field after the result was deliberately fixed to
`summary + evidence_ids`, and the public API still has no artifact array.

### C — write numbers only in summary

**Matches many examples, but violates this repository's provenance decision.**

Open Deep Research and the OpenAI examples commonly pass summaries, but they
do not model reproducible Calculation Artifacts. Copying that limitation would
remove the deterministic identity needed for `[C1]` validation and make it
impossible to distinguish a registered calculation from a number authored by
the model.

## Recommended POC contract: A'

1. The Tool Executor resolves authoritative inputs from trusted instrument,
   dataset, or Evidence references; it does not trust an LLM-authored raw
   numerical series. It creates a stable artifact ID before returning the result.
   The ID must include or bind to Run and Task identity; reducers must not
   generate it.
2. The calculation Tool returns a concise value to the Specialist model, while
   the full typed artifact is attached as application-only PydanticAI
   `ToolReturn.metadata`.
3. The adapter extracts only typed, integrity-valid artifacts bound to the
   current Tenant, Run, Task, outer attempt, registered calculation, and Tool
   call from the accepted run's new messages.
4. After the Specialist's terminal output passes validation, the
   `execute_specialist` adapter returns `TaskSucceeded` and that run's Artifact
   map in one LangGraph state delta. This is a Graph state-update boundary, not
   an external distributed transaction.
5. A `TaskFailed` outcome contributes no artifacts, including artifacts created
   by earlier successful Tool calls during that failed run. Outer retry starts
   a fresh PydanticAI run; only the terminal accepted run may contribute.
6. Parallel branches return contributed artifacts keyed by stable ID. The reducer
   is associative and idempotent; the same ID with different content is an
   invariant failure.
7. Enforce hard per-Task artifact limits within each branch. Enforce the
   aggregate catalog limit deterministically after fan-in, or choose limits
   whose per-Task maxima cannot exceed it. Never silently truncate the committed
   catalog.
8. Before Synthesis, project the artifact map in canonical
   `(round, dispatch_order, task_id, artifact_id)` order. Never derive ordering
   from parallel Tool completion time. Apply both
   count and byte bounds to the model-visible concise representation while
   keeping complete artifacts in internal audit state.
9. Assign deterministic short aliases such as `[C1]` only in the bounded
   Synthesis catalog. Synthesis emits a Calculation placeholder rather than
   retyping its numerical value. The final publication gate resolves the alias,
   validates the Artifact, and renders its canonical formatted value. Synthesis
   may ignore catalog entries; it may not reference an artifact outside the
   catalog.

This gives the system all four desired properties:

- **bounded relevance:** only calculations from accepted, goal-scoped Tasks can
  enter a deliberately small catalog;
- **failure isolation:** only an accepted terminal Task can contribute artifacts;
- **determinism:** identity, validation, merge, ordering, and limits are
  code-owned;
- **contract stability:** `SpecialistResult` remains
  `summary + evidence_ids` and no generic payload framework is introduced.

## Tests implied by the decision

- A successful Specialist calls two valid calculations: both enter the
  Synthesis catalog in deterministic order; Synthesis may cite either or both.
- A Tool creates an artifact and the Specialist later fails: the artifact does
  not enter Graph state or Synthesis input.
- A transiently failed attempt creates an artifact and a later attempt
  succeeds: only artifacts from the accepted attempt may be contributed.
- Two parallel Tasks finish in opposite wall-clock orders: Synthesis receives
  the same canonical artifact order.
- An extracted artifact has an unknown calculation, invalid integrity data, or a
  foreign Tenant/Run/Task/attempt binding: Task acceptance fails closed.
- The artifact limit is exceeded: the Task is rejected rather than silently
  dropping a calculation.
- The Synthesis report cites only supplied `[C<n>]` aliases; malformed or
  unknown aliases fail the publication gate.
- The Synthesis report uses Calculation placeholders, and code renders the
  referenced Artifact's exact canonical value; a model-authored replacement
  value is rejected rather than trusted.
