# LangGraph Agent Patterns gap closure

Date: 2026-09-05
Scope: baseline POC/MVP only

## Executive verdict

The architecture direction is sound, but the current specification is not yet
implementation-closed on the four reviewed gaps. The safe minimum is smaller
than a coverage engine or a new repository: normalize expected Tool
unavailability inside each Tool binding, automatically materialize a small
self-contained provenance record in every accepted Data Gap, derive a
conservative code-owned incompleteness manifest from accepted outcomes, and
render its disclosure after Synthesis.

| Review item | Verdict on current text | Smallest closure |
|---|---|---|
| Expected Tool unavailability during Specialist multi-hop and internal fan-out/fan-in | **FAIL** (correct intent, incomplete executable rule) | Binding-owned timeout and allowlisted exception-to-`ToolReturn` conversion; never let an expected condition escape the Tool coroutine |
| Partial `TaskSucceeded`, `TaskFailed`, and accepted `DataGap` cannot look complete | **FAIL** | Conservative code-owned `IncompleteResearch` manifest plus code-rendered disclosure and `metadata.completion_status="incomplete"` |
| Data Gap provenance | **FAIL** | Adapter automatically materializes one canonical Data Gap for every unavailable Tool result in the accepted attempt; no provider/error payload in the business result |
| Bounds | **FAIL** | Keep the existing structural limits, but define round semantics and concrete actor, timeout, payload, and context caps in one table |

These are specification gaps, not evidence that the chosen LangGraph/PydanticAI
boundary is wrong. No additional persistence subsystem is needed.

## Material reviewed

The review read the complete local normative set:

- `.scratch/langgraph-agent-patterns/spec.md`
- `CONTEXT.md`
- ADR-0001 through ADR-0007
- the directly relevant `app/langgraph_v2` baseline, especially `api.py`,
  `contracts.py`, `graph.py`, `answer.py`, `retrieval.py`, `finalization.py`,
  `stream.py`, `conversation_context.py`, `checkpointing.py`,
  `linear_runtime.py`, `provider_adapters.py`, and `model_usage.py`

The baseline contains an Agent runtime factory seam but no Agent Graph,
Specialist, Task Outcome, or Data Gap implementation yet
(`app/langgraph_v2/api.py:226-330`). The public response already has open
`metadata` (`contracts.py:51-64`), and request streaming already runs with
synchronous durability and defers a checkpoint-terminal frame until a graph
update confirms it (`stream.py:94-149`). Those seams are sufficient for the
recommended closure.

The Linear answer phase is not reusable as the Agent publication path: it emits
answer tokens before the full answer and citations have validated
(`answer.py:250-304`). The Agent path should retain the specification's existing
progress-only Synthesis rule. Likewise, Linear retrieval's broad
`except Exception` and raw `str(exc)` projection (`retrieval.py:45-104`) must not
be copied into an Agent Tool binding, where the expected/fatal distinction is a
core contract.

## Primary-source facts used

Only official LangGraph and PydanticAI documentation/source is used for
framework claims below.

1. PydanticAI `ToolReturn.return_value` is sent to the model, while
   `ToolReturn.metadata` is application-visible and not sent to the LLM. This is
   the precise carrier required for a typed model-visible unavailable value plus
   an application-only provenance record.
   [PydanticAI: Advanced Tool Returns](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#advanced-tool-returns)
2. PydanticAI executes multiple Tool calls from one model response concurrently.
   Therefore each expected failure must be normalized inside its own Tool
   invocation before concurrent collection observes it. Any other exception
   propagates from the Agent run; run cancellation cancels and drains in-flight
   async Tool tasks.
   [PydanticAI: Tool failures, parallel calls, and cancellation](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-execution-retries-and-failures)
   A normal Tool result is returned to the same Agent loop, which may make the
   next model request and call a fallback; this is the direct same-run
   multi-hop mechanism.
3. PydanticAI's built-in Tool timeout is a retry mechanism: timeout produces a
   retry prompt and consumes Tool retry allowance. That is different from this
   POC's required typed unavailable value, so the expected-timeout conversion
   must be owned inside the registered binding rather than delegated to the
   framework timeout setting.
   [PydanticAI: Tool Timeout](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-timeout)
4. `ModelRetry` asks the model to correct and retry. `ToolFailed` creates a
   model-visible failed Tool result and does not consume Tool retry allowance;
   repeated failed results instead need run-level bounding. Neither provides the
   POC's typed success-path provenance contract, so neither is the chosen
   expected-unavailability carrier.
   [PydanticAI: Tool execution, retries, and failures](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-execution-retries-and-failures)
5. PydanticAI validates structured output against the configured output type.
   This validates `SpecialistResult` shape, but it does not validate application
   provenance or final research completeness; those remain deterministic
   application gates.
   [PydanticAI: Structured output data](https://pydantic.dev/docs/ai/core-concepts/output/#structured-output-data)
6. `UsageLimits` can bound model requests and successful Tool executions. The
   Tool-call limit is checked before a parallel set runs; if the set would exceed
   the remaining limit, none of that set executes. Returning a normal
   `ToolReturn` is, by inference, a completed Tool invocation and therefore fits
   this POC's Tool-call cap, while request count remains the backstop against
   repeated failed/rejected calls.
   [PydanticAI: Usage Limits](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits),
   [PydanticAI `UsageLimits` API](https://pydantic.dev/docs/ai/api/pydantic-ai/usage/#pydantic_ai.usage.UsageLimits)
7. LangGraph runs parallel nodes in one superstep; if any branch raises, the
   superstep errors and its state updates are not applied. Application code may
   catch expected failures inside a node. This directly rules out allowing an
   expected Tool/Task condition to escape a partial-success branch.
   [LangGraph: parallel execution and exception handling](https://docs.langchain.com/oss/python/langgraph/use-graph-api#run-graph-nodes-in-parallel)
8. LangGraph does not promise completion-order stability for parallel updates;
   deterministic ordering requires an explicit order key, and shared parallel
   writes require a reducer. This supports the existing stable-ID contribution
   map and barrier design.
   [LangGraph: parallel execution, ordering, reducers, and concurrency](https://docs.langchain.com/oss/python/langgraph/use-graph-api#run-graph-nodes-in-parallel)
9. LangGraph `Send` is the official dynamic map/fan-out primitive. It supplies
   execution mechanics, not a domain success/failure or completeness contract.
   [LangGraph Graph API: `Send`](https://docs.langchain.com/oss/python/langgraph/graph-api#send)
10. LangGraph's recursion limit is a maximum step guard that raises
    `GraphRecursionError`; it is not a graceful business-completion signal.
    [LangGraph: recursion limit](https://docs.langchain.com/oss/python/langgraph/use-graph-api#recursion-limit),
    [LangGraph `GRAPH_RECURSION_LIMIT`](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)
    The dependency-pinned implementation exposes the same fatal error contract.
    [LangGraph source: `GraphRecursionError`](https://github.com/langchain-ai/langgraph/blob/1.1.10/libs/langgraph/langgraph/errors.py#L45-L65)
11. PydanticAI officially supports `TestModel`, `FunctionModel`, model override,
    captured messages, and disabling non-test model requests. These are the
    appropriate seams for exact Tool-call trajectory tests.
    [PydanticAI: Unit testing](https://pydantic.dev/docs/ai/guides/testing/)

The framework sources do not define this product's Data Gap provenance or the
meaning of a complete financial Research Report. The recommendations below are
application invariants derived from those framework mechanics and the local
normative model; they are not presented as framework requirements.

## 1. Expected Tool unavailability

### Decision

Keep expected unavailability as a normal Tool return. Strengthen the text to
make the adapter boundary executable:

```python
ToolUnavailableCode = Literal[
    "source_unreachable",
    "coverage_not_supported",
    "stale_only",
    "call_timeout",
    "response_unusable",
]


class ToolUnavailable(BaseModel):
    kind: Literal["unavailable"] = "unavailable"
    code: ToolUnavailableCode
    requested_coverage: str


class ToolUnavailabilityRecord(BaseModel):
    # application-only ToolReturn.metadata
    unavailability_id: str
    task_id: str
    attempt: int
    tool_call_id: str | None
    tool_id: str
    source_id: str | None
    observed_at: datetime
    code: ToolUnavailableCode
    requested_coverage: str
```

The metadata-only `unavailability_id` is an opaque, application-generated
immutable identity for one normalized current-attempt unavailable outcome;
transparent replay is out of scope, so it need not be reproducible across
processes. The model sees only the stable allowlisted code and sanitized bounded
coverage label. The metadata binds those fields to the current Task/attempt and trusted registered
Tool/logical source at the Run's fixed observation time, so the adapter—not the
model—can construct the canonical Data Gap. It is request-local handoff data,
not a new repository.

An authoritative query that executes successfully and establishes that the
answer is empty or negative is a normal successful Tool result with Evidence,
not `ToolUnavailable`. The unavailable codes mean the requested coverage could
not be established; they must not be used to turn a valid negative finding into
a gap.

Every registered Tool binding must follow this order:

1. Resolve the already-fixed authorization and Research Scope before invoking
   the provider. Authorization/scope failures propagate.
2. Apply the binding's own per-call timeout around the Tool Executor call.
3. Catch only the named expected-unavailability result/exception classes and
   that binding-owned timeout.
4. Return `ToolReturn(return_value=ToolUnavailable(...),
   metadata=ToolUnavailabilityRecord(...))`.
5. Let cancellation, authorization, invariant, configuration, reducer,
   programmer, and unknown provider exceptions propagate unchanged.

Do not configure the framework's Tool timeout for these bindings; its official
contract creates a retry prompt rather than the required typed value. Do not
use `except Exception`. Do not raise `ModelRetry` or `ToolFailed` for an expected
POC unavailable outcome. The last choice is stricter than the framework
requires, but gives one provider-independent schema and one provenance path.

This is sufficient for both patterns:

- Multi-hop: the unavailable value returns to the same PydanticAI run; a later
  model request may call an allowed fallback and still return a valid
  `SpecialistResult`.
- Internal fan-out/fan-in: every concurrently scheduled Tool coroutine reaches
  an ordinary return, so one expected unavailable branch does not inject an
  exception into concurrent collection or cancel successful siblings. The
  terminal Specialist output may combine successes; the adapter still records
  every unavailable call as a canonical Data Gap.

### Spec patch text

Replace the current expected-unavailability paragraph with:

> Every registered Specialist Tool binding normalizes expected inability to
> provide requested read, fetch, or Calculation data before it can escape the
> Tool coroutine. The binding applies its own per-call timeout, catches only its
> explicit expected-unavailability allowlist plus that timeout, and returns
> `ToolReturn(return_value=ToolUnavailable,
> metadata=ToolUnavailabilityRecord)`. The model-visible value contains a
> bounded sanitized requested-coverage label and an allowlisted stable code.
> Application-only metadata assigns an opaque `unavailability_id` and binds it
> to the current Task, outer attempt, registered Tool, normalized code,
> logical source, fixed observation time, and coverage label. An authoritative
> successful empty/negative query is Evidence, not unavailability. After terminal output validates, the
> Specialist adapter automatically materializes exactly one canonical Data Gap
> from every such metadata record in the accepted attempt, even when a later
> fallback succeeds. These
> bindings do not use PydanticAI's Tool-timeout conversion, `ModelRetry`, or
> `ToolFailed` for expected unavailability. Cancellation, authorization,
> invariant, configuration, programmer, and unknown failures are not caught.
> Consequently expected unavailability is an ordinary completed Tool call in
> the same PydanticAI run: later model requests may call a fallback, and a
> parallel sibling Tool call may complete normally.

### Counterexample tests

| Test | Counterexample | Required assertion |
|---|---|---|
| `test_specialist_multihop_expected_unavailable_then_fallback` | Primary Tool produces an allowlisted unavailable outcome; fallback succeeds on the next model turn | One PydanticAI run returns `TaskSucceeded`; captured messages contain typed unavailable then success; adapter preserves the unavailable result as a canonical Data Gap; no retry prompt or second outer attempt |
| `test_parallel_tools_expected_unavailable_does_not_cancel_sibling` | One of two Tool calls in one model response hits the binding-owned timeout while the other completes after it | Both Tool results reach message history; successful Evidence remains; adapter adds exactly one canonical Data Gap; no sibling cancellation |
| `test_framework_tool_timeout_is_not_used_for_expected_unavailability` | A Tool is accidentally registered with PydanticAI timeout behavior | Contract test fails because captured trajectory contains a retry result rather than `ToolUnavailable` |
| `test_unexpected_tool_exception_is_fatal` | Tool raises `AssertionError` or an authorization exception | Exception leaves the Specialist and fails the Run; it is not `ToolUnavailable` or `TaskFailed` |
| `test_parallel_tool_return_order_is_canonicalized` | Reverse completion timing across two successful Tool calls | Accepted metadata/Evidence order is identical because it is sorted by stable Tool-call identity, never completion time |

## 2. Preventing a partial run from looking complete

### Decision

The current Synthesis input correctly excludes raw Task status, retry data, and
runtime limits. Keep that separation. Add one application-only value constructed
after the final accepted barrier:

```python
class IncompleteResearch(BaseModel):
    task_ids: tuple[str, ...] = ()
    data_gaps: tuple[DataGap, ...] = ()
    execution_limit_reached: bool = False
```

Construction is deterministic and conservative:

- include every accepted `TaskFailed.task_id`;
- include every adapter-materialized Data Gap in every accepted successful
  Specialist Result;
- set the flag for Task/Round/context structural exhaustion;
- stable-sort and deduplicate by Graph identity, not prose or completion order.

If all three are empty/false, no claim of global completeness is created; the
Run merely has no accepted incompleteness signal. If any is present:

1. Synthesis still receives only its existing bounded business input, including
   accepted Data Gaps, and no operational state.
2. After the Synthesis candidate passes support-marker and Calculation gates,
   code prepends a fixed `> Incomplete research:` disclosure listing sanitized
   Task objectives for failed Task IDs, accepted Data Gap descriptions, and/or
   the standard structural-limit statement.
3. Final response metadata must set `completion_status="incomplete"` and a
   stable `termination_reason`: `partial_results` when Task failure/Data Gap is
   present, `execution_limit` when that is the only cause, and
   `partial_results_and_execution_limit` when both are present.
4. The exact post-disclosure Markdown is the canonical answer checkpointed
   before publication. Synthesis cannot remove or soften the disclosure.
5. With no eligible Evidence, bypass Synthesis and build the deterministic
   insufficient-Evidence answer with the same disclosure inputs.

For the POC, incompleteness is monotonic: a Data Gap or `TaskFailed` in any
accepted batch keeps the final response incomplete even if later work appears
to cover the same topic. This may over-disclose, but it cannot hide known loss
and requires no global semantic coverage model. Clearing an earlier signal
would require a future domain-specific coverage contract and is not part of
this POC.

The disclosure is code-owned because structured model output validation proves
shape, not semantic completeness (primary-source fact 5). LangGraph supplies
the barrier mechanics but no Research Report completeness model (facts 7-9).

### Spec patch text

Insert after the `TaskOutcome` decision:

> After the final accepted barrier, deterministic code builds one immutable
> application-only `IncompleteResearch` value from all accepted outcomes. It
> contains stable failed Task IDs, all accepted Data Gaps, and whether a
> Task/Round/context structural limit ended research. Any non-empty value makes
> the final response incomplete. For this POC the rule is conservative and
> monotonic: later work does not clear an earlier accepted failure or gap because
> no domain coverage-equivalence contract exists.

Replace the incomplete-publication text with:

> Synthesis does not receive Task status, retries, technical diagnostics, or
> runtime limits. After the Synthesis candidate passes all blocking support and
> Calculation gates, deterministic code prepends the standard incomplete-
> research disclosure derived from `IncompleteResearch`, then checkpoints that
> exact Markdown as the canonical assistant answer. When incomplete, response
> metadata sets `completion_status` to `incomplete` and uses
> `partial_results`, `execution_limit`, or
> `partial_results_and_execution_limit` as the stable termination reason. If no
> eligible Evidence exists, code skips Synthesis and renders the same disclosure
> in the deterministic insufficient-Evidence answer. No absence of known gaps is
> represented as proof of global completeness.

### Counterexample tests

| Test | Counterexample | Required assertion |
|---|---|---|
| `test_task_failed_cannot_publish_complete_looking_report` | One Task fails; a successful sibling supplies Evidence; Synthesis says “all requested areas are covered” | Canonical answer begins the code-rendered incomplete disclosure and metadata is incomplete |
| `test_accepted_gap_is_disclosed_even_when_synthesis_omits_it` | Synthesis returns valid cited Markdown but never mentions the accepted Data Gap | Code inserts the exact sanitized gap description before checkpoint/publication |
| `test_gap_and_execution_limit_combine_reasons` | Accepted gap exists and the next dispatch is blocked by the round limit | One disclosure contains both causes; stable combined termination reason is emitted |
| `test_no_evidence_bypasses_synthesis_and_discloses_failures` | All accepted outcomes are `TaskFailed` | Synthesis call count is zero; deterministic insufficient-Evidence answer names bounded failed coverage |
| `test_no_known_incompleteness_does_not_claim_global_completeness` | All Tasks succeed with no gaps | No incomplete banner; output and metadata contain no statement that code proved global completeness |
| `test_disclosure_is_committed_before_first_answer_frame` | Synthesis succeeds but final checkpoint write is delayed or fails | No answer frame is emitted before commit; failure emits no answer/citations/`done` |

### Prepared Synthesis identity

Delete the proposed `PreparedSynthesis.digest`. The same frozen
`PreparedSynthesis` value and alias maps are passed unchanged to the one repair
invocation, so value equality and the rule “do not rebuild from live catalogs”
already prevent drift in the active process. Transparent active-Run recovery is
explicitly unsupported, and no external consumer verifies the digest. A digest
would therefore duplicate canonicalization without closing a current threat.
Keep the frozen value and fail if repair receives a different value.

Spec patch text:

> Build one frozen `PreparedSynthesis` value before the first Synthesis call. It
> owns the exact model input and deterministic Evidence and Calculation alias
> maps. The optional repair must receive that same value unchanged; passing a
> different value is an invariant failure. No digest is added because active-Run
> cross-process recovery and external verification are outside this POC.

Counterexample: `test_synthesis_repair_reuses_same_prepared_value` passes only
when the repair receives an equal frozen value and alias maps without rebuilding
them from current state; mutation or catalog drift fails.
`test_prepared_synthesis_has_no_unused_digest` asserts that
the POC contract contains no digest field.

## 3. Data Gap provenance

### Decision

Yes, provenance is required, but the model should not author it. Without an
adapter-owned link, a model can omit a real unavailable call or invent a gap
unrelated to one. A dangling opaque ID is not enough after its request-local
handoff record is discarded, so the minimum accepted contract is
self-contained:

```python
class GapProvenance(BaseModel):
    unavailability_id: str
    tool_id: str
    source_id: str | None
    observed_at: datetime


class DataGap(BaseModel):
    requested_coverage: str
    code: ToolUnavailableCode
    provenance: GapProvenance
```

After a Specialist's terminal structured output validates, the adapter scans
the accepted attempt's Tool return metadata and automatically creates exactly
one canonical Data Gap per `ToolUnavailabilityRecord`. Model-authored Data Gaps
are not an acceptance authority; if the output schema retains them for model
reasoning, they are replaced by the adapter's canonical set. This conservative
rule deliberately preserves a gap even if a later fallback succeeds. It may
overstate incompleteness, but it cannot silently publish known unavailability
as complete and requires no coverage-equivalence or resolution relation.

Acceptance rules:

- every canonical gap is created from exactly one `ToolUnavailabilityRecord`
  emitted by the same Run, Task, and accepted outer attempt;
- each unavailable ID appears exactly once; duplicate IDs with different
  content are an invariant failure;
- gaps are stably ordered by unavailable ID, never Tool completion time;
- coverage labels and IDs are bounded and sanitized;
- raw exception, provider payload, retry history, stack trace, Tool arguments,
  source credentials, and provider-specific error codes are forbidden;
- an unavailable Tool result creates no Evidence or Calculation metadata;
- a gap is allowed only on `TaskSucceeded`; `TaskFailed` has no Result and its
  missing coverage is derived from the already accepted Task objective.

The application-only handoff record may be discarded after the accepted batch
has validated. The accepted Data Gap itself retains only the trusted provenance
needed to interpret it: immutable observation identity, registered Tool,
logical source when applicable, and fixed observation time. It does not retain
Task-attempt diagnostics, Tool arguments, or provider error payload. This is a
small value embedded in the existing Specialist Result, not a separate Artifact,
event stream, or repository. It mirrors the already accepted Evidence-metadata
handoff at a much smaller scale and uses `ToolReturn` exactly as documented in
primary-source fact 1.

This contract verifies both directions needed by the POC: no accepted Data Gap
is fabricated, and no expected unavailable call in the accepted attempt is
silently dropped. It does not decide whether a later source is semantically
equivalent. Resolution is intentionally deferred.

Do not add a `coverage_key` in this POC. Such a key is useful only if code is
allowed to decide that later Evidence resolves an earlier unavailable
observation. The selected conservative rule never clears one, so there is
nothing to match. Adding the key now would introduce normalization and
equivalence semantics without changing an acceptance decision.

Coordinator, Synthesis, and public disclosure receive a deterministic
`DataGapView` containing only `requested_coverage`, `code`, and `observed_at`;
trusted internal `unavailability_id`/`tool_id`/`source_id` remain in checkpointed
canonical provenance for validation and audit, not model authority.

### Spec patch text

Replace the Data Gap shape paragraph with:

> After terminal Specialist output validates, the execution adapter automatically
> materializes exactly one canonical Data Gap for every
> `ToolUnavailabilityRecord` collected from the accepted outer attempt. A Data
> Gap contains the bounded sanitized requested-coverage label, stable code, and
> minimal `GapProvenance(unavailability_id, tool_id, source_id, observed_at)`
> copied from that record. The handoff record is bound to the same Run, Task,
> and attempt, but is not model-visible and is not a new persisted Artifact.
> Minimal provenance is embedded in the existing accepted Specialist Result;
> it has no separate repository. Model-authored gap content is not canonical.
> Data Gaps never carry
> raw exceptions, provider payloads, Tool arguments, retry history, stack
> traces, credentials, or provider-specific diagnostics. A missing, stale,
> cross-Task, cross-attempt, or duplicated-conflicting record rejects the
> attempt. Every expected unavailable call remains a canonical gap even after a
> successful fallback; this conservative POC rule may over-disclose and is
> replaced by explicit gap-resolution semantics only in a future design. Do not
> add a `coverage_key`: no current rule resolves or matches gaps. Coordinator,
> Synthesis, and public disclosure receive a bounded `DataGapView` that excludes
> internal Tool/source IDs.

### Counterexample tests

| Test | Counterexample | Required assertion |
|---|---|---|
| `test_adapter_materializes_every_unavailable_as_one_gap` | Specialist omits gaps after one unavailable call | Adapter adds exactly one canonical same-attempt gap before branch staging |
| `test_abandoned_attempt_unavailability_is_excluded` | Collected metadata comes from an abandoned outer attempt | Record is excluded; only accepted-attempt metadata can create canonical gaps |
| `test_unavailable_metadata_contains_no_evidence_or_artifact` | Expected Tool timeout occurs | `ToolReturn.metadata` has only bounded provenance fields; Evidence/Calculation catalogs remain unchanged |
| `test_gap_public_projection_hides_tool_diagnostics` | Internal record includes observation, registered Tool, and logical source identities | Coordinator/Synthesis/public views contain only sanitized coverage, code, and observation time, never internal IDs/provider details |
| `test_fallback_success_remains_conservatively_incomplete` | Primary unavailable, approved fallback supplies the requested data | Result succeeds but retains the primary unavailable gap; final answer is incomplete |

## 4. Bounds

### Decision

The structural direction passes, but the contract is not yet precise enough to
implement consistently. The text gives 32 total Tasks, eight-way fan-out,
“three follow-up rounds,” three outer attempts, and recursion limit 40. It does
not define whether a final `Finish` consumes a round, and repeatedly says
“bounded” for objectives, result/context sizes, Data Gaps, Tool output, Evidence,
and synthesis input without values. It also requires per-call timeouts and
actor-local limits without POC defaults.

Use this single code-owned POC policy. Limits are inclusive; a proposed action
that would make a counter exceed its limit is rejected before execution.

| Limit | POC value and exact semantics |
|---|---|
| Dispatch Rounds | `4` accepted `DispatchBatch` Decisions maximum: initial dispatch plus at most three follow-up dispatches |
| Coordinator Decisions | At most `5` accepted Decisions: up to four dispatches plus one final `Finish`; each accepted `DispatchBatch` or `Finish` consumes one immutable Coordination Round; rejected candidates consume none |
| Total Tasks | `32` accepted Tasks across the Run |
| Batch size / Graph concurrency | `8` Tasks per Dispatch Batch and `max_concurrency=8` for Agent Graph dispatch |
| Prior context references | `8` `context_task_ids` per Task, earlier accepted Rounds only |
| Task objective | `2,000` Unicode characters after normalization |
| Specialist outer attempts | `3` total: initial plus at most two eligible retries; same Task ID; each attempt is a fresh actor run |
| Specialist model requests | `12` cumulative across all outer attempts; adapter passes the remaining allowance into each fresh run |
| Specialist successful Tool calls | `8` cumulative across all outer attempts; typed `ToolUnavailable` returns count as completed calls; model-request limit also bounds validation/failed-call loops |
| Specialist Tool/output retries | `0` hidden retries; the registered outer policy owns whole-run retry |
| Specialist model response | `max_tokens=2,000`; `60 s` per model request |
| Specialist Tool call | `20 s` binding-owned timeout per call; no complete-Specialist elapsed limit |
| Coordinator | At most `2` complete invocations per decision (initial plus one repair), `1` model request per invocation, no Tools, no hidden output retry, `max_tokens=1,500`, `60 s` per request |
| Synthesis | At most `2` complete invocations (initial plus one repair), `1` model request per invocation, no Tools, no hidden output retry, `max_tokens=4,000`, `120 s` per request |
| Tool model-visible return | `4 KiB` canonical UTF-8 JSON per call |
| Specialist Result | `16 KiB` canonical UTF-8 JSON; at most `16` Evidence IDs, `8` Calculation references, and `8` Data Gaps |
| Data Gap | `512` Unicode characters for requested coverage and exactly one bounded `GapProvenance` |
| Specialist prior-Result context | At most the `8` referenced Results and `64 KiB` aggregate canonical UTF-8 JSON |
| Coordinator input | `16 KiB` per projected prior Result and `128 KiB` aggregate canonical UTF-8 JSON |
| Prepared Synthesis input | `256 KiB` aggregate canonical UTF-8 JSON; at most `64` eligible Evidence excerpts of `4 KiB` each and `32` Calculation records of `2 KiB` each |
| Canonical final Markdown | `32 KiB` UTF-8 after code-owned incomplete disclosure is prepended |
| Recursion | Explicit LangGraph `recursion_limit=40`; overflow is fatal and never translated to bounded completion |

The timeout and token values are intentionally conservative POC defaults, not
claims that the frameworks prescribe those numbers. They live in the injected
execution policy/registered Specialist definition, not in model-authored input.
Aggregate model/Tool/token/cost figures across parallel top-level Tasks remain
telemetry and a stop-future-dispatch signal, exactly as already decided.

Exhaustion is also explicit. A binding that cannot form a valid `4 KiB` success
projection returns `response_unusable` when that condition is on its expected
allowlist. A Specialist result or prior-context overflow rejects that attempt;
eligible outer retry may run, otherwise the Task becomes `TaskFailed`. A
Coordinator aggregate-context cap stops further dispatch and enters normal
incomplete completion from accepted outcomes. A Prepared Synthesis or final
Markdown overflow skips publication of the candidate and returns a deterministic
bounded incomplete answer; it never silently drops accepted Results, Evidence,
Calculations, gaps, or the disclosure. Fatal recursion overflow remains distinct.

Because PydanticAI checks `tool_calls_limit` before a parallel Tool set and may
execute none of that set when it would overflow (primary-source fact 6), the
contract test must cover an exact-bound parallel request. Because LangGraph
recursion counts execution steps and raises rather than returning a domain value
(fact 10), the legal-path proof and overflow behavior must remain separate.

### Spec patch text

Replace every scattered numeric/bounded execution paragraph with the preceding
table and add:

> All limits are inclusive and code-owned. There are at most four accepted
> Dispatch Batches (initial plus three follow-ups) and at most one subsequent
> accepted `Finish`; every accepted Decision is a Coordination Round. If the
> fifth Coordinator Decision proposes another dispatch, code does not accept it
> and enters bounded incomplete completion. Rejected Coordinator/Synthesis candidates consume their
> adapter invocation allowance but do not create a Coordination Round or Task.
> Before accepting a Dispatch Batch, code checks the proposed Round, Task, batch,
> and context counts against the remaining limits. If no further accepted
> decision can be executed, code enters bounded incomplete completion from the
> already accepted outcomes. Specialist model-request and successful-Tool-call
> allowances are cumulative across its at most three fresh outer runs; the
> adapter reuses one Task-local usage accumulator and the same limits for its
> sequential outer attempts. Parallel sibling Tasks never share that mutable
> accumulator. Canonical byte caps
> are measured over normalized UTF-8 JSON and overflow fails rather than
> truncating. The LangGraph recursion limit is an independent fatal safety guard,
> not the expected termination mechanism.

### Counterexample tests

| Test | Counterexample | Required assertion |
|---|---|---|
| `test_finish_after_four_dispatch_rounds` | Four accepted dispatch Decisions followed by `Finish` | Five immutable Coordination Rounds; normal completion; no sixth Coordinator call |
| `test_fifth_dispatch_is_not_accepted` | After four accepted batches the fifth Decision proposes another dispatch | No fifth batch or Tasks are accepted; code produces incomplete bounded completion from the four accepted batches |
| `test_batch_and_total_task_limits_are_checked_before_send` | Existing 28 Tasks plus proposed batch of 5 | Entire Decision rejected before any `Send`; no partial four-Task dispatch |
| `test_parallel_tool_set_at_exact_limit_and_over_limit` | Remaining Tool allowance equals set size, then is one smaller | Exact set runs; over-limit set runs no Tools and cannot silently exceed the actor bound |
| `test_outer_retry_receives_only_remaining_actor_allowance` | Attempt 1 consumes 10/12 model requests and fails transiently | Attempt 2 receives allowance 2; no reset to 12 |
| `test_utf8_caps_use_canonical_bytes_without_truncation` | Multi-byte text fits character cap but exceeds aggregate byte cap | Deterministic overflow path; no lossy partial context |
| `test_longest_legal_graph_path_is_below_40` | Maximum legal four-dispatch-plus-Finish path including finalization | Completes below recursion limit |
| `test_recursion_overflow_is_fatal` | Intentional routing loop reaches 40 | `GraphRecursionError`; no assistant Message, answer token, citations, or `done` |

## Combined acceptance gate

The specification should not retain `Status: ready-for-agent` until all of the
following are explicit:

1. expected unavailability is converted inside each registered Tool binding,
   including timeout behavior;
2. every accepted Data Gap has same-attempt, self-contained minimal Tool provenance;
3. accepted Task failures, accepted gaps, and execution-limit completion feed a
   code-owned monotonic incompleteness disclosure;
4. all structural, actor, timeout, payload, and context limits have exact values
   and inclusive semantics;
5. the counterexample suite above passes through both the PydanticAI-native
   contract seam and the real LangGraph HTTP/SSE path where applicable.

Overall current verdict: **FAIL, narrowly remediable by specification patch; no
architecture reset required.**

## Open questions

There are no blocking design questions for the POC if the conservative policy
above is accepted. Two explicitly post-POC questions remain:

1. Should later successful research be allowed to clear an earlier accepted
   failure/gap? Doing that safely requires domain-specific coverage identities
   and equivalence rules. The POC deliberately over-discloses instead.
2. Should the proposed per-call timeout and token defaults change after provider
   canary measurements? Tuning may change numeric policy values, but not timeout
   ownership, typed unavailability, provenance, or publication invariants.
