# Evidence handoff from concurrent Specialist Tools to Synthesis

Date: 2026-09-03

Decision: accepted as option E-A after independent review. The POC guarantees
active-request execution only; transparent process-crash resume is deferred.

## Question

In the LangGraph + PydanticAI Agent POC, how should Evidence created by Tools
inside concurrent Specialist invocations become available to accepted
`SpecialistResult`s and Synthesis while full Evidence bodies remain
request-local and are not checkpointed?

The review compares:

1. an injected request-owned concurrent Evidence catalog;
2. a LangGraph state reducer channel containing Evidence bodies;
3. PydanticAI `ToolReturn.metadata` extraction followed by success-gated
   contribution;
4. LangGraph runtime context, Store, managed values, and `UntrackedValue` as
   possible plumbing.

## Conclusion

Use a composition of options 1 and 3:

> **Each Specialist attempt stages typed Evidence inside its own completed
> PydanticAI run. After the terminal `SpecialistResult` validates, the
> `execute_specialist` adapter extracts Evidence from successful
> `ToolReturnPart.metadata`, validates the Result's Evidence IDs against that
> attempt and already accepted prior-round Evidence, and batch-contributes only
> newly referenced Evidence to one request-owned concurrent catalog. The
> LangGraph branch returns only the accepted `TaskOutcome`, Evidence IDs, and
> lightweight accounting.**

Pass the catalog as a run dependency through typed LangGraph runtime context.
Do not let Tools mutate the accepted catalog directly. Do not put Evidence
bodies in a reducer-backed Graph state channel, LangGraph Store, `Send` packet,
or checkpoint. Do not create a custom managed value or custom untracked reducer
for the POC.

This is close to the reviewer's original recommendation, but it sharpens the
owner and acceptance boundary:

- **owner:** the API request owns exactly one catalog for one Agent Graph
  invocation;
- **staging owner:** one `AgentRunResult` owns one Specialist attempt's
  unaccepted Evidence;
- **body-cache contribution owner:** `execute_specialist`, after terminal output
  validation;
- **consumer:** the barrier derives eligible IDs from accepted
  `TaskSucceeded.result.evidence_ids`; Synthesis resolves only those IDs;
- **durability:** Evidence bodies intentionally have none. A new-process resume
  cannot reconstruct them and must fail closed or restart research.

The final durability limitation is unavoidable: LangGraph cannot both avoid
persisting Evidence bodies (or resolvable durable references) and guarantee
transparent crash recovery after the producing branch has been checkpointed.
That limitation fits the current ADRs, which explicitly do not expose recovery,
resume, or an Artifact repository. It must be revisited before those features
are enabled.

## Why this fits the frameworks

### PydanticAI provides the attempt-local staging surface

PydanticAI's official advanced Tool-return contract separates
`ToolReturn.return_value`, which is serialized and sent to the model, from
`ToolReturn.metadata`, which application code can access and which is not sent
to the LLM. The documentation explicitly notes that other frameworks call such
metadata an artifact
([advanced Tool returns](https://ai.pydantic.dev/tools-advanced/#advanced-tool-returns)).

The completed `AgentRunResult` exposes `new_messages()`, which returns only
messages produced during the current run
([message history](https://ai.pydantic.dev/message-history/#accessing-messages-from-results)).
PydanticAI records ordinary user-defined Tool results as `ToolReturnPart`s, and
the installed source contract defines `ToolReturn.metadata` as application-only
data
([PydanticAI message types](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/messages.py#L860-L886)).
Together these provide a bounded, attempt-local place from which the adapter can
extract typed Evidence after the Agent finishes; no mutable collector is needed
inside the Tool.

Only `ToolReturnPart`s with a successful outcome and the application's exact
typed metadata envelope are candidates. Tool failures are represented as
failed Tool-return parts
([failed Tool results](https://ai.pydantic.dev/tools-advanced/#tool-failed));
arbitrary Tool metadata, synthesized history-repair returns, and model-authored
return values must not be interpreted as Evidence.

An outer retry must start a fresh PydanticAI run and must not carry the failed
attempt's messages into it. PydanticAI's history documentation likewise says
that retrying a failed run requires rebuilding history without the failed
attempt's messages
([run identity and failed-run retry](https://ai.pydantic.dev/message-history/#accessing-messages-from-results)).

### LangGraph state reducers are the wrong place for full Evidence bodies

LangGraph state keys are channels. Parallel nodes writing one key need an
appropriate reducer, and reducers define how node updates become shared state
([Graph API reducers](https://docs.langchain.com/oss/python/langgraph/graph-api#reducers)).
A stable-ID map-union reducer could deterministically merge Evidence bodies,
but it would violate the storage requirement: LangGraph checkpoints persist
Graph state snapshots, and also persist successful per-task writes within an
in-progress parallel superstep for pending-write recovery
([checkpoint and pending-write semantics](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints)).
Therefore a reducer channel containing Evidence bodies would put those bodies
in both checkpoints and/or checkpoint writes.

Reducers remain appropriate for lightweight `TaskOutcome` maps and Evidence ID
sets. Their merge must be associative and idempotent, and stable identity must
be assigned before the write rather than inside a reducer; LangGraph documents
the same constraints for replay-safe aggregation
([Pregel reducer requirements](https://docs.langchain.com/oss/python/langgraph/pregel#bulk-reducer-requirement)).

### Runtime context is the right dependency carrier, not the acceptance mechanism

LangGraph runtime context is specifically intended for static run dependencies
such as user identity or a database connection and is injected into nodes
([runtime context](https://docs.langchain.com/oss/python/langgraph/graph-api#runtime-context),
[Runtime source](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/runtime.py#L74-L139)).
The catalog reference can therefore be carried in a typed request context. The
reference is fixed for the Graph invocation even though the catalog implements
concurrency-safe batch contribution internally.

Runtime context must not be treated as a hidden transactional state channel.
The adapter, not the framework, owns catalog validation and contribution. A
catalog write and the node's returned Graph update are not one distributed
transaction. Correctness instead follows from a narrower rule: Synthesis can
resolve only IDs appearing in accepted `TaskSucceeded` results. An orphan
catalog entry created before a node update is applied is therefore unreachable
and cannot become report support. Accordingly, the request-owned object is a
validated body cache; the authoritative eligible catalog is the deterministic
intersection of accepted Task result IDs and that cache.

### Store and managed values do not fit this lifecycle

LangGraph documents Store as application-defined persistence outside Graph
state, intended for long-term data shared across threads; checkpointers are the
thread-scoped persistence mechanism
([checkpointer versus Store](https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-vs-store)).
Using Store would introduce exactly the Artifact persistence surface rejected
by the current ADRs. Even an in-memory Store compiled into a shared Graph would
have broader lifetime and namespacing/cleanup concerns than one request-owned
catalog.

LangGraph's public managed-value surface currently exposes orchestration values
such as remaining steps and last-step status, while the base managed-value APIs
are framework internals rather than an application Artifact contract
([managed module reference](https://reference.langchain.com/python/langgraph/managed)).
A custom managed value would add a novel framework extension without improving
the acceptance or durability boundary.

### `UntrackedValue` is useful but cannot fan in parallel Evidence maps

The current Linear Graph already marks request-local Evidence lists with
`UntrackedValue`, and the official channel is defined to store the last value
received while never checkpointing it
([UntrackedValue source](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/untracked_value.py)).
This is suitable for one sequential producer. It is not a map reducer:
with its default guard it rejects multiple updates in one step, and with the
guard disabled it keeps one last update. It would therefore reject or lose
concurrent Specialist contributions.

LangGraph also deliberately removes untracked channel values from persisted
node writes and sanitizes them out of persisted `Send` packets
([checkpoint-loop source](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_loop.py)).
Consequently Evidence bodies must not be embedded in dynamic `Send` inputs if
reconstruction is expected. Creating a custom channel that combines untracked
checkpoint behavior with stable-ID reduction is possible in principle, but is
an unnecessary internal-framework extension for this POC.

## Option assessment

### Injected request-owned concurrent catalog

**Recommended as the accepted-Evidence owner, but only with adapter-owned
success gating.**

Advantages:

- full bodies never enter Graph state or checkpoint writes;
- all parallel Specialists in the active request can resolve accepted prior
  Evidence and contribute newly accepted Evidence;
- stable-ID, ownership, capacity, and collision rules live in deterministic
  platform code;
- it reuses the repository's request-owned execution model without adding a
  repository or persistent Store.

Risks and controls:

- direct Tool mutation would make Evidence from failed attempts eligible, so Tools only
  attach metadata to their local PydanticAI run;
- concurrent commits require one lock and validate-before-mutate batch
  semantics;
- the catalog is not transactionally coupled to LangGraph's state write, so
  only accepted Task result IDs are reachable by consumers; a cancellation in
  the narrow interval after cache contribution but before the Graph accepts the
  node update may leave an unreachable body until the request object is
  discarded, but can never authorize it for Synthesis;
- process loss destroys bodies, so mid-run cross-process resume is unsupported
  and must fail closed.

### Graph state reducer channel containing Evidence

**Reject for bodies; retain reducers for IDs and outcomes.**

It is the most native fan-in mechanism and gives LangGraph pending-write
recovery, but that recovery is achieved precisely by persisting the values. It
therefore contradicts the no-checkpoint Evidence-body decision. An
ID-only reducer is safe, but IDs alone cannot supply Synthesis excerpts.

### `ToolReturn.metadata` extraction and success-gated contribution

**Recommended as the staging and promotion path.**

It keeps model-visible Tool output concise while retaining typed application
Evidence, naturally isolates separate outer attempts, and permits the adapter
to promote only Evidence referenced by an accepted Result. It is not a catalog
by itself; the request-owned catalog supplies parallel lookup and collision
control after promotion.

### Runtime context, Store, managed values, or `UntrackedValue`

- **Runtime context:** use only to inject the request-owned catalog reference.
- **Store:** reject because it is persistent and has a wider lifecycle.
- **Managed value:** reject as unnecessary framework extension.
- **`UntrackedValue`:** keep for sequential request-local values if useful, but
  do not use it for concurrent Evidence fan-in.

## Exact minimal POC contract

### Typed Tool metadata

Use one application-owned discriminated envelope so unrelated or synthesized
metadata cannot be mistaken for Evidence:

```text
EvidenceToolMetadata
  kind = "evidence"
  schema_version
  evidence: tuple[Evidence, ...]
```

Each `Evidence` contains at least:

- `evidence_id`;
- trusted `tenant_id` and `run_id`/request identity;
- registered Tool/provider and source-record identity;
- source title/URI or equivalent locator;
- retrieved/as-of time and freshness data;
- bounded excerpt/body required by Synthesis;
- canonical content digest and schema version.

The Tool Executor, not the model, assigns the ID before returning. Derive it
from a canonical, versioned tuple containing Tenant, Run, registered source,
source-record/version or as-of identity, and the canonical content digest. Do
not include attempt number or completion time: a retry or two Specialists
retrieving the same immutable Evidence should produce the same ID and coalesce.

### Adapter algorithm

For each outer Specialist attempt:

1. Start a fresh PydanticAI run with no failed-attempt messages.
2. Tools return bounded model-visible findings/IDs in `return_value` and typed
   `EvidenceToolMetadata` in `metadata`.
3. When the run completes, validate the two-field `SpecialistResult`.
4. Extract candidate Evidence only from successful `ToolReturnPart`s in
   `AgentRunResult.new_messages()` whose metadata validates as the exact
   application envelope.
5. Validate every Result Evidence ID. It must resolve either to:
   - eligible Evidence produced in this accepted attempt; or
   - eligible accepted Evidence selected from earlier-round Task context.
   Unknown, stale, foreign-Tenant, foreign-Run, and unselected prior IDs fail
   the Specialist Result.
6. Batch-contribute only newly produced Evidence IDs actually referenced by the
   accepted Result. Perform shape and ownership validation before locking, then
   validate collisions and aggregate capacity and commit the complete batch
   under one lock so one bad item cannot partially contribute a batch.
7. Return `TaskSucceeded(result)` and lightweight ID/accounting updates. A
   failed, timed-out, or invalid attempt reaches no contribution call and
   returns no accepted IDs. Cancellation immediately after contribution can
   leave only an unreachable request-local body as described above; it never
   creates an eligible ID.

The catalog implements:

- request identity fixed at construction;
- concurrency-safe `contribute_batch()` and read-only `resolve_many()`;
- same stable ID plus byte-equivalent canonical Evidence: idempotent no-op;
- same stable ID plus different canonical Evidence: invariant failure;
- per-Task and request-wide item/byte limits checked without silent truncation;
- immutable returned objects or defensive copies so callers cannot mutate an
  accepted entry.

### Fan-out, barrier, and Synthesis

- Every `Send` branch owns a distinct PydanticAI result and performs no shared
  mutation before its terminal output validates.
- Expected Task failures are converted into terminal `TaskFailed` values so the
  full batch reaches the barrier. Invariant/authorization/programmer failures
  abort the Run and never route to Synthesis.
- The barrier computes eligible Evidence IDs only from accepted
  `TaskSucceeded.result.evidence_ids`, in canonical `(round,
  dispatch_order, evidence_id)` order. Wall-clock completion order is ignored.
- The request catalog resolves those IDs and applies the Synthesis catalog's
  count/byte bounds. Missing bodies fail closed; they are never silently
  dropped from a supposedly supported Result.
- Synthesis receives bounded source metadata and excerpts. It never receives
  the mutable catalog object, raw Tool payloads, or Specialist transcripts.
- Publication gates resolve `[E#]` aliases against the same immutable bounded
  projection used to prompt Synthesis.

## Checkpoint and replay semantics

Graph state and checkpoints retain only stable Task outcomes, their bounded
summaries and Evidence IDs, coordination control, and Conversation messages.
They retain no Evidence body or PydanticAI Specialist transcript.

This gives safe active-request execution and ordinary outer retries, but not
transparent process-crash recovery after an Evidence-producing node succeeds.
LangGraph persists successful branch writes in a partially failed superstep so
those branches need not rerun on resume
([pending writes](https://docs.langchain.com/oss/python/langgraph/persistence#pending-writes));
a newly constructed request catalog would not contain their bodies. The
runtime must therefore treat a missing referenced body on resumed execution as
an invariant/unavailable-runtime failure, never synthesize from IDs alone.

If durable resume or HITL later becomes a product requirement, choose one
explicit extension:

1. persist encrypted Evidence Artifacts under retention and Tenant isolation;
2. checkpoint trusted, versioned source locators and deterministically
   rehydrate/revalidate bodies; or
3. restart the entire research Run under a new request identity.

That future decision cannot be hidden behind the current request-local
catalog.

## Required tests

1. **Metadata separation:** Tool Evidence metadata is available to adapter code
   but absent from the model-visible Tool return.
2. **Successful promotion:** a validated Result referencing current-attempt
   Evidence contributes it and can be synthesized/cited.
3. **Unreferenced Evidence:** Evidence produced but omitted from the accepted
   Result is not eligible for Synthesis.
4. **Unknown/foreign/stale ID:** Result validation fails before
   `TaskSucceeded` and before catalog mutation.
5. **Failed-attempt isolation:** provider failure, timeout, cancellation,
   invalid terminal output, and abandoned retry contribute nothing even after
   an earlier Tool call succeeded.
6. **Retry idempotence:** a successful retry producing identical Evidence gets
   the same ID; only its accepted run contributes.
7. **Parallel convergence:** reversed completion order yields the same accepted
   ID set, Synthesis aliases, and report input.
8. **Collision:** concurrent byte-identical writes are idempotent; same ID with
   different canonical content aborts as an invariant failure.
9. **Batch atomicity and limits:** one invalid/overflowing entry commits none;
   count and byte overflow fail without truncation.
10. **Expected mixed batch:** successful and expected-failed Tasks reach the
    barrier; only successful referenced Evidence is resolvable.
11. **Checkpoint inspection:** PostgreSQL checkpoint state and pending writes
    contain Evidence IDs but no Evidence bodies, excerpts, raw Tool payloads,
    or Specialist messages.
12. **Missing catalog on resume:** a checkpoint containing accepted Evidence
    IDs but a fresh empty request catalog fails closed before Synthesis.
13. **Tenant/Run isolation:** catalog resolution and contribution cannot cross
    either identity even when an Evidence ID is supplied directly.

## Repository fit

This recommendation preserves the accepted boundaries in:

- [ADR-0001](../adr/0001-langgraph-orchestration-with-pydantic-ai-model-boundary.md):
  LangGraph orchestrates while PydanticAI owns model-facing actors and Tools;
- [ADR-0003](../adr/0003-postgresql-is-the-durable-conversation-and-checkpoint-store.md):
  only lightweight Graph control and Conversation state are durable, with no
  retrieval Artifact store;
- [ADR-0004](../adr/0004-enforce-calculation-and-evidence-gates-in-code.md):
  Evidence eligibility and publication are deterministic code gates;
- [ADR-0006](../adr/0006-use-bounded-rolling-coordination-for-agent-research.md):
  parallel batches use bounded rolling coordination rather than a workflow DSL.

The existing Linear Graph demonstrates the local-storage intent by placing
Evidence lists in `UntrackedValue`, and its retrieval code already creates
stable request-bound IDs from canonical document data
([Linear Graph](../../app/langgraph_v2/graph.py),
[retrieval](../../app/langgraph_v2/retrieval.py)). The Agent Graph should reuse
the domain `Evidence` vocabulary and ID discipline, but replace the Linear
Graph's sequential untracked list with the request-owned concurrent catalog
because fan-out adds multiple producers.

## Decision requested

Accept the refined recommendation if the POC intentionally supports only
active-request execution for Evidence-backed Agent Runs:

> **Request-owned catalog injected through runtime context + PydanticAI
> `ToolReturn.metadata` staging + adapter-owned success-gated batch
> contribution; Graph state/checkpoints contain IDs only.**

If transparent process-crash resume is required now, reject this recommendation
and reopen the existing no-Artifact-store ADR boundary. There is no design that
provides both non-persisted bodies and cross-process recovery after producers
are skipped by pending-write replay.
