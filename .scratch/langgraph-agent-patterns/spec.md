# LangGraph Agent Patterns POC

Status: ready-for-agent

## Problem Statement

The Linear LangGraph baseline cannot express multi-source concurrency,
multi-hop research, bounded iterative coordination, or Specialist Agent
delegation. Professional users need these patterns without allowing an LLM to
control permissions, calculations, provenance, or unbounded execution.

## Solution

Allow the compatible `/v2/query/stream` endpoint to run either the existing
`linear` Graph or the new `agent` Graph according to the authenticated Tenant's
server-side mode configuration. The client cannot select or override the mode.
Use a fixed LangGraph safety skeleton around a bounded Coordinator–Specialist
loop, PydanticAI actors, registered read-only Tools, Agent Skills, deterministic
collection, and publication gates. `agent` is one runtime mode, not a collection
of Tenant-selected pattern implementations. Query variability is expressed as
successive bounded dispatch batches: independent work fans out within one batch,
later batches may depend on completed results, and the Coordinator may stop or
revise the next research step after each batch.

## User Stories

1. As a researcher, I want market data, reports, and news fetched concurrently,
   so that one Research Report can combine independent sources efficiently.
2. As an analyst, I want later Tasks to select accepted earlier Specialist
   Results, so that the platform can perform bounded multi-hop research.
3. As an analyst, I want concurrent Task outcomes collected deterministically,
   so that wall-clock completion order cannot change the final report.
4. As a user, I want Specialist Agents to perform bounded domain work, so that
   complex research remains modular and auditable.
5. As a user, I want consecutive Requests in one Conversation to retain only
   complete user/assistant pairs, so that follow-up questions have useful context
   without inheriting stale Run control state.
6. As a user, I want an ambiguous query to return a normal clarification, so
   that I can resolve it without creating a resumable half-Run.
7. As a user, I want the existing request boundary to keep Requests for one
   Conversation sequential, so that a later Run sees one completed predecessor
   rather than concurrent checkpoint mutations.
8. As a user, I want Requests for different Conversations to run concurrently,
   so that the existing per-Conversation execution invariant does not serialize
   unrelated work.
9. As a financial user, I want numerical outputs produced by registered
   functions over trusted inputs, so that the LLM cannot invent formulas,
   authoritative series, or calculated values.
10. As a financial user, I want incomplete research clearly disclosed, so that
    unavailable data or a structural execution limit cannot be mistaken for a
    globally complete conclusion.
11. As a compliance reviewer, I want every published Evidence and Calculation
    reference validated against the current Run's eligible support catalogs, so
    that report support is traceable and cannot name unauthorized material.
12. As a compliance reviewer, I want canonical calculated values rendered by
    code, so that published values exactly match their Calculation Artifacts.
13. As a compliance reviewer, I want the final assistant Message durably
    checkpointed before any answer frame is emitted, so that a published answer
    never precedes its canonical Conversation state.
14. As a Tenant administrator, I want models, Skills, Specialists, Tools,
    freshness, and search constraints controlled by policy, so that Agents cannot
    expand their own authority.
15. As a Tenant administrator, I want untrusted Tool content treated only as
    typed data, so that indirect prompt injection cannot change permissions,
    routing, Tool availability, or publication rules.
16. As an operator, I want the Agent Graph to rely on the existing guarantee that
    one `thread_id` cannot have concurrent Requests, so that it does not add a
    second admission, locking, or database-session lifecycle.
17. As an operator, I want bounded actor calls, coordination, retries, and
    recursion, so that an Agent Run cannot loop without a known limit.
18. As an operator, I want unsupported recovery to fail closed, so that missing
    request-owned Evidence is never silently converted into a plausible report.
19. As a developer, I want active parallel batches promoted atomically, so that
    an expected failure or invalid branch cannot leave ambiguous eligible state.
20. As a developer, I want one static Agent Graph assembled through narrow
    dependency interfaces, so that the POC proves reusable orchestration without
    introducing a workflow DSL or configurable Team framework.

## Implementation Decisions

- Make retry ownership explicit at every PydanticAI actor boundary. Coordinator
  and Synthesis disable PydanticAI Tool and output retries because their adapters
  own repair. POC Tools do not raise PydanticAI `ModelRetry` or `ToolFailed`;
  each registered binding catches only its allowlisted expected inability to
  provide requested read, fetch, or Calculation data and returns a PydanticAI
  `ToolReturn` whose model-visible `return_value` is a bounded, discriminated
  `ToolUnavailable` value. The binding applies its own per-call timeout and
  catches only that timeout plus explicitly registered expected-unavailability
  outcomes; it never uses a broad `except Exception`. The model-visible value
  contains only an allowlisted stable code and a bounded sanitized
  requested-coverage label. The complete POC code set is
  `source_unreachable`, `coverage_not_supported`, `stale_only`, `call_timeout`,
  and `response_unusable`; adding a code requires changing the registered Tool
  contract and its tests. Application-only `ToolReturn.metadata` carries a typed
  `ToolUnavailabilityRecord` that assigns an
  opaque unavailability ID and binds it to the current Run, Task, current outer
  attempt, Tool call, registered Tool, logical source when applicable, the Run's
  normalized UTC observation time, code, and coverage label. It contains no Evidence,
  Calculation Artifact, Tool arguments, credentials, raw exception, provider
  payload, retry history, or stack trace. The adapter later derives canonical
  Data Gaps from this metadata; it does not trust the model to report them. An
  authoritative successful query that establishes an empty or negative answer
  is an Evidence-producing success, not `ToolUnavailable`. The typed return does
  not end the PydanticAI run: a Specialist may call
  an allowed fallback, continue multi-hop, or combine successful siblings from
  internal fan-out/fan-in. Expected unavailability does not consume an outer
  Specialist retry. Disable hidden provider, transport, MCP, and LangGraph retry
  layers for this POC. Do not configure PydanticAI's Tool-timeout retry behavior
  for these bindings; the binding-owned timeout is the only path that converts a
  timeout to `ToolUnavailable`. Configure the actor end strategy explicitly so an
  accepted structured result cannot trigger additional business Tools. A
  registered Specialist may execute independent read-only Tool calls
  concurrently; dependent calls remain model-led, and canonical collection
  never depends on completion order. Cancellation, authorization, invariant,
  configuration, programmer, and unknown failures remain exceptions.
- Depend on the Linear Core's request-owned v2 SSE, Conversation/request identity,
  official shared PostgreSQL checkpoints and Tenant isolation. Keep
  `linear` as the initial Tenant default and enable `agent` only through trusted
  Tenant configuration.
- A Run remains the domain name for one execution, but it has no application-owned
  persistence model. LangGraph State and its official checkpoint are the durable
  execution authority; do not add a `runs` table, Run repository, duplicate
  checkpoint pointer, transport Event journal, or Redis recovery state.
- Keep the derived checkpoint `thread_id` stable for one Tenant, Subject,
  runtime mode, and Conversation. Use the Request's stable `request_id` as the
  logical Run identity and as the root of Graph-owned batch, Task, attempt, and
  publication correlation; this does not create a persisted Run entity. Reuse of
  one Request ID with different query content is a conflict.
- Treat the existing request-runtime guarantee that one complete derived
  `thread_id` can never have concurrent Requests as a prerequisite. The Agent
  Graph does not add advisory locks, a second admission policy, HTTP 409
  double-texting behavior, a dedicated run-session pool, or lock-lifecycle
  monitoring. If that upstream guarantee changes, concurrency admission must be
  designed as a separate capability rather than added implicitly here. Different
  checkpoint threads remain concurrent.
- Resolve the Tenant's configured mode before starting a request. One Tenant has
  exactly one active mode. Include that trusted mode in the derived checkpoint
  `thread_id`, so Linear and Agent state cannot mix even for the same public
  Conversation UUID. Runtime mode is fixed for the current deployment
  configuration; config reload must reject a mode change.
- Within an Agent-mode Conversation, coordinate every request independently from that
  request's query and authorized capabilities. Consecutive requests may use different
  execution shapes; do not store a Tenant-level or Conversation-level active
  Agent pattern.
- Across Agent requests, inherit only checkpointed `conversation_messages`.
  Reset the new request's Coordination Rounds, Tasks, Evidence, answer, errors,
  and every other Agent-local field before execution.
- Perform that reset in one unique initializer node before any parallel dispatch.
  Reducer-backed Run-local channels are cleared with LangGraph `Overwrite`, while
  scalar channels receive ordinary values such as `None`. Do not send overwrite
  values from concurrent branches, and do not apply `Overwrite` wrappers to
  scalar channels.
- Use the existing PydanticAI Query Understanding actor as the only Agent-mode
  actor that receives bounded complete prior user/final-assistant Conversation
  pairs. In one typed result it resolves the current query to a non-empty
  `standalone_query` and selects one Business Intent from the Tenant's compact
  Intent Catalog. Coordinator, Specialist, and Synthesis actors do not receive
  Conversation history or prior actor transcripts.
- If Query Understanding instead returns a validated clarification request,
  that result wins and the current Request completes successfully without
  resolving Research Scope or invoking Coordinator, Specialist, or Synthesis.
  Publish one normal terminal `done` event whose `answer` is the user-visible
  clarification, whose `clarification` contains the structured request, and
  whose `citations` is empty. Commit the current user message and final
  assistant clarification as one complete Conversation pair before releasing
  `done`. The user's answer starts a clean Run under a new Request ID and the
  same Conversation ID; it is not a LangGraph interrupt/resume or an error.
- Deterministic Graph code validates the selected Business Intent against the
  trusted Tenant Intent Catalog, resolves its server-owned execution profile,
  and fixes one immutable Research Scope for the Run before coordination. The
  scope may restrict eligible Specialists, callable Tools, data sources, and
  enforced search constraints, but it can never expand Tenant authority. The
  model-authored Intent Result does not name or grant Skills, Tools, data
  sources, permissions, or search filters.
- Share Query authorization and request-owned streaming mechanics, and inject
  the same official checkpointer. Keep each Graph's builder and execution policy
  inside that Graph's own module rather than adding Agent branches to the Linear
  Graph.
- Build the static Agent Graph through ordinary dependency injection of the
  Coordinator actor, Synthesis actor, typed Specialist registry, and execution
  policy. Each Specialist definition binds its actor factory, prompt/model,
  Tool allowlist, and eligible Skills. Do not add an `AgentTeamDefinition`, Team
  registry, configuration loader, persistence model, or workflow DSL in this
  POC; another Team can assemble the same builder with different dependencies.
- Use one static Agent Graph: initialize → pre-moderate → understand query →
  validate Intent and resolve Research Scope → coordinate → validate decision
  → dispatch one batch → stage and collect at a barrier → coordinate again,
  or prepare synthesis → synthesize → output gates → finalize state →
  publish. Do not compile a query-specific Graph, complete
  Task DAG, or execution IR. The POC Coordinator Agent uses its planning prompt
  and does not discover or activate Skills.
- The Coordinator Agent returns only a typed `DispatchBatch | Finish` decision.
  A Dispatch Batch contains one or more Task proposals with registered
  Specialist IDs, business objectives, and `context_task_ids` selecting Results
  from successful Tasks in earlier Coordination Rounds. The Coordinator does
  not author Task IDs, invoke Specialists as PydanticAI Tools, or name Skills,
  business Tools, timeout, retry, or budget values. The Agent Graph is the only
  delegation execution path. `Finish` is a discriminated decision with no
  business payload. It means only that the Coordinator proposes no further
  dispatch; it is not a claim of global research completeness. Code-owned
  termination causes remain Graph runtime state and are not passed to the
  Synthesis Agent.
- Deterministic code validates each proposed decision before dispatch. It checks
  registered, Tenant-eligible, and Research-Scope-eligible Specialists,
  non-empty batch size,
  `context_task_ids` existence, current Tenant/Run ownership, earlier-Round
  membership and successful accepted Result status, maximum Round, total Task,
  concurrency, bounded objective size, and the aggregate canonical UTF-8 JSON
  size of the selected prior Results after materialization. This validation
  occurs before `Send`; an oversized prior-Result context rejects the
  Coordinator Decision through its same-round repair rather than consuming a
  Specialist outer attempt. A different size observed after successful
  validation is a fatal invariant failure. Per-actor
  model-request, Tool-call, retry, and output limits are also checked at their
  owning adapter boundaries as loop-prevention guards. The current system has
  only read, fetch, and deterministic Calculation operations, so aggregate
  model/Tool counts are telemetry or a signal to stop later scheduling, not a
  business budget requiring atomic reservation across parallel branches.
  Tasks in one Dispatch Batch are mutually independent and may reference only
  Results completed before that batch; same-batch references are invalid.
  The Coordinator adapter is the sole owner of one explicit same-round repair:
  it makes at most two complete actor invocations with equivalent built-in
  output retry disabled, and a rejected attempt is not an accepted Coordination
  Round. After validation, the Agent Graph assigns each
  accepted Task a stable ID from Graph-owned Run, Round, and dispatch-order
  identity; model output never controls identity or retry correlation.
- Record every accepted Coordinator Decision as an immutable Coordination Round
  with a monotonic revision number. Multi-hop dependencies are stable references
  from a later round to completed earlier results. This bulk-synchronous POC does
  not start a downstream Task while another Task in the preceding
  batch is still running; dependency-driven pipeline scheduling remains a future
  extension.
- On every Coordination Round, deterministically rebuild the Coordinator Agent's
  model input from the standalone query, selected Business Intent, compact
  Specialist Descriptors already filtered by the Research Scope, all accepted
  earlier Coordinator Decisions, and all bounded `TaskOutcome`/Specialist
  Results in stable round and dispatch order. Canonical Data Gaps are projected
  to their bounded `DataGapView`; internal provenance IDs never enter the prompt.
  Do not add
  a separate model-authored `ResearchLedger`, provide only the latest batch, or
  pass Specialist internal messages, raw Tool payloads, or Evidence bodies.
  Enforce both per-result and aggregate Coordinator-context limits. If the
  bounded POC exceeds the aggregate limit, finish as incomplete rather than
  silently replacing canonical results with a lossy summary. Keep context
  construction behind a deterministic projection seam so later retrieval or
  compaction can change model input without replacing checkpoint truth.
- Let the injected LangGraph checkpointer naturally checkpoint lightweight
  Coordination Rounds, active batch control, staged contribution metadata,
  immutable accepted batches, status, and count/usage counters. An accepted
  batch contains its validated Task Outcomes,
  Evidence IDs, and complete bounded Calculation Artifact records. Do not build
  a separate checkpoint model, repository, public identity, or recovery
  contract. Evidence bodies, raw Tool payloads, and Specialist internal messages
  remain request-local.
- A Task is the accepted unit of delegated business work scheduled by the Agent
  Graph, not a record of every model or Tool call. In the current POC, a Task
  has a Graph-owned stable ID and delegates one objective to a registered
  Specialist Agent. Its `context_task_ids` select the bounded prior Specialist
  Results that the Graph materializes for that invocation. The Specialist may
  make multiple internal read-only Tool and deterministic calculation calls;
  those calls emit audit and SSE progress but do not become top-level Tasks.
- Dispatch the Tasks in one accepted batch concurrently and wait at one barrier
  before invoking the Coordinator again. Use the following single code-owned POC
  policy; limits are inclusive, and code rejects an action before execution if
  it would exceed one. The first four accepted Coordination Rounds may dispatch
  batches: one initial batch and up to three follow-up batches. After a fourth
  dispatch, the next Coordinator Decision may only be `Finish`; an invalid
  dispatch candidate receives the one allowed same-round repair, after which
  code finishes as incomplete. A rejected candidate consumes its adapter
  invocation allowance but is not an accepted Coordination Round.

  | Limit | POC value and semantics |
  |---|---|
  | Coordination | At most `5` accepted Decisions: at most `4` `DispatchBatch` Decisions plus one terminal `Finish` |
  | Tasks | `32` accepted Tasks per Run; `8` Tasks per batch; Agent Graph `max_concurrency=8` |
  | Dependency context | At most `8` earlier accepted `context_task_ids` per Task |
  | Task objective | At most `512` canonical UTF-8 bytes after single-line normalization; the same bounded value is used in incomplete disclosure |
  | Specialist outer attempts | `3` total: initial plus at most two eligible retries, all with the same Task ID and a fresh actor run |
  | Specialist actor work | `12` model requests and `8` completed Tool calls cumulative across all outer attempts; hidden Tool/output retries disabled |
  | Specialist calls | `60 s` per model request, binding-owned `20 s` per Tool call, model `max_tokens=2,000`; no whole-Specialist timeout |
  | Coordinator | At most `2` actor invocations per Decision, one model request each, no Tools, `60 s` per request, `max_tokens=1,500` |
  | Synthesis | At most `2` actor invocations total, one model request each, no Tools, `120 s` per request, `max_tokens=4,000` |
  | Tool model-visible return | At most `4 KiB` canonical UTF-8 JSON per call |
  | Specialist Result | At most `16 KiB` canonical UTF-8 JSON, `16` Evidence IDs, and `8` Data Gaps |
  | Data Gap | At most `256` UTF-8 bytes of requested coverage plus one bounded `GapProvenance` |
  | Gap provenance | Opaque unavailability ID and registered Tool ID at most `64` ASCII characters each; optional logical source ID at most `256` UTF-8 bytes; observation time is normalized UTC |
  | Calculation Artifacts | At most `8` per Task contribution and `4 KiB` canonical UTF-8 JSON each; the complete accepted Run collection is at most `1 MiB` as one canonical UTF-8 JSON value including IDs and container framing; at most `32` projections of `2 KiB` each in Prepared Synthesis |
  | Request-owned Evidence bodies | At most `16 KiB` each and `8 MiB` total; the cache rejects a write before either cap is exceeded |
  | Specialist prior-Result context | At most the `8` referenced Results and `64 KiB` aggregate canonical UTF-8 JSON |
  | Coordinator context | At most `16 KiB` per prior Result and `128 KiB` aggregate canonical UTF-8 JSON |
  | Prepared Synthesis | At most `256 KiB` canonical UTF-8 JSON, `64` Evidence excerpts of at most `4 KiB` each, and `32` Calculation records of at most `2 KiB` each |
  | Canonical final output | At most `192 KiB` UTF-8 including the code-owned incomplete disclosure |
  | LangGraph recursion | Explicit `recursion_limit=40`; overflow is fatal rather than bounded completion |

  Specialist request and Tool-call counters are Task-local and cumulative across
  its sequential outer attempts; parallel sibling Tasks never share a mutable
  counter. Aggregate model requests, Tool attempts, tokens, and cost across
  parallel Tasks are measured and reported, but are not atomically reserved.
  Exact cross-branch reservation is unnecessary in this read/fetch/calculate-only
  system. Canonical byte limits are measured after normalized serialization. A
  Tool that cannot form a valid bounded success projection returns the
  allowlisted `response_unusable` unavailable outcome. Specialist output that
  fails schema or size validation follows the structured-output-invalid retry
  classification below. Coordinator aggregate-context
  overflow stops further dispatch. At the barrier, an active batch that would
  exceed the accepted Calculation-state cap is not promoted. In the same
  terminal state update, the barrier records `calculation_state_limit`, clears
  the reducer-backed staging map and scalar active-batch manifest, and routes to
  completion from earlier accepted batches; no outcome or Artifact from the
  rejected batch becomes canonical. An Evidence body
  cache overflow stages a `TaskFailed` contribution without outer retry or
  Evidence IDs; any body written before the overflow remains only as an
  unreachable request-local orphan, so concurrent identical sibling writes are
  never deleted. Successful siblings still reach the barrier.
  Prepared Synthesis or final-Markdown overflow bypasses the candidate and
  returns a deterministic bounded incomplete answer.
  None of these paths silently truncates accepted Results, Evidence,
  Calculations, gaps, or the disclosure; recursion overflow remains fatal.
- Keep expected Tool unavailability inside the active Specialist run as the
  typed value described above. A Tool-call timeout is converted inside that Tool
  binding, so it neither aborts multi-hop work nor cancels successful internal
  fan-out siblings. If the Specialist can form a valid bounded Finding, the
  adapter returns `TaskSucceeded` with the available Evidence and the canonical
  Data Gaps derived from the accepted attempt's unavailable Tool metadata; this
  includes partially successful and successful negative research. Only an
  allowlisted failure that prevents the Specialist run itself from producing a
  valid terminal result—including exhausted provider/model request failure or
  exhausted model-output validation—may become `TaskFailed` after outer retry
  policy is exhausted. Deterministic collection still waits for one terminal
  outcome per Task, then the Coordinator may request substitute research,
  accept the disclosed gaps, or finish. External cancellation, programmer
  errors, reducer conflicts, authorization-boundary violations, and corrupted
  control state are fatal and must fail the Run instead of being disguised as
  Tool or Task outcomes.
- Separate automatic technical retry from research follow-up. The registered
  Specialist execution policy may retry the same Task for eligible
  transient failures, retaining its Task ID, incrementing its attempt,
  and consuming its original actor-local count limits; this does not call the
  Coordinator or create a Coordination Round. After technical retry is
  exhausted, the failed `TaskOutcome` is collected. Any subsequent change of
  Specialist, source, scope, or objective is a new Coordinator Decision with a
  new Task ID in a later Coordination Round, not another attempt of the failed
  Task.
- Make the Specialist execution adapter's failure classification a closed
  allowlist. This POC pins the concrete mapping for the repository's locked
  `pydantic-ai==1.93.0` V1 baseline; its implementation and CI use the frozen
  lock and assert that version rather than resolving the declared open lower
  bound to a different release. If the locked version changes first, stop and
  update this mapping explicitly. The later PydanticAI migration must replace
  these concrete symbols for its selected V2 version without changing the stable
  categories or behavior:

  | Adapter category | Outer retry | Terminal behavior |
  |---|---|---|
  | Allowlisted transient model timeout, connection failure, HTTP 429, or HTTP 5xx | Up to two retries after the initial attempt; each is a fresh actor run subject to the Task's cumulative actor-local limits | `TaskFailed` after the third total attempt |
  | Structured output rejected by schema, size, or deterministic output validation | Up to two retries after the initial attempt; each is a fresh actor run with bounded deterministic validation feedback and unchanged Task input | `TaskFailed` after the third total attempt |
  | Actor-local model-request or Tool-call count exhausted | None | `TaskFailed` |
  | Request-owned Evidence-cache capacity exceeded while accepting a Tool success | None | `TaskFailed` with no Evidence IDs; retain any unreachable body only under the cache rule above |
  | Expected registered Tool unavailability | None | Return the typed `ToolReturn` and continue the same actor run |
  | Cancellation, authorization, configuration, invariant, programmer, or unknown failure | None | Re-raise and fail the Run |

  Apply the current V1 mapping in this order so a superclass cannot make the
  allowlist broader than intended:

  | Current V1 signal | Stable classification |
  |---|---|
  | The built-in `TimeoutError` raised by the adapter-owned per-model-request timeout | Allowlisted transient model timeout |
  | `pydantic_ai.exceptions.ModelHTTPError` with status `429` or `500` through `599` | Allowlisted transient HTTP failure |
  | Any other `ModelHTTPError` | Fatal; re-raise |
  | A non-HTTP `pydantic_ai.exceptions.ModelAPIError` whose typed direct cause is `openai.APIConnectionError`, including its timeout subtype, at a configured Azure/OpenAI model-call boundary | Allowlisted transient connection failure |
  | `httpx.ConnectError` or `httpx.TimeoutException` escaping directly from the configured Google model-call boundary | Allowlisted transient connection or model timeout |
  | `pydantic_ai.exceptions.IncompleteToolCall`, but only when the dedicated invocation capture proves that the truncated call is that actor's registered terminal output Tool | Structured-output invalid |
  | `UnexpectedModelBehavior` for which the dedicated invocation capture proves rejection of that actor's terminal structured output | Structured-output invalid |
  | `pydantic_ai.exceptions.UsageLimitExceeded` raised by limits configured for this invocation's model-request or Tool-call counts, with PydanticAI token limits left unset | Actor-local count exhausted |
  | `ContentFilterError`, any other `UnexpectedModelBehavior`, or any unlisted PydanticAI/provider exception | Fatal; re-raise |

  `ModelHTTPError` is a `ModelAPIError` subclass in the pinned V1 baseline, and
  `ContentFilterError` is an `UnexpectedModelBehavior` subclass, so the adapter
  must classify the narrower cases first. When the public exception type alone
  is insufficient, classification may use only typed facts from that attempt's
  dedicated captured messages, known model-provider boundary, typed direct
  cause, configured usage-limit dimensions, the known actor output schema and
  output-Tool identity, and application-owned validation results; it must not
  match error message text or inspect provider bodies. A bare `ModelAPIError`, a
  Google `APIError` with a non-HTTP code, an `IncompleteToolCall` for a business
  Tool, a `TimeoutError` escaping a business Tool, and any signal whose origin
  cannot be proven are fatal. Binding-owned business-Tool timeout remains the
  earlier typed `ToolUnavailable` path and does not reach this classifier.
  Schema, canonical-size, and
  deterministic domain validation performed after a returned draft use one
  application-owned structured-output-invalid signal. No broad
  `UnexpectedModelBehavior` or `ModelAPIError` catch may convert failures from
  an unregistered provider or a different actor phase.

  No unlisted exception is retryable or convertible to `TaskFailed`, and the
  adapter must not use `except Exception` as a Task-failure conversion boundary.
  Every fresh actor run retains the same Task ID, increments its attempt, and
  consumes the same cumulative request and Tool-call allowances.
- Keep Run, Task, and Specialist state separate inside LangGraph State. A
  Specialist branch never writes directly to canonical accepted results. It
  contributes one immutable, stable-ID `BatchContribution` containing its
  terminal Task Outcome, referenced Evidence IDs, and complete bounded
  Calculation Artifacts. Reducers merge staged contributions associatively and
  idempotently; the same ID with different content is an invariant failure.
- Make the barrier the sole promotion authority. It verifies that the active
  batch has exactly one valid terminal contribution for every expected Task,
  validates identity, attempt, Evidence, and Calculation Artifact membership,
  and then promotes the complete batch as one immutable `AcceptedBatch` in one
  state update. That update also clears the reducer-backed staging map and the
  scalar active-batch manifest using their correct channel semantics. Accepted
  batches are the only canonical source from which Coordinator, Synthesis, and
  publication views are derived.
- A mixed batch of successful and expected-failed Tasks is complete and may be
  promoted after validation; successful siblings remain eligible. External
  cancellation, authorization failure, corrupted state, reducer conflict,
  programmer error, and checkpoint failure are fatal exceptions: they abort the
  Run and are never converted into failed Task Outcomes.
- Model terminal Task outcomes as a discriminated
  `TaskSucceeded(task_id, result) | TaskFailed(task_id)` union. The union contains
  only the fields required by deterministic batch collection and Coordinator
  replanning. Specialist model usage is returned through a separate Graph-owned
  accounting update. Failure classification, attempts, retry history, raw
  exceptions, and stack traces remain execution-adapter telemetry and are not
  placed in the Coordinator-visible outcome.
- On a research-completion path, before constructing the terminal answer,
  deterministic code builds one
  immutable, application-only `IncompleteResearch` projection from accepted
  state and the selected terminal path. It
  contains stable failed Task IDs, every canonical Data Gap from accepted
  successful Results, a stably ordered set of code-owned structural reasons,
  and an `insufficient_evidence` boolean. Code sets that boolean when the
  terminal accepted state has no eligible Evidence. An authoritative successful
  query that establishes an empty or negative business answer still produces
  eligible Evidence and therefore does not set it. An accepted Evidence ID whose
  body cannot be resolved is instead unsupported recovery and fails the Run; it
  is never converted to insufficient Evidence.
  Clarification and pre-moderation terminal paths do not enter research
  completion, do not construct `IncompleteResearch`, and are never marked
  insufficient merely because they have no Evidence.
  The complete POC reason set is `task_limit`, `coordination_limit`,
  `coordinator_context_limit`, `calculation_state_limit`,
  `prepared_synthesis_limit`, and `final_markdown_limit`; adding a reason
  requires a contract and test change. A
  Synthesis candidate that would exceed the final-output cap is discarded; code
  then constructs the projection with `final_markdown_limit` and renders the bounded
  deterministic incomplete answer instead. It
  resolves failed Task IDs to their accepted
  bounded objectives for disclosure, but never includes a technical failure
  reason. Ordering follows stable Round, Task, and unavailable-outcome identity.
  Any failed Task, Data Gap, structural reason, or true
  `insufficient_evidence` value makes the response incomplete. This POC is
  deliberately conservative and monotonic: later research does not clear an
  earlier accepted failure or gap because the platform has no deterministic
  domain contract for equivalent coverage. This may over-disclose but cannot
  hide observed data loss, and it adds no Graph channel, repository, or global
  semantic completeness model.
- PydanticAI builds role-configured actors with model abstraction, activated Skill instructions, approved tool bindings, structured outputs, and usage. The Coordinator Agent does not execute business Tools.
- Treat configured count bounds as code-owned routing policy. When the maximum
  Task, Coordination Round, or aggregate Coordinator-context count prevents
  further dispatch, finish research from already accepted outcomes. If eligible
  Evidence exists, invoke Synthesis; without eligible Evidence, skip Synthesis
  after setting `IncompleteResearch.insufficient_evidence` and return the
  deterministic insufficient-Evidence answer. Both paths use the same
  code-owned incomplete-research disclosure and checkpoint the exact
  published assistant Message. Response metadata sets `completion_status` to
  `incomplete` whenever `IncompleteResearch` contains any signal. Its stable
  `termination_reason` is `partial_results`, `execution_limit`,
  `partial_results_and_execution_limit`, or `insufficient_evidence`. The last
  value applies only when absent eligible Evidence is the sole incompleteness
  signal; accepted Task failures or Data Gaps take the partial-results category,
  structural reasons take the execution-limit category, and both retain the
  combined category even when Evidence is also absent. No new terminal SSE type
  is added.
- Configure the Agent Graph with an explicit recursion limit of 40. This limit
  counts LangGraph supersteps, not model requests, Tool calls, Specialist outer
  attempts, or Coordinator/Synthesis repairs performed inside an owning node.
  The longest intended application-node route has at most 31 supersteps: four
  fixed entry steps; four dispatch cycles of at most five steps each for
  coordinate, validate, dispatch, Specialist fan-out, and barrier; two steps for
  the terminal `Finish` decision and validation; and five final steps for
  prepare-synthesis, synthesis, output gates, `finalize_state`, and publish.
  The remaining headroom covers framework bookkeeping but is not permission to
  add another research loop. A topology change must recompute this static bound.
  Exceeding 40 is a fatal Graph error with no terminal success event, not a
  bounded-completion outcome.
- The Agent Graph owns no Run-wide wall-clock cutoff and no whole-Task elapsed
  timeout. External cancellation/disconnection before a successful
  `finalize_state` checkpoint commit propagates and leaves no final assistant
  Message or `done`; already-emitted progress and earlier non-final checkpoints
  may remain. A cancellation that races with that commit is resolved by the
  actual checkpoint outcome, not by the transport exception: an uncommitted
  final update leaves no final Message, while a committed update remains the
  canonical Conversation truth. Cancellation after the commit never rolls back
  or deletes that response; publication may deliver zero or partial final frames
  and no `done`. Authorization, reducer, corrupted-state, programmer, and
  checkpoint failures remain fatal, are never converted into bounded completion,
  and do not produce `done`.
- Project failure events to stable, sanitized public messages. Raw provider
  exceptions, counters, retry history, stack traces, and internal diagnostics
  remain telemetry and must not appear in SSE payloads.
- Keep research control and final authorship separate. The PydanticAI
  Coordinator emits only bounded dispatch-or-finish decisions and never drafts
  the final answer. After coordination finishes, a separate PydanticAI
  Synthesis Agent consumes a deterministic projection of accepted batches to
  produce the typed Research Report before deterministic publication gates.
  Its model-visible `SynthesisInput` contains only the standalone query, selected Business
  Intent, accepted bounded Specialist Results—including their bounded
  `DataGapView` values—an eligible bounded Evidence catalog with resolved excerpts, and an
  eligible bounded Calculation catalog.
  The Calculation catalog exposes aliases and concise interpretation metadata,
  but not canonical values for the model to retype. Do not add Conversation
  history, a global `known_gaps`, `report_scope`, runtime termination reason,
  Task status, retry, budget, or execution-limit state to this input. Accepted
  Data Gaps are observations from actual unavailable Tool outcomes, not a claim
  that code knows every globally missing topic. If no eligible
  Evidence remains, the Agent Graph returns the deterministic
  insufficient-Evidence outcome without invoking the Synthesis Agent; that
  deterministic answer sets the code-owned insufficient-Evidence signal and
  uses the same incomplete disclosure rather than exposing raw Tool or Task
  failures.
- Build one frozen `PreparedSynthesis` value before the first Synthesis call.
  It owns the exact model input, deterministic Evidence and Calculation alias
  maps. A repair receives that same value unchanged plus deterministic
  validation errors; reconstructing it or changing catalog membership, ordering,
  or aliases between attempts is an invariant failure. No digest is added
  because this POC has no active-Run cross-process recovery or external digest
  consumer. Count and byte overflows follow their owning adapter's explicit
  failure or incomplete-completion path and never silently truncate canonical
  accepted support.
- Invoke every registered Specialist through one generic
  `execute_specialist` LangGraph node. One `Send` Task is one complete
  invocation of that parent-Graph node: it resolves the trusted Specialist
  definition, materializes bounded prior context, runs the full PydanticAI
  Specialist Agent asynchronously, and returns one `TaskOutcome`. The
  Specialist works autonomously inside that invocation, selecting its own
  eligible Skills and performing bounded multi-hop or internal fan-out/fan-in
  over read-only Tool calls. Expected Tool unavailability is an ordinary typed
  Tool result inside this invocation and cannot cancel successful sibling calls.
  Do not expand a
  Specialist into a variable-length parent-Graph branch in this POC; keeping
  each branch to one node preserves the dynamic batch barrier. The registry
  execution seam may adapt a complex Specialist to an internal runner without
  changing the parent Agent Graph. The LangGraph adapter constructs the
  minimal terminal `TaskOutcome`, updates Graph accounting separately, and sends
  operational diagnostics to telemetry; none of these platform records is the
  Specialist Agent's function signature. Tool bindings still delegate to the
  platform Tool Executor for typed Tool outcomes, Evidence, audit, and SSE
  events.
- The PydanticAI terminal output and the accepted platform result are two small,
  explicit types. `SpecialistFindingDraft(summary, evidence_ids)` is the complete
  model-authored output; it has no Data Gap field. PydanticAI validates that
  draft, then deterministic platform validators verify Evidence references and
  repairable domain constraints. After draft validation, the execution adapter
  validates current-attempt Tool metadata and automatically materializes
  exactly one canonical Data Gap for every unavailable Tool metadata record in
  that attempt. It then marks the attempt accepted and constructs
  `SpecialistResult(summary, evidence_ids, data_gaps)`. A canonical Data Gap
  contains its bounded
  sanitized requested-coverage label, stable allowlisted code, and only the
  self-contained provenance that remains useful after request-local metadata is
  discarded: opaque unavailability ID, registered Tool ID, logical source ID
  when applicable, and fixed observation time. The metadata record is also bound
  to the same Run, Task, and attempt, but is not model-visible or a new persisted
  Artifact. Missing, stale, cross-Run, cross-Task, cross-attempt, or
  duplicated-conflicting provenance is a fatal invariant violation; it never
  consumes an outer retry or becomes `TaskFailed`. Identical duplicate records
  are idempotent. Failed and abandoned attempts contribute no Data Gaps; an
  accepted unavailable outcome remains a gap even if a later fallback succeeds.
  A partial Finding remains successful and discloses that conservative record.
  Coordinator, Synthesis, and public disclosure receive a bounded `DataGapView`
  containing only requested coverage, stable code, and observation time; opaque
  IDs and internal Tool/source identities stay in accepted checkpoint state for
  validation and audit. Do not add a `coverage_key` or gap-resolution relation:
  this POC never clears an accepted gap.
  The LangGraph adapter wraps the resulting Specialist Result in the
  platform-owned `TaskOutcome`. A future Specialist may define task-specific
  typed output when a real machine consumer requires it, but the POC does not
  implement a generic payload extension framework, Result Contract registry,
  typed-edge model, target ports, or cross-Task schema inference.
- Each successful Tool that produces Evidence returns bounded model-visible
  content through PydanticAI `ToolReturn.return_value` and a typed Evidence
  envelope through application-only `ToolReturn.metadata`. An expected
  unavailable outcome instead returns a `ToolReturn` whose model-visible value
  is `ToolUnavailable` and whose application-only metadata is the provenance
  record used to derive its Data Gap; it contains no Evidence or Calculation
  Artifact. The Tool binding must normalize it before it can fail concurrent
  sibling execution. After a Specialist terminal Finding validates,
  `execute_specialist` extracts accepted-attempt metadata,
  verifies every referenced Evidence ID, stages the referenced IDs with the
  branch contribution, and places newly referenced bodies in a concurrency-safe
  cache owned by the active Request and injected through typed LangGraph runtime
  context. Tools never write directly to canonical accepted state.
- The body cache is not an eligibility authority and is not transactionally
  coupled to the branch's state update. Synthesis may use only the deterministic
  intersection of Evidence IDs in accepted successful Task outcomes and bodies
  resolvable from the current Request cache. Stable identical writes are
  idempotent; the same ID with different canonical content is an invariant
  failure. An orphan body without an accepted outcome reference is unreachable.
  Checkpoints, `Send` packets, and reducers contain Evidence IDs but never
  Evidence bodies. Missing bodies on a new-process continuation fail closed;
  transparent crash recovery is outside this POC.
- Before final synthesis, resolve each accepted Finding's Evidence IDs to a
  bounded, authorized representation containing the source metadata and excerpts
  needed to assess support; IDs alone are insufficient model context. The
  Synthesis Agent produces one concrete `FinancialResearchReport` whose
  canonical content is coherent Markdown containing inline Evidence and
  Calculation Artifact markers from that supplied catalog. Strict deterministic
  parsing rejects malformed, unknown, ineligible, stale, cross-Tenant, or
  cross-Run references and invalid Calculation Artifacts, then derives the
  existing public `CitationReference` values before publication. The POC does
  not add a model-authored `claims` array, generic output envelope, output
  registry, or parallel source list. Code does not claim to determine whether
  every factual statement has a citation, whether Evidence semantically entails
  adjacent prose, or whether the research is globally complete; those judgments
  remain prompt/evaluation concerns and groundedness remains advisory.
- Treat Evidence excerpts, Tool content, Specialist Results, and activated Skill
  references as untrusted data even after their identity is authorized. Typed
  parsing, fixed Tool registration, Research Scope checks, deterministic marker
  binding, and publication gates—not text instructions—control authority and
  execution. Content that asks an Agent to ignore policy, reveal hidden data,
  call another Tool, or forge a support marker cannot change those controls.
- Keep the complete reproducibility record for each Calculation Artifact in
  internal Run/audit state. When validated Markdown contains a Calculation
  placeholder, code resolves it to the eligible Artifact and renders the
  canonical formatted value plus only the concise disclosure needed to
  interpret it: human-readable method, unit, relevant period or as-of time, and
  material assumptions. Synthesis must not copy or retype the calculated value
  into that slot. Method version, normalized input references and hashes,
  complete audit inputs, and execution details remain internal. This guarantees
  that a value represented as a registered calculation exactly matches its
  Artifact; it does not claim that code can identify every uncited number or
  infer whether arbitrary model prose contains an unmarked calculation.
  Evidence markers alone produce public `CitationReference` values; a
  Calculation Artifact is derived support and must not be represented as source
  Evidence. Do not add `calculation_artifacts[]` to the public response until a
  real external machine consumer requires it.
- All Specialist Tasks use one prompt-visible brief containing a
  goal, stable references to completed prior results, and bounded declarative
  constraints derived from the immutable Research Scope. A Specialist never
  receives prior user/final-assistant Conversation pairs.
  They do not present Specialist Agents as functions or expose complete JSON
  schemas to the Coordinator Agent. The brief is the work assignment an autonomous
  Specialist receives, not its runtime function signature, and it does not copy
  complete upstream outcomes into the Coordinator Decision.
- Materialize a Specialist Task's ordinary upstream dependencies as
  deterministic, bounded context from its validated `context_task_ids`, ordered
  by stable round/dispatch/Task keys. Include only accepted result content and
  Evidence IDs; never attach every earlier Result, raw Tool payloads, internal
  model messages, or an arbitrary serialization of `TaskOutcome`.
- Resolve Specialist Agents from a small typed code registry. Each registered
  Specialist binds its actor factory and allowed Tool and Skill identities. The
  POC provides concrete mock registrations and a loader seam, but does not
  implement a configurable Agent catalog or an Agent Team DSL. Tenant policy
  may further restrict the registered catalog.
  The Coordinator Agent sees only compact Specialist descriptors describing
  what each Specialist can do; it does not see function-style Specialist input
  or output schemas.
- Resolve one Specialist invocation's callable business Tools as the intersection
  of registered Tools, trusted Tenant policy, the immutable Run Research Scope,
  and that Specialist's Tool allowlist. This set is fixed before the PydanticAI
  run begins. Tool adapters also enforce the Research Scope's server-owned data
  source and search constraints; model-supplied parameters may narrow but never
  remove or widen them. Activating a
  Skill supplies instructions, references, and Tool-use guidance only; it never
  grants, adds, or dynamically rebinds a Tool. Reject Skill activation if its
  declared required Tools fall outside the invocation's effective Tool set.
- Support Specialist-owned Skill discovery, activation, and references. Trusted
  Specialist registration defines Specialist-scoped Skills and a small set of
  Skills shared by all Specialists; shared Skills are not implicitly visible to the
  Coordinator Agent. Each Specialist chooses from its own effective eligible set
  based on the delegated Task goal and content. Expose at most the first 20
  eligible summaries in trusted registration order to one Specialist invocation;
  the POC does not implement Skill search or ranking. Activation fails closed unless the
  Skill is eligible for that Specialist and pins its name, version, and content
  hash for the Run. Any Tool names declared by the Skill are advisory or required
  capability checks against the already-fixed effective Tool set, never an
  authority source. Registry cache presence never grants activation authority.
  Skill scripts and assets are not executed.
- Provide only the mock registered business Tools exercised by the golden path:
  price series, fund holdings, fund reports, and company news. Preserve
  production-shaped provenance, time, unit, currency, status, retry, Evidence,
  and Calculation Artifact contracts.
- Provide only the simple registered calculations exercised by that path:
  period return, annualized volatility, and maximum drawdown. The LLM selects
  only allowed versioned methods and never generates executable formulas. A
  calculation Tool accepts trusted instrument, period, dataset, or Evidence
  references and resolves the actual series through the Tool Executor; it never
  accepts an LLM-authored raw price series as authoritative input.
- Return each successful calculation from PydanticAI as a `ToolReturn` whose
  concise `return_value` is model-visible and whose typed
  `CalculationArtifact` metadata is application-visible. Only after the same
  Specialist run produces a valid terminal Specialist Result may the
  `execute_specialist` adapter extract and validate those metadata records and
  stage them together with its single terminal Task Outcome in one branch
  contribution. Only the barrier may promote those records into an
  AcceptedBatch. Failed, cancelled, abandoned, outer-retried, and rejected-batch
  runs contribute no canonical Calculation Artifacts; every outer attempt
  starts a fresh PydanticAI run. Do not add an Artifact collector, repository,
  transaction layer, or IDs to `SpecialistResult` for this POC.
- Merge parallel Calculation Artifact updates by stable, content-bound Artifact
  ID. Identical duplicate writes are idempotent; the same ID with different
  content is an invariant failure. At the barrier, validate that every Artifact
  belongs to the accepted attempt of a successful Task, enforce aggregate
  limits without truncation, and assign aliases in stable Task/Artifact order,
  never completion-time order. Reuse the exact alias mapping for Synthesis
  repair and publication.
- Enforce Evidence eligibility, freshness, conflicts, Calculation Artifact
  validation, and inline support-marker referential integrity in code.
  Groundedness remains advisory in this POC.
- The financial POC's PydanticAI Synthesis output is the concrete
  `FinancialResearchReport` with one non-empty `markdown_report` field. After
  strict support-marker parsing, binding, and Calculation rendering, code
  prepends a fixed `Incomplete research` block whenever the frozen
  `IncompleteResearch` projection contains any signal. The block lists the bounded,
  Markdown-escaped objectives of accepted failed Tasks, each accepted Data Gap's
  requested-coverage label, the fixed insufficient-Evidence statement when
  applicable, and/or the standard structural-limit statement. The deterministic
  insufficient-Evidence answer uses this renderer exactly once rather than
  duplicating the statement. The block never includes
  Tool identity, Tool arguments, raw provider details, exceptions, retries, or
  counters. This post-gate Markdown—not the model candidate—is the canonical
  answer published through the existing `answer: string` and `citations[]` wire
  fields. When the projection is empty, code adds no completeness claim or
  disclosure block.
  Do not introduce a generic base class or payload wrapper before a second real
  domain demonstrates a shared machine-consumed contract. During Synthesis,
  emit progress only. Do not expose provider text deltas or any answer text
  until the complete report passes the deterministic publication gates. These
  token events are transport chunks of the approved report, not live provider
  tokens. Linear mode retains its existing streaming behavior.
- After the gates succeed, construct the canonical final response and run a
  dedicated `finalize_state` node that writes the response, derived citations, final
  assistant Message, and publication manifest without emitting answer content.
  Run the Graph with synchronous checkpoint durability. Only after that state
  update commits may a dedicated `publish` node read those canonical fields
  and emit `token`, `citations`, and `done`. The same state-before-first-answer
  rule applies to clarification and deterministic bounded-completion outcomes.
  A `finalize_state` checkpoint failure emits no answer frame and never enters
  publication.
- Publication is a transport projection of committed canonical state, not a
  second source of truth. The POC does not promise exactly-once SSE delivery,
  answer replay, or an atomic transaction spanning PostgreSQL and the network;
  it promises only that canonical state exists before the first answer frame.
  Absence of `done` does not imply absence of a committed canonical response.
- If strict Synthesis output or support-marker validation rejects the complete
  candidate, the Synthesis adapter owns one explicit bounded repair by the same
  Agent with deterministic validation errors and the unchanged support catalog.
  It makes at most two complete actor invocations with equivalent built-in
  output retry disabled. This is not a new Task, Coordination Round, or research
  step. If the repaired candidate still fails, fail the Run without publishing answer tokens,
  citations, a final answer, or an assistant Conversation Message. Never delete
  invalid markers and publish the remaining prose.
- Stream planning, Task, Tool, gate, warning, citation, and final progress through existing SSE event types. Do not expose chain-of-thought.
- Use the typed code-registered Specialist catalog with the minimum mock
  Specialist definitions needed by the golden fixture: market analysis and fund
  research. The latter may receive separate holdings/report and follow-up news
  Tasks in different Coordination Rounds. This roster is a fixture, not a
  platform constraint; Tenant policy may restrict the registered catalog.
  Deterministic calculations
  are registered Tools used inside the responsible Specialist, not top-level
  Tasks in the current POC.
- Keep retrieved chunks and other Evidence bodies in the request-owned
  runtime-context body cache, and keep raw Tool payloads request-local.
  Checkpoint Coordination Rounds, count/usage control state,
  active-batch staging metadata, immutable AcceptedBatch values containing Task
  Outcomes and bounded Calculation Artifact records, and stable Evidence
  identifiers.
- Configure the official `JsonPlusSerializer` explicitly as
  `JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=None,
  allowed_msgpack_modules=None)` and inject that instance into the shared
  `AsyncPostgresSaver` through `serde`; do not rely on an environment variable
  or constructor defaults for strict behavior. Persist application-owned values
  as JSON-native data and use only the already-approved framework Message types.
  At the request/Graph entry that consumes checkpoint-loaded state, validate all
  application-owned channel values through one application-owned typed
  checkpoint-state adapter before any node or Conversation projection uses
  them. An unknown custom type must never be reconstructed, and a value degraded
  by the serializer to a dict or constructor-argument representation must fail
  this typed boundary rather than be coerced into valid runtime state. Add an
  exact-symbol serializer allowlist only if a required type cannot be safely
  projected at this checkpoint boundary.

## Testing Decisions

- Prefer tests that assert externally observable contracts rather than node
  layout or private helper calls. Use `/v2/query/stream` with an Agent-configured
  Tenant as the primary seam, with application fake actors, deterministic mock
  Tools, the real LangGraph runtime, and checkpointed state. Keep pure
  unit/property seams only for reducer, projection, support-binding, calculation,
  and publication invariants that the HTTP seam cannot diagnose precisely.
- Preserve the existing application fakes because they make routing, limits,
  concurrent scheduling, checkpointing, and SSE behavior deterministic. Add a
  narrow PydanticAI-native contract suite using the official `TestModel`,
  `FunctionModel`, actor override, captured run messages, and
  `ALLOW_MODEL_REQUESTS=False` to prohibit accidental model requests. Do not
  rewrite all Graph tests around SDK test models.
- Cover one successful external golden path whose single request combines a
  cross-round dependency chain, concurrent fan-out with deterministic collect,
  Specialist delegation, and outcome-driven follow-up rounds. Exercise it through the real
  HTTP/SSE, LangGraph, PydanticAI actor, and PostgreSQL-checkpointer boundaries
  using fake models and deterministic mock Tools. Use a fixed as-of date and
  synthetic fund and benchmark IDs: the first batch concurrently delegates
  market analysis and fund holdings/disclosure research; the market Specialist
  performs registered calculations internally. After the barrier, a follow-up
  company-news Task explicitly selects the fund-research Result through
  `context_task_ids`. The next Coordinator Decision finishes and routes to
  synthesis. During the path, a scripted Specialist model selects and activates
  an eligible Skill before using its already-authorized business Tools; execute
  the real progressive Skill activation boundary rather than preloading the
  Skill into the Specialist prompt.
- Cover missing, stale, conflicting, cross-Tenant, malformed, duplicated, and
  failed Tool/Evidence/calculation outcomes. Malformed or unauthorized inline
  Evidence and Calculation markers, and markers resolving to invalid
  Calculation Artifacts, must never be published.
- Verify bounded coordination rounds, structural Task limits, actor-local work
  limits, Specialist permissions, approved Skill activation, deterministic
  staged contribution merging and AcceptedBatch ordering, retry idempotency, and
  additive SSE compatibility. Aggregate parallel model/Tool counts remain
  telemetry or stop-future-work assertions rather than atomic authorization
  assertions.
- Verify a mixed batch of successful and expected-failed Tasks reaches the
  barrier with exactly one outcome per Task. Expected failures appear to the
  Coordinator only as `TaskFailed(task_id)`; usage still contributes to separate
  Graph accounting. Verify the final code-owned disclosure names the bounded
  objective of each accepted failed Task and sets incomplete response metadata,
  even when Synthesis returns complete-sounding prose. Verify cancellation,
  authorization failures, reducer conflicts, and unexpected exceptions
  propagate instead of becoming failed outcomes, while a successful negative
  finding remains `TaskSucceeded`.
- Verify Tool-return Evidence metadata is invisible to the model, only bodies
  referenced by a validated successful Finding enter the request-owned cache,
  concurrent identical writes converge, conflicting bodies fail, and neither
  checkpoints nor persisted pending writes contain Evidence bodies. A
  new-process continuation with an accepted ID and empty body cache must fail
  closed as unsupported recovery rather than synthesize or return `TaskFailed`.
- Verify active-batch atomicity with real LangGraph pending-write behavior: a
  mixed success plus expected Task failure can be accepted; reversed completion
  order produces the same accepted value; external cancellation and fatal
  failures propagate without promoting an incomplete active batch.
- Treat serialized same-`thread_id` execution as an existing request-runtime
  contract, not a new Agent Graph behavior. Do not add competing same-thread
  requests, HTTP 409, advisory-lock lifecycle, or dedicated-session tests to this
  feature. Agent-mode tests use sequential Requests for one Conversation and may
  continue to verify that different Conversations execute independently.
- Verify two consecutive Runs in one Conversation inherit only complete
  Conversation Messages. Every reducer-backed Run-local channel is actually
  empty after initialization, every scalar is reset without an overwrite wrapper,
  and no actor can run before initialization completes.
- With deterministic counters, verify Task/Round exhaustion with Evidence
  invokes Synthesis and publishes an incomplete `done`, while exhaustion without
  Evidence publishes deterministic insufficient Evidence with
  `completion_status=incomplete`. Add a first-Decision `Finish` case and a
  successful accepted-result case with no eligible Evidence; both skip Synthesis,
  use `termination_reason=insufficient_evidence` when no other cause exists, and
  checkpoint the exact published answer. In contrast, an authoritative
  successful negative or empty query produces Evidence and may remain complete.
  Verify retries consume
  their actor-local limits, aggregate token/cost remains reporting-only, and
  cancellation or fatal failures do not become bounded-completion `done` events.
  Verify four accepted dispatch Decisions permit only `Finish` on the next
  Coordinator Decision; a fifth dispatch candidate is never sent. Exercise the
  exact boundary and one-over-boundary case for every numeric policy row,
  including cumulative allowances across Specialist outer attempts.
- Verify Agent-mode publication ordering: Synthesis progress may stream while
  the candidate is generated, but no answer token precedes support-marker and
  Calculation Artifact validation or the synchronous final-state checkpoint. On
  success, concatenated token events equal both the validated canonical Markdown
  and `done.answer`; on gate failure, no answer token, Citation event, or
  assistant Conversation Message is committed. A blocked or failed final-state
  saver must produce zero answer, citation, and done frames. Use controlled
  barriers or a fake saver rather than timing-sensitive races to verify the
  cancellation boundary: cancellation before the final commit leaves no final
  assistant Message or final frames; cancellation after commit but before
  publication leaves the canonical Message while delivering no final frames;
  cancellation during publication may deliver only a prefix without `done`,
  while the canonical Message remains complete and is not duplicated.
- Configure the golden fixture with at least one shared Skill and one
  Specialist-scoped Skill. Verify that Specialists see only the union of their
  scoped and shared Skill summaries, that the Coordinator sees neither catalog,
  that full instructions enter only the activating Specialist's current run,
  and that activation of an ineligible Skill fails closed. Skill activation must
  not change the invocation's effective Tool set.
- Verify that separate Linear-configured and Agent-configured Tenants use the
  same query route and cannot override mode through request input.
- Verify consecutive requests in one Agent-mode Conversation can execute different
  shapes—for example, fan-out/fan-in followed by a combined multi-hop Run.
- Add a consecutive-request follow-up fixture in which Query Understanding uses
  bounded Conversation pairs to resolve a pronoun or implicit benchmark into a
  self-contained standalone query. Verify that Coordinator and Synthesis receive
  that standalone query and selected Business Intent, Specialists receive only
  their Task-specific context, and none of those downstream actors receives the
  Conversation history. Verify an unknown Intent and every attempted Tool,
  Specialist, data-source, or filter expansion outside the resolved Research
  Scope fails closed before data access.
- Add a clarification-path fixture that verifies no Research Scope resolution,
  Coordinator, Specialist, or Synthesis call occurs; one normal `done` event is
  emitted with the structured clarification, matching answer, and no citations;
  no `IncompleteResearch` projection or block is created and response metadata
  contains neither `completion_status=incomplete` nor
  `termination_reason=insufficient_evidence`;
  the complete user/assistant pair is checkpointed before publication; and a
  later Request in the same Conversation resolves the answer from bounded
  Conversation history without resuming the prior Run.
- Add a pre-moderation terminal fixture that proves it never enters research
  completion, constructs no `IncompleteResearch` projection or block, and emits
  neither `completion_status=incomplete` nor
  `termination_reason=insufficient_evidence` merely because no Evidence exists.
- Verify the same Agent Graph builder can be assembled with fake actor and
  Specialist-registry dependencies without importing financial actor
  implementations into graph control flow. This proves the assembly seam; it
  does not require a second Team product model or configuration format.
- Add focused unit/property tests for Coordinator Decision and batch validation,
  concurrent dispatch and its barrier, cross-round result references, stable-ID
  outcome reducers, Evidence gates, the no-eligible-Evidence synthesis bypass,
  and registered calculation contracts. Calculation tests must reject
  model-authored raw series, contribute metadata only with a validated successful
  Specialist outcome, prevent failed/retried-attempt leakage, remain stable
  across parallel completion order, detect reducer collisions and aggregate
  overflow, preserve aliases across Synthesis repair, and verify that code—not
  Synthesis—renders each referenced Artifact's canonical value. Add an
  alternate-outcome coordination
  test in which the holdings Task returns no accepted finding or fails, and
  verify that the next decision differs from the successful path while the Run
  still terminates within the round bound.
- Add PydanticAI adapter contract tests that assert Tool registration and output
  schema, `ToolReturn` metadata isolation, `new_messages()` boundaries, usage
  extraction, and exact model-request trajectories. Use one fresh
  `capture_run_messages()` context per actor invocation or outer attempt; never
  share one context across multiple `run*` calls. For a successful invocation,
  retain its immutable `result.new_messages()`; for an invocation that raises,
  retain the messages from its dedicated capture context. The adapter-test
  harness may aggregate those per-invocation snapshots for assertions, but an
  abandoned attempt's messages or metadata must never enter an accepted result.
  Coordinator and Synthesis invalid output produces one actor request before
  the adapter-owned repair; expected POC Tool unavailability produces a typed
  result with no hidden
  `ToolFailed` adaptation turn. A multi-hop test proves the same PydanticAI run
  can call an allowed fallback and finish; the accepted Result still contains
  the adapter-derived Data Gap. An internal fan-out test proves one binding-owned
  Tool timeout neither cancels successful siblings nor prevents a successful
  partial Finding. Captured messages and metadata prove that every accepted
  unavailable Tool outcome creates exactly one same-attempt canonical gap even
  though `SpecialistFindingDraft` has no gap field, while abandoned attempts
  create none. Missing, cross-Run, cross-Task, cross-attempt, and conflicting
  unavailability provenance is fatal without outer retry or `TaskFailed`.
  Exercise every row of the closed failure-classification table using the exact
  current-V1 exception mapping above, including an unlisted or unknown exception
  that must remain fatal. The later V2 migration must replace these mapping
  tests without changing the stable categories. Run them against the frozen lock
  and assert `pydantic-ai==1.93.0`. Include origin-sensitive negative cases: an
  adapter-owned model timeout is retryable while the same exception escaping a
  business Tool is fatal; an `IncompleteToolCall` for the registered terminal
  output Tool is retryable while a truncated business-Tool call is fatal; a
  `ModelAPIError` with an `openai.APIConnectionError` cause and the allowlisted
  raw Google `httpx` transport signals are retryable while a bare
  `ModelAPIError` is fatal; and `ContentFilterError` remains fatal before its
  superclass is considered. Prove the same `httpx.ConnectError` and
  `httpx.TimeoutException` are fatal when they escape a business Tool, an
  Azure/OpenAI boundary, or an unregistered provider boundary. A
  `ModelAPIError` with an `openai.APIConnectionError` cause is likewise fatal
  outside the configured Azure/OpenAI model-call boundary. Exercise
  `ModelHTTPError` at `429`, `500`, and `599`
  as retryable and at `401`, `408`, and `600` as fatal. Exercise
  `UsageLimitExceeded` as count exhaustion only with request/Tool-call limits
  configured for that invocation and all PydanticAI token limits unset. Prove an
  oversized prior-Result context is rejected before `Send` and does not start or
  retry a Specialist;
  any post-validation size mismatch is fatal. A Specialist performs at most
  three outer attempts under one cumulative actor-local limit policy.
- Verify Calculation Artifact item and accepted-state caps at the barrier, plus
  Evidence body item and request-cache caps during Tool collection. A
  Calculation-state overflow leaves the active batch unaccepted and publishes
  the structural disclosure from earlier accepted state; its terminal barrier
  update leaves staging empty, clears the active manifest, and records
  `calculation_state_limit`. Exercise a collection whose canonical serialization
  fits exactly and one whose IDs/container framing make it exceed `1 MiB`. An Evidence-cache
  overflow stages one `TaskFailed` without deleting identical sibling bodies or
  making any orphan body eligible.
- Verify the code-owned incomplete block is present in the canonical assistant
  Message, token stream, and `done.answer` whenever any accepted Task failed,
  any accepted Data Gap exists, a structural limit ended research, or no eligible
  Evidence was established. Synthesis omission cannot remove it. With multiple
  causes, ordering and `termination_reason` are deterministic, the insufficient-
  Evidence statement appears at most once, and partial/limit causes retain their
  existing reason precedence. With no accepted incompleteness signal, no block
  is added and code does not claim global completeness.
- Verify Synthesis repair receives the same frozen `PreparedSynthesis` value and
  alias maps. Rebuilding or mutating it is an invariant failure, and no unused
  catalog digest exists in the POC contract.
- Add serializer tests that use the explicit strict constructor and shared saver:
  approved JSON-native checkpoint state and framework Messages round-trip;
  pickle payloads are rejected; unknown dataclass/Pydantic constructors are not
  reconstructed; and any downgraded dict or constructor-argument value is
  rejected by the typed checkpoint-state boundary. Add recursion tests that
  enumerate the compiled application-node route and inspect checkpoint history
  or `langgraph_step` to prove the longest legal route remains below 40.
  Specialist attempts, actor repairs, model requests, and Tool calls are asserted
  independently rather than counted as Graph steps. An intentional Graph loop
  must raise a recursion failure without publishing a successful terminal event.
- Add adversarial data fixtures in which Evidence, Tool output, Specialist
  Results, and Skill references contain instruction-like text. Assert that they
  cannot expand Research Scope, change Tool bindings, expose hidden state, alter
  routing, or forge eligible Evidence and Calculation markers.
- Organize 8–12 application-owned financial cases with Pydantic Evals. Keep
  deterministic trajectory, final-output, and single-step gate evaluators in
  ordinary CI. Run small real-provider Azure and Google canaries nightly or
  before release, with exact model identity and strict per-run limits; canaries
  are not ordinary PR blockers and do not replace deterministic tests.

## Out of Scope

- PydanticAI dependency-version migration and version-specific compatibility
  work. That work belongs to a separate migration spec and is not part of this
  Agent-pattern design or implementation scope.
- Real financial providers, production-grade financial methodology, arbitrary code execution, or an isolated sandbox.
- Financial Tools and calculations not exercised by the golden path, including
  instrument search, point quotes, sector membership, Sharpe ratio, support
  levels, Entry Zones, and general time-series aggregation.
- Skill scripts/assets, recursive Specialist creation, unconstrained ReAct loops, automatic trading, or order execution.
- Long-term Memory implementation, frontend changes, LangGraph Platform/Cloud, or removal of the legacy engine.
- `AgentTeamDefinition`, Team configuration/registry/persistence, and a workflow
  DSL.
- Checkpoint Resume, execution recovery, and human-in-the-loop interrupts.
- Externally mutating Tools, transfers, orders, writes, and strict business
  budgets or quotas. Because the platform is limited to read, fetch, and
  deterministic Calculation operations, it does not need Run-wide atomic
  reservation and settlement across parallel branches.
- Run-wide wall-clock termination and whole-Specialist elapsed-time limits. The
  POC uses per-call timeouts and structural/actor-local loop limits instead.
- Same-thread double-texting policies, advisory locking, lock-session monitoring,
  and fencing. The existing request boundary makes concurrent Requests for one
  `thread_id` impossible; revisit these only if that product invariant changes.
- Exactly-once or replayable SSE delivery and an atomic database/network
  publication transaction.
- Production use of the evolving PydanticAI Harness prompt-injection defender.
  Typed trust boundaries and adversarial tests are in scope; a pinned Harness
  defense-in-depth pilot follows the POC.
- The three additional complete financial scenarios and the shared
  50-concurrent-stream hardening gate; these follow the mixed-pattern core E2E.

## Future Extension Seams

- Preserve explicit Graph transitions from decision validation to dispatch,
  from deterministic batch collection to the next coordination round or
  synthesis, and from verified draft to publication. Future human review or
  clarification may be inserted at those boundaries without changing the Task
  execution contracts.
- This POC does not add dormant human-review nodes, approval state, interrupt
  payload contracts, or a Resume API.
- TODO: add a narrow `RegisteredReduction` join policy only when a real
  cross-Task business aggregation requires deterministic typed reduction.
  The current POC implements deterministic barrier, keyed collection, conflict
  detection, and canonical ordering before Agentic synthesis; it does not claim
  deterministic business reduction.

## Further Notes

- This spec starts after the `langgraph-orchestration-core` contracts and Linear
  graph are available. It intentionally does not prescribe or schedule the
  separate PydanticAI dependency migration.
- The root glossary and ADR-0001 through ADR-0007 are normative. ADR-0006
  supersedes the Agent-pattern planning and Calculation-placement clauses of
  ADR-0001.
- Architecture decisions required for deterministic CI and the core Graph are
  closed. Before real-provider or production rollout, operators must choose the
  Azure and Google canary deployments and cost ceiling, the owner and thresholds
  for the 8–12 financial evaluation cases, PostgreSQL checkpoint retention, and
  ingress streaming timeouts.
- The reviewed community-practice solution is the evidence record behind this
  revision. It does not itself represent completed dependency migration,
  production code, tests, or deployment hardening.
