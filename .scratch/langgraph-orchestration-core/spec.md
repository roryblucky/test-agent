# LangGraph Linear Core POC

Status: redesign-approved

## Goal

Replace the Flow Engine with the smallest clean-room LangGraph linear core while
preserving the production UI contract. The legacy engine remains available
during migration, but no Flow Engine execution abstraction is reused by v2.

## Public contract

- Implement only `POST /v2/query/stream`. Do not implement a blocking
  `/v2/query` endpoint or a public Resume endpoint.
- Preserve the existing request fields, headers, SSE framing, event names and
  final response shape. Additive fields are allowed; existing fields may not be
  removed or retyped.
- A request begins Graph execution immediately in the receiving FastAPI
  instance. There is no queue, worker, detached runtime or background pickup.
- Return the database-generated public `conversation_id` and stable logical
  `request_id` needed by the UI.
  Keep the checkpointer `thread_id` internal and do not introduce a public
  `run_id` merely to map to it.
- The API Gateway supplies trusted Tenant and Subject identity. The application
  must not trust client-overridable identity headers or expose a route that
  bypasses that boundary.

## Linear Graph

The fixed graph is:

```text
query
  -> pre-moderation
  -> question refinement
  -> retrieval
  -> reranking
  -> LLM answer
  -> groundedness
  -> post-moderation
  -> finalization
```

- LangGraph owns graph state, node order, streaming and checkpoints.
- PydanticAI owns LLM/model interaction, Agent configuration, Skills, Tools and
  structured output.
- Retrieval, reranking, moderation and groundedness use adapters over the
  existing provider abstractions and tenant-specific provider implementations.
- All new production code belongs in `app.langgraph_v2`. Do not import or wrap
  the legacy Flow Engine, flow configuration, handlers, `StepHandler`,
  `ExecutorContext`, coordinator or Agent handler.

## Answer publication

- Pre-moderation is the only blocking content gate in this slice.
- PydanticAI Answer deltas are forwarded to the UI through SSE in real time.
- After streaming completes, obtain the complete PydanticAI result and use it
  as the unchanged canonical Answer.
- Groundedness and post-moderation are advisory analysis phases. They never
  halt publication, replace the Answer or change `done.answer`.
- Persist the same original complete Answer as the assistant Message.
- Record groundedness and post-moderation assessments in the BigQuery audit
  path. Existing non-breaking response fields may continue to expose assessment
  metadata, but they must not affect the Answer.
- Never emit chain-of-thought.

## Conversation, history and authorization

- PostgreSQL `Conversation` is the durable authority for Tenant and Subject
  ownership. Every query and history operation authorizes the requested
  Conversation before accessing its Messages or checkpoints.
- Persist the Conversation's fixed `runtime_mode`. Derive LangGraph `thread_id`
  from Tenant plus Conversation UUID; never store or accept it as authorization
  input.
- Persist one user Message and at most one final assistant Message with the same
  stable logical `request_id`. Use `UNIQUE (conversation_id, request_id, role)`
  for retry idempotency and an atomically allocated per-Conversation `sequence`
  for history order. Do not store Turn identity or a separate idempotency key.
- History initially uses the configured token-budgeted sliding window. LLM
  context compression is a separate policy to add later.
- Redis is an optional cache only. Redis loss or eviction cannot grant access
  or override PostgreSQL/Checkpoint state.

## Checkpoint and disconnect

- Use the official shared PostgreSQL LangGraph checkpointer with a stable,
  server-scoped `thread_id` and `durability="sync"`.
- The application does not maintain a duplicate checkpoint pointer or a second
  execution journal.
- When the execution-owning SSE disconnects, cancel and await the active Graph
  iterator. Do not continue hidden work.
- There is no computation-recovery API in this stage. After disconnect or
  process failure, the client starts a new request and the Graph executes normally.
- This pre-release repository has no deployed or stamped v2 database. Before
  the first deployment, superseded v2 schema can therefore be consolidated in
  its original migration; no pre-release schema upgrade path is supported.

## Persistence boundary

Keep only records with independent product value:

- Conversation and Message history;
- LangGraph PostgreSQL checkpoints;
- lightweight document/source citation metadata needed by the final response;
- BigQuery audit records;
- deterministic calculation inputs/results when that later capability is added.

Remove the application-owned `runs`, transport `events`, generic
`phase_results`, `cancellation_intents`, heartbeat, claim, lease,
`execution_epoch`, owner-instance, live follower and exact replay machinery.
Do not replace them with equivalent Redis state.

Retrieved document chunks and raw provider payloads are request-local
`UntrackedValue` channels. They are not persisted in PostgreSQL checkpoints or
application tables.

## Verification

- Contract tests compare v2 request and SSE behavior with the released UI/v1
  contract, including a real-time Answer stream and unchanged `done.answer`.
- Tests prove low groundedness and flagged post-moderation do not suppress,
  replace or alter the Answer or persisted assistant Message.
- Real PostgreSQL tests prove Conversation Tenant/Subject authorization,
  request-paired Message idempotency, atomic sequence order, fixed runtime mode,
  and official checkpoint persistence.
- Disconnect tests run through real Uvicorn and the deployment proxy boundary
  and prove the Graph iterator is cancelled and awaited.
- An opt-in load profile proves 50 simultaneous `/v2/query/stream` requests
  enter execution without application queueing. Capacity admission/rate limiting
  is not part of this slice.

## Deferred

- Admission capacity control and platform rate limiting.
- Remote cancel, exact SSE replay and historical-checkpoint time-travel APIs.
- Computation recovery and human-in-the-loop Resume APIs.
- Strict distributed single-flight/fencing.
- Agent patterns beyond Linear: fan-out/fan-in, multi-hop, map-reduce,
  supervisor/sub-agent, Skills orchestration and deterministic sandbox tools.
- LLM history compression, long-term memory and human-in-the-loop.
- OpenTelemetry instrumentation and exporter configuration.
- Deletion of the legacy Flow Engine.

## Ticket status

Tickets 01-29 describe the implementation history of the first design. Their
completed checkboxes are not evidence that this redesigned spec is satisfied.
Do not implement unfinished Tickets 24-29 or use the old ticket definitions as
the target for further work. Tickets 30-50 are the current small, reviewable
expand-contract sequence for the removal and replacement work.
