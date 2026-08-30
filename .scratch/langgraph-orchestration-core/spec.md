# LangGraph Linear Core POC

Status: redesign-approved

## Goal

Replace the Flow Engine with the smallest clean-room LangGraph linear core while
preserving the production UI contract. The legacy engine remains available
during migration, but no Flow Engine execution abstraction is reused by v2.

## Public contract

- Implement only `POST /v2/query/stream` for a new query and
  `POST /v2/threads/{thread_id}/resume/stream` for explicit request-owned
  recovery. Do not implement a blocking `/v2/query` endpoint.
- Preserve the existing request fields, headers, SSE framing, event names and
  final response shape. Additive fields are allowed; existing fields may not be
  removed or retyped.
- A request begins Graph execution immediately in the receiving FastAPI
  instance. There is no queue, worker, detached runtime or background pickup.
- Return the stable public `conversation_id`, `thread_id` and `turn_id` needed
  by the UI. Do not introduce a public `run_id` merely to map to `thread_id`.
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

- LangGraph owns graph state, node order, streaming, checkpoints and Resume.
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
  ownership. Every query, history and Resume operation authorizes the requested
  Conversation before accessing its Messages or checkpoints.
- Store the stable LangGraph `thread_id` on the Conversation and resolve Resume
  through the Tenant/Subject-authorized Conversation; a client-supplied
  `thread_id` is never sufficient authorization by itself.
- A user `Message` represents the start of one Turn. Its `created_at` is the
  fixed Resume TTL anchor:

  ```text
  resume_deadline = user_message.created_at + configured_resume_ttl
  ```

  A retry or Resume never extends this deadline.
- Persist the user Message once at Turn creation and the assistant Message once
  after successful finalization. Store the same `turn_id` in Message records and
  LangGraph State and use it for idempotency.
- History initially uses the configured token-budgeted sliding window. LLM
  context compression is a separate policy to add later.
- Redis is an optional cache only. Redis loss or eviction cannot grant access,
  extend TTL or override PostgreSQL/Checkpoint state.

## Checkpoint and Resume

- Use the official shared PostgreSQL LangGraph checkpointer with a stable,
  server-scoped `thread_id` and `durability="sync"`.
- The latest durable checkpoint is authoritative for recoverable Graph state.
  The application does not maintain a duplicate checkpoint pointer.
- When the execution-owning SSE disconnects, cancel and await the active Graph
  iterator. Do not continue hidden work.
- Resume is an explicit new HTTP request. After Conversation/Subject
  authorization and Turn TTL validation, invoke the same Graph with the same
  `thread_id` and resume from its latest checkpoint. An incomplete node may run
  again.
- Bind the checkpoint state to `turn_id` and reject Resume if the checkpoint is
  complete, belongs to another Turn, or a newer Turn has started.
- Resume is computation recovery, not exact SSE token replay and not LangGraph
  time travel. A partially streamed Answer is discarded; Resume produces a new
  Answer stream from the durable node boundary.
- This read-oriented POC accepts the narrow possibility of overlapping old and
  resumed executions. Strict distributed single-flight/fencing is deferred
  until write-capable tools or a zero-overlap requirement exists.

## Persistence boundary

Keep only records with independent product value:

- Conversation and Message history;
- LangGraph PostgreSQL checkpoints;
- document/source Artifacts and provenance needed by the final response;
- BigQuery audit records;
- deterministic calculation inputs/results when that later capability is added.

Remove the application-owned `runs`, transport `events`, generic
`phase_results`, `cancellation_intents`, heartbeat, claim, lease,
`execution_epoch`, owner-instance, live follower and exact replay machinery.
Do not replace them with equivalent Redis state.

Provider/model calls may execute again after a node-boundary Resume. This POC
accepts repeated read/model cost. Any future irreversible side effect must be
idempotent at its domain boundary.

## Verification

- Contract tests compare v2 request and SSE behavior with the released UI/v1
  contract, including a real-time Answer stream and unchanged `done.answer`.
- Tests prove low groundedness and flagged post-moderation do not suppress,
  replace or alter the Answer or persisted assistant Message.
- Real PostgreSQL tests prove Conversation Tenant/Subject authorization,
  Message-based TTL, stable-thread checkpoint Resume across two application
  instances, and rejection of complete/stale/wrong-Turn Resume.
- Disconnect tests run through real Uvicorn and the deployment proxy boundary
  and prove the Graph iterator is cancelled and awaited.
- An opt-in load profile proves 50 simultaneous `/v2/query/stream` requests
  enter execution without application queueing. Capacity admission/rate limiting
  is not part of this slice.
- Graph node names and State schema must remain backward compatible across a
  rolling deployment while resumable checkpoints are retained.

## Deferred

- OpenTelemetry implementation, until separately approved.
- Admission capacity control and platform rate limiting.
- Remote cancel, exact SSE replay and historical-checkpoint time-travel APIs.
- Strict distributed single-flight/fencing.
- Agent patterns beyond Linear: fan-out/fan-in, multi-hop, map-reduce,
  supervisor/sub-agent, Skills orchestration and deterministic sandbox tools.
- LLM history compression, long-term memory and human-in-the-loop.
- Deletion of the legacy Flow Engine.

## Ticket status

Tickets 01-29 describe the implementation history of the first design. Their
completed checkboxes are not evidence that this redesigned spec is satisfied.
Do not implement unfinished Tickets 24-29 or use the old ticket definitions as
the target for further work. Tickets 30-50 are the current small, reviewable
expand-contract sequence for the removal and replacement work.
