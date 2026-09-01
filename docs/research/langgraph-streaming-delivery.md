# LangGraph streaming delivery for self-hosted FastAPI

Date: 2026-08-29

> Historical research note: the current POC no longer exposes Resume or
> computation recovery. Resume designs below are retained for a possible future
> phase and are not current implementation requirements.

## Question and conclusion

This note evaluates three delivery architectures for a self-hosted FastAPI application:

1. Run the graph while a second loop reads application-persisted PostgreSQL events.
2. Consume `graph.astream(...)` (or `astream_events(...)`) and forward the stream to SSE.
3. Add durable reconnect/replay or explicit checkpoint resume semantics.

For this repository's newly agreed contract—**an SSE disconnect interrupts the run; the client explicitly resumes it**—the recommended baseline is:

```text
FastAPI StreamingResponse
  -> graph.astream(..., stream_mode=["updates", "custom", ...])
  -> translate each LangGraph StreamPart to the existing SSE contract
  -> on disconnect/cancellation: stop the invocation and mark the Run interrupted
  -> explicit resume endpoint: invoke the graph from the durable checkpoint
```

The current “detached `ainvoke` producer + PostgreSQL event follower” is not the standard LangGraph OSS streaming path. It is a custom implementation of Agent Server-like background execution/rejoin semantics. That complexity was justified by the old “disconnect does not stop execution” requirement, but is no longer justified by the new contract.

PostgreSQL should remain authoritative for Runs, checkpoints, conversation history, artifacts, terminal results, and any events that compliance or the UI must replay. It should not be used as a 10 ms polling transport between a graph invocation and the same HTTP request.

For the narrower crash-recovery question, the user's model is correct: the active graph loop lives in the FastAPI process, so killing that process kills the invocation. Another FastAPI instance using the same graph definition, shared PostgreSQL checkpointer, and `thread_id` can start a new invocation from the latest durable checkpoint. LangGraph OSS does not require an application heartbeat or lease for that operation.

## What official LangGraph provides

### Native graph stream

LangGraph graphs expose `stream()` and `astream()` as iterators. The current v2 stream envelope has a stable `{type, ns, data}` shape and supports these modes ([official streaming guide](https://docs.langchain.com/oss/python/langgraph/streaming)):

| Mode | Intended signal |
| --- | --- |
| `updates` | Node state deltas after each step; the natural source for phase progress. |
| `values` | Full graph state after each step; usually too large for routine UI progress. |
| `messages` | LLM message/token chunks plus graph/model metadata. |
| `custom` | Application-defined progress or chunks emitted through `get_stream_writer()`. |
| `checkpoints` | Checkpoint snapshots; requires a checkpointer. |
| `tasks` | Task start/finish/error signals; requires a checkpointer. |
| `debug` | Broad diagnostic stream; not a normal product contract. |

For a model that is not a LangChain chat-model integration, the official guide explicitly recommends emitting its chunks through `custom` mode. That is the relevant path for preserving PydanticAI as the LLM abstraction: a PydanticAI actor can publish its streaming deltas through LangGraph's stream writer without moving model ownership into LangChain ([official arbitrary-LLM example](https://docs.langchain.com/oss/python/langgraph/streaming#use-with-any-llm)).

The newer typed event-streaming layer is built on the same Pregel stream modes and provides projections such as messages, values, output and custom extensions ([official event-streaming guide](https://docs.langchain.com/oss/python/langgraph/event-streaming)). It is useful when the application needs that richer protocol. For the current, already-released SSE contract, direct `astream()` plus a small translation layer is the simpler seam.

`astream_events()` is the generic LangChain Runnable callback-event surface. It exposes detailed `on_*` lifecycle events and can be useful for tracing or unusual callback filtering, but current LangGraph documentation presents `astream()` stream modes as the graph-native delivery API. The repository should not choose `astream_events()` merely to obtain phase or token streaming already covered by `updates`, `messages`, and `custom`.

### Persistence and resume

LangGraph checkpointers persist thread-scoped graph state for conversation continuity, human-in-the-loop, time travel, and fault recovery ([official persistence guide](https://docs.langchain.com/oss/python/langgraph/persistence)). Checkpoints are created at super-step boundaries, and completed writes within a super-step can be retained for recovery ([detailed persistence concepts](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints)).

Resume is separate from stream replay. A graph resumes using the same durable thread/checkpoint identity; a node interrupted in-flight may start again from its node boundary. The official interrupt documentation requires a durable checkpointer in production, the same `thread_id`, and re-invocation to resume ([official interrupt/resume guide](https://docs.langchain.com/oss/python/langgraph/interrupts)). Consequently, model/provider calls and other side effects must be idempotent or journaled if repeating them is unacceptable. LangGraph's Functional API guide likewise says to encapsulate API calls/side effects and make them idempotent because an incomplete task can execute again ([official Functional API guidance](https://docs.langchain.com/oss/python/langgraph/functional-api#idempotency)).

### Cross-instance crash recovery does not require heartbeat or lease

The checkpointer is shared state, not a running worker. Its job is to load and store checkpoints and pending writes. The official documentation says checkpointing provides fault tolerance by allowing a graph to restart from the last successful step, and that `thread_id` identifies the persisted thread state ([official checkpointer concepts](https://docs.langchain.com/oss/python/langgraph/checkpointers)). The official Functional API uses `None` plus the same thread configuration to resume a failed execution; the same continuation semantics are available through the graph streaming APIs ([official resuming example](https://docs.langchain.com/oss/python/langgraph/functional-api#resuming)).

Therefore, after an executing FastAPI instance really dies:

1. its in-memory coroutine and model stream are gone;
2. checkpoints written before the crash remain in PostgreSQL;
3. another instance can call `graph.astream(None, config={"configurable": {"thread_id": ...}}, durability="sync")`;
4. LangGraph loads the latest checkpoint and continues from its next node/super-step boundary.

`durability="sync"` is important for this contract because each checkpoint write completes before execution proceeds to the next step. The lower-latency asynchronous mode can leave a small window in which a process crash loses the most recent checkpoint write; exit-only durability does not support mid-run recovery ([official durability modes](https://docs.langchain.com/oss/python/langgraph/checkpointers#durability-modes)).

Neither the OSS checkpointer interface nor the PostgreSQL implementation defines a run owner, heartbeat, lease, or worker claim. The upstream PostgreSQL schema stores checkpoint identity, state, blobs, and task writes; its conflict handling is checkpoint persistence, not distributed single-flight coordination ([upstream PostgresSaver schema and SQL](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py)). The `AsyncPostgresSaver` lock is local to one saver object and protects cursor/connection use; it is not a cross-process per-thread lock ([upstream async saver](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py)).

**Conclusion:** `heartbeat_at`, lease renewal, and claim expiry are not required for this repository's explicit crash-resume path and should be removed unless the product later introduces background runs or automatic failover.

### The separate risk: two live invocations on one thread

A shared checkpointer does not prove that the old invocation is dead. If the browser loses its connection while the original FastAPI request is still executing, and a resume request reaches another instance immediately, both invocations can write the same thread concurrently. The checkpointer does not serialize those graph executions.

This is why LangGraph Agent Server exposes separate concurrent-run or “double texting” strategies such as reject, enqueue, interrupt, and rollback ([official double-texting documentation](https://docs.langchain.com/langsmith/double-texting)). Those are server-layer policies, not behavior supplied by the OSS `PostgresSaver`.

For this repository, the minimum appropriate policy is:

- keep execution request-owned;
- on normal SSE disconnect, cancel and await the graph/model stream before finishing request cleanup;
- let the UI keep its existing one-active-request rule;
- make provider calls and terminal writes idempotent because an unfinished node can run again;
- allow explicit resume from the shared checkpoint without heartbeat, lease, worker, or detached runtime.

There is an unavoidable network-partition edge case: the client may believe the connection is dead before the server observes cancellation. If the product requires a hard guarantee that two live invocations can never overlap, add a **small application-level per-thread single-flight guard**. A PostgreSQL session advisory lock is one possible self-releasing guard: a dead process loses its database session and therefore the lock, while a still-live process causes resume to fail fast with `409`. This is not a LangGraph best-practice requirement and has a real pool-capacity cost because a session is held for the run's duration. For the current POC contract, the simpler documented choice is to accept this narrow race rather than reintroduce heartbeat/lease machinery.

This recovery is not “time travel.” Official LangGraph terminology uses time travel for explicitly selecting a **prior** `checkpoint_id` to replay or fork; nodes after that selected checkpoint execute again ([official time-travel guide](https://docs.langchain.com/oss/python/langgraph/use-time-travel)). Continuing from the latest checkpoint after failure is better named **resume**, **fault recovery**, or **durable execution**.

### Rejoin/replay is a server feature, not the basic OSS stream

LangGraph Agent Server exposes explicit policies for disconnect behavior. Its API distinguishes `on_disconnect="cancel"` from keeping a run alive, and its frontend join/rejoin design requires both continued server execution and resumable stream state ([official cancel-on-disconnect guide](https://docs.langchain.com/langsmith/cancel-run#cancel-on-disconnect), [official join/rejoin guide](https://docs.langchain.com/oss/python/langchain/frontend/join-rejoin)). The latter guide explicitly states that join/rejoin requires LangGraph Agent Server.

This is useful as an architectural reference:

- **Cancel/interruption on disconnect**: the request owns the invocation; reconnect uses checkpoint resume.
- **Continue on disconnect**: the server must own background execution and retain/replay stream events; this needs a run service beyond `graph.astream()`.

The project selected the first policy. It should not keep a partial reimplementation of the second policy in the hot path.

## Comparison

| Property | A. `ainvoke` + PostgreSQL follower | B. Direct `astream` -> SSE | C. Background durable run + rejoin |
| --- | --- | --- | --- |
| Native OSS LangGraph path | No; application-defined | Yes | No; requires a server/run layer |
| First-event latency | Adds DB commit/read/wakeup latency | Lowest | Depends on server buffer/pub-sub |
| Backpressure | DB log decouples producer and client, but polling can overload DB | Natural: slow SSE consumer slows iteration; use a bounded buffer only if needed | Server buffer must enforce bounds/retention |
| Disconnect behavior | Can keep detached execution alive | Cancels/interrupts request-owned invocation | Configurable continue/cancel |
| Replay across instances | Yes, if every client-visible event is durably committed | No stream replay by itself; checkpoints resume execution | Yes, if server retains ordered events |
| Resume after failure | Checkpointer plus application journals | Checkpointer; add idempotency/journals where required | Server plus checkpointer |
| Complexity | High: duplicate event model, sequencing, polling, wakeups, fencing | Lowest | Highest, but warranted for long background runs |
| Fit for the newly agreed contract | Poor | Best | Overbuilt |

## Backpressure and delivery guarantees

Direct `astream()` couples graph progress to consumption. This is normally desirable for request-scoped SSE: network backpressure propagates instead of allowing unbounded in-memory output. If individual client writes can stall too long, use a **small bounded queue** and explicit send timeout; do not use an unbounded queue.

An application event log is justified when at least one of these is a real contract:

- a reconnect must replay every client-visible event, not merely resume graph computation;
- multiple instances or devices must attach to the same live run;
- an immutable compliance/audit record of particular domain events is required;
- downstream consumers need an ordered outbox independent of the HTTP connection.

Even then, persist **selected domain events** with stable IDs/sequences before exposing them, rather than storing full graph state snapshots or debug events as the UI protocol. The checkpointer and event log solve different problems: checkpoint state resumes computation; an outbox replays delivery.

For the selected explicit-resume contract, the simplest consistent guarantee is:

- events already sent may be repeated after resume unless the client supplies a cursor and the server deduplicates;
- graph computation resumes from the latest durable checkpoint;
- phase/provider results whose duplicate execution is unacceptable remain journaled/idempotent;
- no promise is made that an in-flight token stream can resume at the exact token.

## Assessment of this repository

The following findings are based on the current `app/langgraph_v2` implementation.

### 1. The hot path bypasses LangGraph streaming

`api.py::_persist_graph_result()` invokes `selected_graph.ainvoke(...)`; it does not consume `graph.astream(...)`. `api.py::_subscribe_to_run()` independently queries the application event table until the execution task completes. This means the public stream is derived from the custom PostgreSQL event journal, not from LangGraph's streaming protocol.

**Assessment:** change the request-owned execution path to consume `astream()` and map its `StreamPart`s directly to the established SSE types. Keep persistence at deliberate domain boundaries.

### 2. The initial stream polls PostgreSQL every 10 ms

`api.py::_subscribe_to_run()` repeatedly calls `_stream_unseen_events()`/`list_events()` and sleeps for `0.01` seconds. Unlike the reconnect follower, this path does not use `PersistedEventFollower`'s Redis wakeup plus bounded PostgreSQL fallback.

**Assessment:** this is unsuitable for the 50-concurrent-request baseline. In the no-event periods alone, 50 streams can approach 5,000 polling loops per second, with multiple database queries per loop. Direct `astream()` removes this database transport load.

### 3. “Token streaming” is post-hoc chunk pacing

`answer.py::run_answer()` waits for `actor.answer()` to return the complete structured answer, splits the completed string into sentence/codepoint chunks, commits all token events atomically, and the HTTP layer then delays chunks by the configured 200–500 ms interval.

**Assessment:** this is deterministic response chunking, not model token streaming. Keep it only if v1 contract parity explicitly requires paced synthetic chunks. Otherwise, use PydanticAI's streaming actor capability and emit deltas through LangGraph `custom` mode. Structured final output can still be validated and committed when complete.

### 4. Event data is duplicated in graph state and application tables

Each node appends event dictionaries to `TracerState.events`; phase commits also persist stable events into `langgraph_v2.events`; after `ainvoke()` the final state is walked again to persist/suppress duplicates.

**Assessment:** let graph state hold business state, not the full transport history. Emit progress through LangGraph stream modes, persist only replay/compliance events, and keep stable phase results for idempotent resume where they add value.

### 5. The phase-result journal still has value

The custom `phase_results` journal atomically records normalized provider results and stable events. This is stronger and more business-specific than checkpointing alone, especially for costly/non-deterministic financial providers and LLM calls.

**Assessment:** do not remove it mechanically. Re-evaluate it phase by phase as an idempotency/result-cache boundary. It should no longer be responsible for transporting every live SSE update.

### 6. Replay and resume should be separated in the API contract

Current code supports both a persisted event follower and an explicit checkpoint resume endpoint. Under the new contract, disconnect should stop the active invocation and transition the Run to `interrupted`; resume should start a new request-owned `astream()` from the checkpoint. A replay endpoint may remain for already committed events/history, but it must not imply that replay continues graph execution.

## Recommended target shape

```text
POST /v2/query/stream
  1. transactionally create Conversation/Message/Run
  2. build graph with Postgres checkpointer
  3. async for part in graph.astream(...):
       - updates -> phase progress SSE
       - custom -> PydanticAI/provider progress or token SSE
       - messages -> only if a LangChain model is used
       - terminal state -> final SSE + durable Run/message commit
  4. on request cancellation/disconnect:
       - cancel graph iteration
       - atomically mark Run interrupted

POST /v2/runs/{run_id}/resume/stream
  1. validate tenant/run ownership and the configured resume TTL
  2. load exact checkpoint
  3. consume graph.astream(None, config=checkpoint_config, durability="sync", ...)
  4. stream through the same adapter
```

Recommended product modes:

- `updates` for the nine phase transitions.
- `custom` for PydanticAI tokens and application progress.
- optionally `checkpoints`/`tasks` for internal diagnostics, not the public UI contract.
- avoid `values` in the public stream unless a full state snapshot is specifically needed.
- avoid `debug` in production responses.

## Implementation-review checklist

- [ ] Does `/v2/query/stream` iterate `graph.astream()` rather than `ainvoke()`?
- [ ] Does disconnect cancel the graph iterator and atomically mark the Run `interrupted`?
- [ ] Does `/resume/stream` reuse the same thread and exact durable checkpoint?
- [ ] Is crash recovery free of heartbeat, lease, claim-expiry, worker, and detached-runtime machinery?
- [ ] Is the accepted same-thread overlap race documented, or protected by a deliberately chosen single-flight guard?
- [ ] Are phase-result/provider side effects idempotent under node re-execution?
- [ ] Is `TracerState.events` removed or limited to actual business state needs?
- [ ] Are public SSE events mapped from a documented subset of `updates`/`custom`/`messages`?
- [ ] Are checkpoint/task/debug events excluded from the public contract by default?
- [ ] Is any remaining event table justified by replay, compliance, or downstream consumers?
- [ ] Does any buffer have a strict bound and a slow-client policy?
- [ ] Is real token streaming distinguished from post-hoc answer chunking in tests and docs?
- [ ] Do 50 concurrent idle/active streams avoid database polling amplification?
- [ ] Do disconnect/resume tests cover interruption inside every expensive phase?

## Primary sources

- [LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [LangGraph event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph checkpointers and durability modes](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [LangGraph interrupts and resume](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Functional API resuming](https://docs.langchain.com/oss/python/langgraph/functional-api#resuming)
- [LangGraph Functional API idempotency guidance](https://docs.langchain.com/oss/python/langgraph/functional-api#idempotency)
- [LangGraph time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)
- [LangGraph Agent Server: cancel on disconnect](https://docs.langchain.com/langsmith/cancel-run#cancel-on-disconnect)
- [LangGraph Agent Server: concurrent-run strategies](https://docs.langchain.com/langsmith/double-texting)
- [LangGraph Agent Server: join and rejoin](https://docs.langchain.com/oss/python/langchain/frontend/join-rejoin)
- [Upstream PostgresSaver schema and SQL](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py)
- [Upstream AsyncPostgresSaver implementation](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py)
- [Official Python SDK run options (`stream_resumable`, `on_disconnect`)](https://github.com/langchain-ai/langgraph/blob/main/libs/sdk-py/langgraph_sdk/_async/runs.py)
- [Official runtime source: checkpoint re-entry semantics](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_loop.py)
- [Maintainer recommendation for subgraph streaming via `astream(..., subgraphs=True)`](https://github.com/langchain-ai/langgraph/issues/3772#issuecomment-2715638273)

## Architecture audit: table deletion and explicit resume

Date: 2026-08-29

This section audits the later, smaller proposal:

- self-hosted, multi-instance FastAPI;
- request-owned SSE and real PydanticAI output streaming;
- shared `AsyncPostgresSaver`;
- explicit user-triggered resume;
- no SSE token replay;
- delete application-owned `runs`, `events`, and `cancellation_intents` tables;
- no heartbeat, claim, lease, execution epoch, or remote cancel endpoint;
- API Gateway supplies a trusted `subject_id` header;
- a public conversation/thread identifier is returned to the UI.

Where it conflicts with earlier sections of this note, **this audit supersedes the earlier recommendation to retain an application Run record**.

### Overall verdict

Deleting all three application tables is sound for the agreed POC contract, but only if the remaining responsibilities are placed deliberately:

| Proposed deletion | Verdict | Required replacement or accepted limitation |
| --- | --- | --- |
| `langgraph_v2.events` | Delete | The product does not promise exact SSE replay. Checkpoints resume computation, not token delivery. Preserve provenance/compliance data in Messages, Artifacts, and selected domain records instead of transport events. |
| `langgraph_v2.cancellation_intents` | Delete | Cancellation is local to the request that owns the graph. There is no cross-device, administrator, or detached-run cancellation requirement. |
| `langgraph_v2.runs` | Delete | Keep a stable `turn_id` in Graph State, Messages, and any retained PhaseResult journal. Persist conversation ownership in the existing Conversation model. Do not replace the table with a second Run-like Redis model. |
| heartbeat / claim / lease | Delete | LangGraph OSS crash recovery does not require them. Another instance reads the shared PostgreSQL checkpoint. |
| `execution_epoch` | Delete for this POC | This explicitly accepts a narrow overlap/re-execution race. All externally visible commits and non-repeatable side effects must be idempotent. |
| cancel endpoint | Delete under current scope | Browser abort closes the owned stream. Restore an endpoint only if cancellation must originate from another HTTP connection or actor. |

The core cross-instance recovery claim is supported by LangGraph's persistence model: `thread_id` identifies a checkpoint sequence, and `AsyncPostgresSaver` loads the latest checkpoint for that thread when no `checkpoint_id` is supplied ([LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [upstream `AsyncPostgresSaver`](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py)). The new instance must use the same PostgreSQL database, a stable thread identity, and a Graph/State definition compatible with the saved checkpoint.

### Mandatory items before deployment

The following are not optional refinements. They are correctness or security requirements created by the simplified design.

#### 1. Treat Gateway identity as a real trust boundary

It is acceptable for the API Gateway to provide `tenant_id` and `subject_id` in headers, provided that:

- the Gateway authenticates the caller, removes any client-supplied copies of those headers, and writes authoritative values;
- FastAPI cannot be reached through a network path that bypasses that enforcement, or it independently authenticates such traffic;
- Tenant and Subject never come from the request body or public thread ID;
- every query, resume, history, artifact, message, and checkpoint access performs object-level authorization.

An unpredictable thread UUID is not authorization. OWASP requires every endpoint that receives an object identifier to verify the authenticated caller's access to that object ([OWASP API1:2023 Broken Object Level Authorization](https://owasp.org/API-Security/editions/2023/en/0xa1-broken-object-level-authorization/)).

#### 2. Separate the public conversation ID from the internal checkpoint key

The UI may receive an opaque public `conversation_id` and use it in the resume URL. The server should derive or look up the internal checkpointer `thread_id` only after authorization, for example from a stable namespace containing the Tenant and Conversation. It must not pass an unscoped client-controlled identifier directly to the shared checkpointer.

`thread_id` is LangGraph's persistent cursor and primary lookup key, not a tenant-aware access-control primitive ([LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)).

#### 3. Persist conversation ownership

Because the product includes a durable chat-history sidebar, `owner_subject_id` (or an equivalent ACL/membership relation) belongs in the durable Conversation domain model. This is mandatory independently of resume. A two-hour Redis key cannot authorize opening or continuing a historical conversation weeks later.

The minimum durable authorization query is therefore equivalent to:

```text
(tenant_id, conversation_id, owner_subject_id) -> authorized Conversation
```

Shared conversations can later replace the single owner with a membership/ACL model without changing LangGraph checkpoint tables.

#### 4. Bind Resume to the interrupted Turn, not only the Thread

A thread accumulates multiple turns/runs. Resuming "whatever is latest for this thread" is safe only if no later turn has begun. Retain an application `turn_id` in:

- Graph State;
- the durable user Message;
- the assistant Message idempotency key;
- any retained PhaseResult idempotency key;
- the temporary resume grant, if Redis is used.

Before resuming, authorize the Conversation, fetch the latest snapshot, and verify that its `turn_id` is the expected interrupted turn and that it has pending work. If another turn is already current or the graph is complete, reject the resume instead of executing an unrelated latest checkpoint.

For normal fault recovery, the latest checkpoint is the correct target. Supplying a prior `checkpoint_id` intentionally replays/forks historical state and is LangGraph time travel, a different product capability ([LangGraph time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)). Resume should call the graph with `None` input and the existing thread configuration; providing a new query would start/merge new input rather than simply continue the checkpoint ([LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [upstream runtime loop](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_loop.py)).

#### 5. Make cancellation cooperative and test it through the real network stack

Request-owned `StreamingResponse` is aligned with Starlette. Depending on ASGI version, Starlette either races a disconnect listener against the body stream and cancels the task group, or detects a failed send and raises `ClientDisconnect` ([Starlette `StreamingResponse` source](https://github.com/Kludex/starlette/blob/main/starlette/responses.py)). This does not make browser disconnect an instantaneous distributed fact.

The SSE generator must:

- use `try/finally` to close and await the LangGraph and PydanticAI iterators;
- propagate cancellation instead of swallowing it;
- avoid detached tasks that outlive the request;
- use async/cooperatively cancellable provider clients and explicit timeouts;
- commit no partial assistant Message when the answer stream is cancelled.

This behavior must be integration-tested with the pinned Starlette/Uvicorn versions and the actual enterprise Gateway/proxy. A unit test that merely cancels a Python task is insufficient. Uvicorn documents disconnect delivery and write-flow backpressure, but a proxy or network partition can delay server observation ([Uvicorn server behavior](https://github.com/Kludex/uvicorn/blob/main/docs/server-behavior.md)).

#### 6. Use synchronous checkpoint durability for the stated crash-recovery guarantee

`durability="sync"` waits for the checkpoint write before proceeding to the next step. `async` has a small crash-loss window and `exit` does not provide mid-run recovery. Because this product explicitly requires process/power-failure recovery, `sync` is the appropriate mandatory choice despite its latency cost ([LangGraph durability modes](https://docs.langchain.com/oss/python/langgraph/checkpointers#durability-modes)).

Recovery is at super-step/node boundaries, not at an arbitrary Python line or token. An Answer node interrupted mid-stream can execute again from the node boundary; the new stream replaces the abandoned partial answer ([LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)).

#### 7. Make repeated nodes and side effects safe

LangGraph Graph API nodes can re-run from their beginning after interruption. Therefore:

- pure reads and LLM calls may repeat, with duplicate latency/cost accepted;
- deterministic calculations must re-run from the same durable inputs;
- assistant Message writes require a stable uniqueness/idempotency key based on Tenant + Turn + Role;
- non-repeatable tool actions, external writes, orders, notifications, or charges require provider idempotency keys or an application journal/outbox;
- a PhaseResult journal should be retained only for phases where avoiding a repeated expensive or non-repeatable operation is valuable, not as an SSE transport.

This is an explicit LangGraph durability requirement, not project-specific ceremony ([LangGraph Functional API idempotency guidance](https://docs.langchain.com/oss/python/langgraph/functional-api#idempotency), [LangGraph backward compatibility and node re-entry](https://docs.langchain.com/oss/python/langgraph/backward-compatibility)).

#### 8. Define a same-thread concurrency policy

LangGraph OSS and `AsyncPostgresSaver` do not provide a distributed per-thread execution lock. The saver object's `asyncio.Lock` is process-local. Agent Server addresses this separately with reject, enqueue, interrupt, and rollback policies; those policies are explicitly not part of LangGraph OSS ([LangGraph double texting](https://docs.langchain.com/langsmith/double-texting), [upstream `AsyncPostgresSaver`](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py)).

For the POC, the documented policy may be:

```text
one active request per UI + reject a mismatched Turn on resume
+ accept a narrow network-partition overlap
+ idempotent durable commits and side effects
```

That makes deleting `execution_epoch` defensible, but it is an accepted at-least-once/re-execution risk, not a strict single-flight guarantee. Before enabling write-capable financial tools, or if multiple tabs/devices must be supported safely, a server-side `reject`/single-flight mechanism becomes mandatory. A plain Redis TTL lock is not a strict substitute: Redis's own locking guidance describes bounded validity, failover races, and the need for fencing tokens when stale holders can write ([Redis distributed locks](https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/)).

#### 9. Preserve Graph and State compatibility across deployments

LangGraph resumes a checkpoint with the latest deployed Graph; it does not pin the thread to the code version that created it. Removing or renaming a pending node, deleting/renaming a State key, or making an optional field required can break resume. The official guidance is to add optional/defaulted fields, deprecate before removal, and record a `flow_version` in state when business behavior must branch ([LangGraph backward compatibility](https://docs.langchain.com/oss/python/langgraph/backward-compatibility)).

This is mandatory for rolling multi-instance deployments. Compatibility lasts longer than the two-hour failed-turn resume TTL if old Conversations can be continued from history. Deployment tests must load representative old PostgreSQL checkpoints using the new Graph.

#### 10. Operate the PostgreSQL checkpointer as production state

Production setup must run the official checkpointer schema setup/migrations, use the supported asynchronous connection configuration, and use safe serialization settings. The upstream PostgreSQL package documents `setup()`, connection requirements, and strict MessagePack/allowed-module controls ([LangGraph PostgreSQL checkpointer README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/README.md)).

Checkpoint retention must align with both products:

- failed-turn Resume TTL may be one or two hours;
- historical conversation continuation may require checkpoints for much longer, unless state can be reconstructed from durable Messages and summaries.

Resume authorization expiry must not be confused with checkpoint deletion.

### Redis assessment

Redis can hold a temporary value such as:

```text
resume:{tenant_id}:{subject_id}:{conversation_id} -> turn_id  EX 7200
```

but this is only safe as a **fail-closed, best-effort availability mechanism**:

- expiration is supported directly ([Redis `EXPIRE`](https://redis.io/docs/latest/commands/expire/));
- a configured eviction policy may remove the key before its TTL ([Redis key eviction](https://redis.io/docs/latest/develop/reference/eviction/));
- RDB, AOF, and replication have explicit data-loss windows depending on configuration ([Redis persistence](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/), [Redis replication](https://redis.io/docs/latest/operate/oss_and_stack/management/replication/)).

Losing the key must deny Resume; it must never fall back to trusting the public thread ID. If the product promises that every authorized user can resume throughout the complete configured one-to-two-hour window, then one of these is mandatory:

1. store owner and the Turn's resume-expiry anchor durably in the existing Conversation/Message domain schema; or
2. operate Redis with an explicit persistence, high-availability, capacity, and no-premature-eviction SLA.

For this repository, option 1 is simpler. Conversation ownership is already durably required for history, and the user Message's `created_at` can anchor the Turn TTL. It avoids making Redis a new availability dependency and still does **not** require a `runs` table. Redis remains optional as a cache or single-use resume token, not the source of truth for ownership.

Redis itself must remain private, authenticated with ACLs, and protected with TLS where traffic crosses an untrusted network ([Redis security](https://redis.io/docs/latest/operate/oss_and_stack/management/security/)).

### Answer streaming verdict

The Answer must use the real PydanticAI model stream, forward text deltas through LangGraph `custom` streaming, and accumulate the complete text only for final validation/persistence. PydanticAI's event stream exposes model events while its final result arrives at run completion; abandoning the context cancels the background/model stream ([PydanticAI Agent streaming](https://github.com/pydantic/pydantic-ai/blob/main/docs/agent.md), [PydanticAI streamed output](https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md)).

The public contract is:

```text
partial Answer token SSE: ephemeral and replaceable
completed Answer Message/PhaseResult: durable and idempotent
disconnect: no exact token replay
resume: node-boundary re-execution and a replacement Answer stream
```

Therefore the Event journal remains unnecessary. If regulatory policy later requires an immutable trace of selected source/provenance decisions, add a domain audit/outbox record for those decisions; do not restore a generic copy of every transport token.

#### Mandatory publication-gate decision

Real model deltas cannot simultaneously be public final-answer tokens and be
blocked by the later `groundedness` and `post_moderation` nodes. Once a delta has
crossed SSE, a later safe-answer replacement cannot undo that disclosure. The
current implementation confirms that post-moderation is a publication gate: a
flagged result replaces the answer with `SAFE_MODERATION_MESSAGE`
([`post_moderation.py`](../../app/langgraph_v2/post_moderation.py)), while the
Graph runs groundedness and post-moderation only after Answer
([`graph.py`](../../app/langgraph_v2/graph.py)).

One of these contracts must be chosen explicitly before implementation:

1. **Gated final answer (recommended for the stated financial/compliance
   audience):** consume the real PydanticAI stream internally, run groundedness
   and post-moderation on the completed result, then publish only the validated
   answer. SSE remains the transport, but publication is not raw model-token
   latency.
2. **Public draft stream:** publish raw deltas as explicitly non-final draft
   events, then replace/confirm them after the gates. This requires a new UI
   contract and accepts that ungrounded or later-flagged text was already shown.
3. **Incrementally gated stream:** buffer and moderate bounded chunks before
   publication. This can reduce moderation exposure but does not make whole-
   answer groundedness available before each chunk and materially increases the
   design scope.

Calling raw deltas the final Answer while retaining post-moderation as a blocking
gate is internally inconsistent and is not an acceptable implicit choice.

When PydanticAI tools are enabled, raw `run_stream_events()` text must also not be
forwarded indiscriminately: the stream contains events across model/tool turns,
and the final run result arrives only at completion. The implementation must
distinguish final-output text from intermediate model text and tool activity
([PydanticAI Agent streaming](https://github.com/pydantic/pydantic-ai/blob/main/docs/agent.md)).

### Generic PhaseResult journal assessment

A generic PhaseResult table is not required by LangGraph. With a production
checkpointer, completed super-steps and successful pending writes are already
durable. An interrupted node may run again; external side effects must therefore
be idempotent regardless of whether a generic result cache exists
([LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence),
[LangGraph idempotency guidance](https://docs.langchain.com/oss/python/langgraph/functional-api#idempotency)).

For this POC's read-only retriever/reranker/moderation/LLM calls, the simplest
design is to remove the generic `phase_results` journal together with its Run
foreign key and accept repeat cost for an interrupted node. Keep durable domain
records that have independent product value—Messages, source/provenance
Artifacts, and deterministic calculation inputs/results. If a future phase has
a non-repeatable external write or an explicit provider-result reuse requirement,
add an idempotency/outbox record at that domain seam rather than restoring one
cross-cutting journal for every phase.

### Linear order correction

The current source and spec execute pre-moderation before question refinement:

```text
query -> pre_moderation -> question_refinement -> retrieval -> reranking
      -> answer -> groundedness -> post_moderation -> finalization
```

That matches the later product correction that moderation must run first and
prevents rejected input from being sent to refinement/retrieval/model providers.
Any plan that places pre-moderation after reranking is stale and must be corrected
before implementation.

### Final mandatory/optional boundary

Mandatory now:

- trusted Gateway header boundary and object-level Tenant + Subject authorization;
- durable Conversation ownership;
- stable, server-scoped checkpoint thread identity;
- Turn-bound Resume validation;
- shared production PostgreSQL checkpointer with `durability="sync"`;
- real iterator cleanup/cancellation behavior tested through Uvicorn and the Gateway;
- idempotent terminal writes and every non-repeatable side effect;
- an explicit same-thread concurrency policy and documented POC overlap limitation;
- backward-compatible Graph/State deployment discipline;
- checkpoint security, migration, retention, and serializer configuration.
- an explicit Answer publication contract that does not silently bypass
  groundedness/post-moderation gates.

Optional under the agreed POC scope:

- application `runs`, `events`, and `cancellation_intents` tables;
- heartbeat, claim, lease, owner instance, and execution epoch;
- Redis resume grant when durable Conversation/Turn records enforce authorization and TTL;
- remote cancel endpoint;
- exact SSE replay;
- explicit historical `checkpoint_id` / time-travel UI;
- strict distributed single-flight, until zero-overlap or write-capable tools become a requirement.

## Addendum: accepted Redis authority and PydanticAI final-result semantics

This addendum records the narrower product decisions made after the review above.

### Redis-only Resume grant: accepted, with one required configuration check

For the agreed scope, Redis may be the sole authority for the temporary Resume
grant. A duplicate PostgreSQL authority is **optional**, not mandatory, provided
that a missing Redis grant fails closed and the product accepts loss of Resume
availability if the managed Redis service loses that key. The grant must bind
`tenant_id`, trusted Gateway `subject_id`, conversation/thread identity, and
`turn_id`.

The expiry is a fixed deadline derived from Turn creation:

```text
resume_deadline = turn_created_at + configured_resume_ttl
```

Create the grant with that absolute deadline; Query retries and Resume attempts
must not extend it. Authorization to access historical Conversations remains a
separate durable domain concern and is not delegated to this two-hour grant.

The exact Google Cloud product and configuration still need to be verified.
Memorystore for Redis Cluster is regional; its cross-region topology uses one
read-write primary and asynchronously replicated read-only secondaries. Google
warns that unplanned in-region failover can lose acknowledged writes and that a
cross-region disaster-recovery promotion can lose data that has not caught up
([cluster HA](https://docs.cloud.google.com/memorystore/docs/cluster/ha-and-replicas),
[cross-region replication](https://docs.cloud.google.com/memorystore/docs/cluster/about-cross-region-replication)).
These are accepted availability limits for this scope, rather than reasons to
add a `runs` table.

One caveat is independent of HA and therefore remains mandatory if the product
expects a grant to survive until its TTL: Redis Cluster's documented default
`maxmemory-policy` is `volatile-lru`, which makes expiring Resume keys eligible
for eviction before their deadline. Configure `noeviction`, reserve and monitor
memory headroom, and treat a failed grant write as failure to create a resumable
Turn; do not return Resume capability before the write succeeds
([supported instance configurations](https://docs.cloud.google.com/memorystore/docs/cluster/supported-instance-configurations)).
If the product instead accepts premature false-denial under memory pressure,
`noeviction` becomes optional, but that weaker contract must be explicit.

Google recommends combining HA with persistence for best durability. Redis
Cluster supports either AOF or RDB persistence, with documented loss/staleness
windows; neither changes asynchronous cross-region replication into a zero-loss
commit protocol
([persistence overview](https://docs.cloud.google.com/memorystore/docs/cluster/persistence-overview)).
This does not make PostgreSQL duplication mandatory under the accepted
fail-closed managed-Redis contract.

### PydanticAI complete result after streaming: yes, but not retroactive gating

PydanticAI does provide the completed, validated output after streaming:

- `run_stream_events()` ends with `AgentRunResultEvent`, whose `result.output`
  is the final run output;
- `run_stream()` returns `StreamedRunResult`, whose `await get_output()`
  consumes the complete response, validates it, and returns the final output;
- `agent.iter()` exposes `agent_run.result` once the run reaches `End`.

There is no general `agent.result()` call. In the repository's locked
PydanticAI 1.93.0, the precise APIs are `AgentRunResultEvent.result`,
`StreamedRunResult.get_output()`, and `AgentRun.result`. The official API docs
describe the same completed-result surfaces
([agent streaming](https://pydantic.dev/docs/ai/core-concepts/agent/#running-agents),
[streamed result API](https://pydantic.dev/docs/ai/api/pydantic-ai/result/#pydantic_ai.result.StreamedRunResult),
[run result event](https://pydantic.dev/docs/ai/api/pydantic-ai/run/#pydantic_ai.run.AgentRunResultEvent)).

This solves accumulation and final-result validation; it does **not** solve a
publication gate after raw deltas have already crossed SSE. Groundedness and
post-moderation can inspect `result.output`, but they cannot retract text already
rendered by the browser. PydanticAI also documents that
`stream_text(delta=True)` yields raw deltas and skips result validators, while
`stream_output()` applies partial validation to accumulated snapshots and full
validation to the final output
([streamed output semantics](https://pydantic.dev/docs/ai/core-concepts/output/#streamed-results)).

Therefore the supported gated design is to consume the real model stream into a
server-side buffer, obtain the final PydanticAI result, run groundedness and
post-moderation, and only then publish the accepted Answer. If the raw deltas are
published immediately, the contract is necessarily a public Draft stream and
the later gates can only confirm or replace it; obtaining the final result does
not change that disclosure boundary.

## Final product decision: durable authorization and advisory analysis

This section supersedes the Redis-only Resume grant and gated-publication
recommendations immediately above.

### Durable Resume authority

Resume authorization and expiry use existing durable domain/checkpoint data:

- `Conversation` owns the trusted Tenant and Subject authorization relation;
- the user `Message` identifies the Turn and its `created_at` is the fixed TTL
  anchor;
- the latest shared PostgreSQL LangGraph checkpoint determines whether that Turn
  has recoverable pending work;
- Redis is an optional cache and is never authoritative.

The deadline is always:

```text
resume_deadline = user_message.created_at + configured_resume_ttl
```

Resume and retry do not renew it. The latest checkpoint timestamp must not be
used as the TTL anchor because checkpoint progress would silently extend the
product window. No application `runs` table, Redis grant or duplicate checkpoint
pointer is required for these responsibilities.

### Real-time Answer is the canonical Answer

The selected compatibility contract is real-time publication, not a later
publication gate:

- pre-moderation remains blocking;
- PydanticAI Answer deltas are public SSE Answer tokens as they arrive;
- the completed PydanticAI output remains the canonical Answer;
- groundedness and post-moderation are advisory and are recorded through the
  BigQuery audit path;
- neither advisory result may halt publication, replace the Answer, change
  `done.answer`, or change the assistant Message persisted in history.

This deliberately accepts that later analysis can flag content already shown.
That is the established Flow Engine product contract to preserve. It removes the
earlier requirement to buffer the Answer until groundedness/post-moderation pass
and removes the current v2 safe-answer replacement behavior.

### Resulting minimal architecture

The request-owned FastAPI invocation streams LangGraph/PydanticAI output
directly. A disconnect cancels and awaits that invocation. Explicit Resume
authorizes the Conversation and Turn, checks the Message-derived TTL, and invokes
the same Graph/thread from the latest durable checkpoint. Exact token replay,
application Run/Event journals, heartbeat/claim/lease/epoch machinery, detached
runtime and Redis authority are outside this design.
