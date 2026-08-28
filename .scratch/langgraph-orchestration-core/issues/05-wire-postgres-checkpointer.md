# 05: Wire the fenced PostgreSQL LangGraph checkpointer

**What to build:** Connect the minimal graph to the official async PostgreSQL checkpointer using an execution-epoch namespace, and persist the exact authoritative checkpoint identity on the application Run.

**Blocked by:** 04: Add fenced direct-execution claims.

**Status:** completed

- [x] `langgraph-checkpoint-postgres` and its PostgreSQL driver are pinned direct root dependencies and `AsyncPostgresSaver` is wired through `app.langgraph_v2`.
- [x] A documented collision-free encoding maps Tenant + Conversation to `thread_id` and Tenant + Run + `execution_epoch` to `checkpoint_ns`; the Run records the exact latest `checkpoint_id` and namespace.
- [x] The supported first-use setup/migration path runs through documented deployment or application-lifespan setup and works against the disposable PostgreSQL fixture.
- [x] A test proves a committed minimal graph checkpoint can be read by a fresh checkpointer instance without editing LangGraph-owned tables.
- [x] A cross-Tenant lookup cannot resolve another Tenant's checkpoint even when Conversation or Run identifiers collide.
- [x] Every resume/read specifies the application-authoritative exact `checkpoint_id` and namespace and never asks for namespace-latest.
- [x] Two-saver fencing tests prove a stale owner may write only to its old epoch namespace; its later checkpoint is unreachable from the new owner and cannot update the Run pointer.
- [x] Checkpoint ordering is explicit: the saver commits first; only after success may an epoch-fenced application transaction update the authoritative Run checkpoint pointer. A crash between those commits leaves an unreachable checkpoint and replays the node at least once; it never advances the Run pointer without a committed checkpoint.
- [x] Checkpoint persistence is internal journal state and OpenTelemetry only. It creates no client Event and never appears in SSE or replay.
