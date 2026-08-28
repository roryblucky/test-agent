# LangGraph v2 PostgreSQL checkpointer

The v2 runtime uses the official `AsyncPostgresSaver` from the direct
`langgraph-checkpoint-postgres` dependency. `postgres_lifespan` opens one
bounded `AsyncConnectionPool` per FastAPI instance, calls `setup()` on startup,
and closes the pool on shutdown. The pool uses `autocommit=True` and
`prepare_threshold=0`, which are required by the checkpointer's concurrent
index migrations.

Application identifiers are encoded as URL-safe base64 JSON tuples:

- `thread_id` is `("thread", tenant_id, conversation_id)`.
- `checkpoint_ns` is `("checkpoint", tenant_id, run_id, execution_epoch)`.

The root LangGraph graph reserves its runtime namespace as `""`; the fenced
saver translates that root persistence operation to the Run's encoded
namespace without changing LangGraph's execution rules. The Run stores the
exact checkpoint ID and namespace returned by the committed saver.

Checkpoint ordering is deliberately one-way: `AsyncPostgresSaver.aput()`
commits first, then the epoch-fenced Run pointer transaction executes. If the
second transaction fails, the committed checkpoint remains an unreachable
orphan and the node may run again; a Run pointer is never advanced to an
uncommitted checkpoint. Reads and resumes must use
`exact_checkpoint_config()` with the stored checkpoint ID and namespace; no
namespace-latest lookup is authoritative.

Checkpoint rows are internal journal state. They do not create application
Events and are not emitted in the query SSE stream.
