# LangGraph v2 PostgreSQL checkpointer

The v2 runtime uses the official `AsyncPostgresSaver` from the direct
`langgraph-checkpoint-postgres` dependency. `postgres_lifespan` opens one
bounded `AsyncConnectionPool` per FastAPI instance, calls `setup()` on startup,
and closes the pool on shutdown. The pool uses `autocommit=True` and
`prepare_threshold=0`, which are required by the checkpointer's concurrent
index migrations.

Application identifiers are encoded as URL-safe base64 JSON tuples:

- `thread_id` is derived as `("thread", tenant_id, conversation_id)` and is not
  stored on the Conversation.
- `checkpoint_ns` is the empty string used by the root LangGraph graph.

Query passes the shared official saver directly to LangGraph. PostgreSQL holds
the official checkpoint state; there is no application checkpoint pointer,
fenced saver, claim, lease, heartbeat, or execution epoch. This stage exposes
no Resume or computation-recovery API. Completed finalization writes the
assistant Message idempotently by logical request ID.

Checkpoint rows are internal journal state. They do not create application
Events and are not emitted in the query SSE stream.

## First deployment

There is no deployed or stamped pre-release v2 database to upgrade. Apply the
current migration head to a fresh database for the first deployment. The
equivalent complete application DDL is documented in `docs/sql/langgraph_v2.sql`.
