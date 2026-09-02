# LangGraph v2 PostgreSQL checkpointer

The v2 runtime uses the official `AsyncPostgresSaver` from the direct
`langgraph-checkpoint-postgres` dependency. `postgres_lifespan` opens one
bounded `AsyncConnectionPool` per FastAPI instance, calls `setup()` on startup,
and closes the pool on shutdown. The pool uses `autocommit=True` and
`prepare_threshold=0`, which are required by the checkpointer's concurrent
index migrations.

Application identifiers are encoded as URL-safe base64 JSON tuples:

- `thread_id` is derived as
  `("thread", tenant_id, subject_id, runtime_mode, conversation_id)`. The first
  three values come only from trusted server context; clients cannot supply the
  internal ID or escape their checkpoint scope.

The application supplies only `thread_id` in the checkpoint configuration.
`checkpoint_ns` remains part of the official checkpointer schema and is used
internally by LangGraph for root Graph and Subgraph checkpoint namespaces; the
Query API does not set or control it.

Query passes the shared official saver directly to LangGraph. PostgreSQL holds
the official checkpoint state; there is no application checkpoint pointer,
fenced saver, claim, lease, heartbeat, or execution epoch. This stage exposes
no Resume or computation-recovery API.

The root Graph state carries `conversation_messages` with LangGraph's
`add_messages` reducer. Only the logical user Message and final assistant
Message enter that channel, using stable request-and-role IDs. Completed
finalization checkpoints the final assistant Message before `done` is released
to the client. Model context is projected from complete prior request pairs;
incomplete failed, halted, or disconnected requests are ignored. There is no
separate application Message History table.

There is also no application Conversation registry. An unknown valid
Conversation UUID therefore starts an empty checkpoint thread. Retention and
explicit checkpoint deletion are deferred; see the recorded TODO.

Checkpoint rows are internal journal state. They do not create application
Events and are not emitted in the query SSE stream.

## First deployment

There is no deployed or stamped pre-release v2 database to upgrade. Apply the
current migration head to a fresh database for the first deployment. The
equivalent complete application DDL is documented in `docs/sql/langgraph_v2.sql`.
