# LangGraph v2 PostgreSQL checkpointer

The v2 runtime uses the official `AsyncPostgresSaver` from the direct
`langgraph-checkpoint-postgres` dependency. `postgres_lifespan` opens one
bounded `AsyncConnectionPool` per FastAPI instance, calls `setup()` on startup,
and closes the pool on shutdown. The pool uses `autocommit=True` and
`prepare_threshold=0`, which are required by the checkpointer's concurrent
index migrations.

Application identifiers are encoded as URL-safe base64 JSON tuples:

- `thread_id` is `("thread", tenant_id, conversation_id)`.
- `checkpoint_ns` is the empty string used by the root LangGraph graph.

Query and Resume pass the shared official saver directly to LangGraph. Resume
authorizes the Tenant, Subject, Conversation, and Turn before reading the
latest checkpoint, then pins the exact authorized checkpoint ID when execution
starts. PostgreSQL checkpoint state is authoritative; there is no application
checkpoint pointer, fenced saver, claim, lease, heartbeat, or execution epoch.

Reads and resumes use `exact_checkpoint_config()` with the checkpoint ID and
empty namespace returned by the official saver. Completed finalization writes
the assistant Message idempotently by Turn.

Checkpoint rows are internal journal state. They do not create application
Events and are not emitted in the query SSE stream.

## Deployment compatibility boundary

Migration `0014_drop_run_lifecycle` removes the superseded application `runs`,
`events`, `phase_results`, and `cancellation_intents` tables. Deploy it in two
ordered stages across every application instance:

1. Fully deploy the task48 runtime, which no longer reads or writes those
   tables, and confirm every old instance has stopped.
2. Upgrade the database through `0014_drop_run_lifecycle`.

Do not run pre-task48 application instances against the 0014 schema, and do not
mix old and new schema expectations during a rolling deployment. Downgrading
0014 recreates an empty 0013-compatible journal schema; deleted lifecycle rows
are intentionally not restored. Conversation, Message, Artifact, and official
LangGraph checkpoint data are unaffected by this migration.
