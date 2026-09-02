# Move Conversation context to LangGraph checkpoints

Type: task
Status: completed

## Goal

Remove product Message History while preserving multi-turn Linear Graph context
through the official PostgreSQL LangGraph checkpointer.

## Acceptance criteria

- Keep only the minimal PostgreSQL Conversation registry used for ownership,
  fixed runtime mode, and lifecycle.
- Remove the application Message table and Message History repository APIs.
- Store only logical user and final assistant Conversation Messages in Graph
  state using `add_messages`; never store retrieved chunks or internal Agent
  messages in that channel.
- Build model history only from complete request pairs, excluding the current
  request and incomplete, failed, halted, or disconnected requests, under the
  existing token budget.
- A repeated `clientRequestId` with the same query does not duplicate messages;
  reusing it for a different query returns a conflict.
- Reset all request-local Linear state before each execution.
- Prove multi-turn continuity and Tenant, Subject, and runtime-mode isolation
  against PostgreSQL.
- Do not implement distributed single-flight; record it as deferred.

## Test seams

- `/v2/query/stream` for request identity, multi-turn behavior, interruption,
  and authorization.
- `ConversationRepository` for the minimal registry boundary.
- Alembic upgrade and bootstrap DDL against PostgreSQL.
- Pure Conversation-context projection for complete-pair and token-budget rules.

## Comments

- The repository is pre-release, so the new migration may destructively remove
  the superseded Message table without a compatibility path.
- Same-request concurrency remains explicitly deferred. The current checkpoint
  preflight is not a substitute for cross-process single-flight or fencing.
- Implemented with a minimal `ConversationRepository`, checkpointed
  `conversation_messages`, complete-pair context projection, and explicit HTTP
  `409` for request-ID/query conflicts. Same-ID/same-query retries re-execute
  and reducer IDs prevent duplicate Messages.
- Verification: 237 tests passed and 1 skipped against PostgreSQL; Ruff passed;
  Pyright strict reported 0 errors and 0 warnings; `git diff --check` passed.
- Review closure:
  1. PostgreSQL now reads the latest checkpoint directly to prove same-request
     retry leaves exactly one stable user and assistant Message, and terminal
     checkpoint failure leaves no assistant Message.
  2. A fixed Linear Conversation under an Agent Tenant is rejected before Graph
     streaming or checkpoint access.
  3. Raw checkpoint parsing moved behind typed helpers in `checkpointing.py`;
     the API only maps `RequestIdentityConflict` to HTTP 409, and shared initial
     state uses `list[BaseMessage]`.
  4. `updated_at` is retained as Conversation activity time and is refreshed
     only after authorization and request-identity validation accept a query.
  5. Request-owned stream tests were renamed to describe their actual SSE and
     response-header assertions; durable Message idempotency remains covered by
     the PostgreSQL checkpoint tests.
- Final review: Spec PASS, Standards PASS, zero unresolved review comments.
- Task 53 later removed the remaining pre-release Conversation registry; this
  task remains the historical record of the checkpoint-message migration.
