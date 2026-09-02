# Remove the Conversation registry

Type: task
Status: completed

## Goal

Remove product Conversation persistence while preserving checkpoint-backed
multi-turn context and trusted Tenant, Subject, and runtime-mode isolation.

## Acceptance criteria

- Drop `langgraph_v2.conversations` and remove `ConversationRepository`.
- Generate a Conversation UUID in the application when the request omits one.
- Accept any valid supplied UUID; an unknown UUID starts an empty thread.
- Derive internal thread identity from trusted Tenant, Subject, runtime mode,
  and public Conversation UUID.
- Preserve complete-pair context, request-ID idempotency/conflict handling, and
  exclusion of failed, halted, or disconnected requests.
- Keep same-Conversation distributed single-flight deferred.
- Verify PostgreSQL migrations, isolation, multi-turn behavior, Ruff, and
  Pyright strict.

## Comments

- No History API, Conversation list, retention job, Resume, or concurrency lock
  is introduced by this task.
- Added Alembic revision `0018_drop_registry`; the complete DDL contains no
  application Conversation or Message tables.
- Real PostgreSQL coverage proves generated UUIDs, unknown UUID multi-turn
  continuity, complete-pair context, request-ID conflict handling, and isolation
  across Tenant, Subject, and runtime-mode checkpoint scopes.
- Verification: 236 tests passed and 1 skipped; Ruff passed; Pyright strict
  reported 0 errors and 0 warnings; `git diff --check` passed.
- Standards review closure:
  1. Query no longer checks or receives the raw application PostgreSQL pool;
     its only persistence dependency is the official checkpointer. The Agent
     runtime factory no longer reserves a generic database-pool parameter.
  2. Test request identity now uses synchronous `create_request_scope()` with
     no fake database argument, async seed seam, or no-op PostgreSQL setup.
