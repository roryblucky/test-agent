# 46: Migrate Message persistence to Turn identity

**What to build:** Persist and retrieve Conversation history using Turn identity rather than an application Run foreign key, while preserving exactly-once user and assistant Messages.

**Blocked by:** 33: Establish Turn Message identity and Resume TTL; 45: Delete the transport Event journal.

**Status:** completed

- [x] User and assistant Message records are uniquely associated with Tenant, Conversation, and Turn without requiring a Run record.
- [x] Query retry, node re-execution, and Resume cannot duplicate either Message.
- [x] Assistant Message persistence uses the original complete Answer even when advisory post-moderation is flagged.
- [x] Sliding-window history and Message-derived Resume TTL continue to operate from the migrated records.
- [x] Existing data is migrated or backfilled deterministically by a forward-compatible schema change.

## Comments

- Implemented in `fa909fe`: added migration `0012_message_turn_identity`, removed `messages.run_id`, migrated Message/Turn models and persistence to Turn identity, retained transitional Run fencing only around finalization, and covered Query/Resume retries plus advisory post-moderation through public HTTP seams.
- Review fixes in `9752bcb` authorize standalone assistant writes by trusted Subject, restore role-appropriate transitional Run identity during downgrade, and clean stale Run terminology from history tests.
- Final verification: Pyright strict and Ruff pass; complete `tests/` suite 274 passed against real PostgreSQL. Final Standards findings: 0. Final Spec findings: 0. Unresolved review comments: 0.
