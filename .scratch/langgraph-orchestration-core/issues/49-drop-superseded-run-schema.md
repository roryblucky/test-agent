# 49: Drop superseded Run lifecycle schema

**What to build:** Remove the unused application execution-journal schema through a forward migration after every production caller has moved to Conversation, Turn, Artifact, BigQuery, and LangGraph checkpoint authorities.

**Blocked by:** 44: Delete generic PhaseResult infrastructure; 45: Delete the transport Event journal; 48: Remove API and finalization Run lifecycle dependency.

**Status:** completed

- [x] A new forward migration drops application runs, events, phase results, cancellation intents, claims, leases, epochs, and orphaned indexes/constraints without editing applied migration history.
- [x] Conversation, Message, Artifact, and official LangGraph checkpointer schema and data remain intact.
- [x] Both a clean database setup and an upgrade from the currently released v2 schema succeed against real PostgreSQL.
- [x] No production or test code references the dropped schema after migration.
- [x] Rollout notes identify the migration's compatibility boundary for multi-instance deployment.

## Comments

- Implemented in `0572ad5`: migration `0014_drop_run_lifecycle` removes the four superseded lifecycle tables in dependency order. Downgrade recreates the exact compatible 0013-era schema empty; deleted lifecycle rows are intentionally unrecoverable.
- Real PostgreSQL migration coverage proves clean `head`, populated `0013 → head`, preservation of Conversation, Message, Artifact, and official LangGraph checkpoint data, and compatible `head → 0013` rollback. Migration-boundary tests retain only the legacy SQL required to construct and verify those historical states; current production and behavior tests have no dropped-schema dependency.
- Rollout documentation requires task48 code on every instance and all old instances stopped before applying 0014; pre-task48 code and the 0014 schema cannot coexist.
- Verification: migration suite 8 passed, focused suite 56 passed, LangGraph v2 suite 144 passed, full `tests/` suite 252 passed, Pyright strict 0 errors, changed-file Ruff pass, and diff check clean.
- Final Standards findings: 0. Final Spec findings: 0. Unresolved review comments: 0.
