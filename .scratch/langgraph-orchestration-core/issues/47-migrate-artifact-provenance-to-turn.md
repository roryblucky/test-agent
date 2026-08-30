# 47: Migrate Artifact provenance to Turn identity

**What to build:** Retain retrieval Evidence, Citations, and source provenance by Conversation and Turn without using an application Run as their ownership key.

**Blocked by:** 42: Remove the Evidence-phase journal; 45: Delete the transport Event journal.

**Status:** completed

- [x] Artifact ownership and lookup are Tenant/Conversation/Turn scoped and require the authorized Conversation Subject.
- [x] Retrieval and Resume can reuse or deterministically deduplicate retained Artifacts without a Run foreign key.
- [x] Citation identity, source metadata, timestamps, and final response compatibility are preserved.
- [x] Cross-Tenant, cross-Subject, and wrong-Turn Artifact access is indistinguishable from missing data.
- [x] A forward-compatible migration and real-PostgreSQL provenance tests are included.

## Comments

- Implemented in `53e7912`: added migration `0013_artifact_turn_provenance`, authorized Turn-scoped Artifact storage and lookup, Run-independent retrieval identity, scoped public reads, and Resume/citation compatibility tests.
- Review fixes in `3172447` enforce the user-Message Turn foreign key, cover legacy Run-derived backfill and timestamp preservation, verify wrong-Conversation and nonexistent-Turn boundaries, and replace the generic test graph proxy with one shallow scope-seeding helper.
- Deliberate review disposition: deterministic Artifact UUID construction remains separately encoded in runtime code, migration code, and migration tests. The migration must remain self-contained after runtime code evolves, while independently calculated test expectations detect drift; sharing one helper would weaken both guarantees.
- Final verification: Pyright strict and Ruff pass; complete `tests/` suite 277 passed against real PostgreSQL. Final Standards findings: 0. Final Spec findings: 0. Unresolved review comments: 0.
