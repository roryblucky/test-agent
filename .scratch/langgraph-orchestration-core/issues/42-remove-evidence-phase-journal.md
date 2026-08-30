# 42: Remove the Evidence-phase journal

**What to build:** Let retrieval and reranking run directly under LangGraph checkpoint recovery while retaining durable source Artifacts and provenance as independent domain data.

**Blocked by:** 41: Remove the input-phase journal.

**Status:** completed

- [x] Retrieval and reranking no longer read or write generic PhaseResult or transport Event records.
- [x] Existing tenant-specific provider adapters remain the only bridge to configured retrieval and reranking implementations.
- [x] Graph State carries Artifact references rather than raw duplicated provider payloads.
- [x] An interrupted node may repeat its read-only provider call without duplicating retained Artifacts or corrupting Citation identity.
- [x] Mixed operation remains supported while output phases still use the old journal.

## Comments

- 2026-08-30 historical review audit: checkpoint-event duplication is resolved in `922cc38`; cross-Resume Artifact/Citation identity had already been corrected in `bf1c8c2`. Unresolved review comments: 0.
