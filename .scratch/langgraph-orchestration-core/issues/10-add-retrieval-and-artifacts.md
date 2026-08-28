# 10: Add retrieval and Artifacts

**What to build:** The refined question invokes a mock retriever through a new adapter, persists production-shaped Document Artifacts, and passes stable references to the next phase.

**Blocked by:** 09: Add question refinement.

**Status:** completed

- [x] Retrieval receives the refined question and emits compatible document count and ID progress.
- [x] Documents and raw mock payloads are persisted behind the v2 Artifact seam rather than embedded as checkpoint payloads.
- [x] Artifact repository operations require `tenant_id`, and another Tenant receives `404` for a known Artifact identifier.
- [x] Empty and failed retrieval outcomes are explicit and covered through the public stream.
- [x] Crash after phase commit/before checkpoint reuses the first Artifact references without another retriever call or duplicate Events.
