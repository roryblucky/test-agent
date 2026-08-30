# 37: Resume a Graph interrupted before Answer

**What to build:** Let an authorized user explicitly recover a Turn interrupted before the Answer phase by addressing its Conversation thread and continuing from the latest durable LangGraph checkpoint.

**Blocked by:** 33: Establish Turn Message identity and Resume TTL; 36: Switch Query to request-owned execution.

**Status:** completed

- [x] `POST /v2/threads/{thread_id}/resume/stream` authorizes the Tenant, Subject, Conversation, and expected Turn before reading checkpoint state.
- [x] Missing or unauthorized targets return not-found, expired Turns return gone, and complete, wrong-Turn, or superseded targets are rejected without Graph execution.
- [x] Valid interruptions at pre-moderation, refinement, retrieval, or reranking continue on the same thread from the latest checkpoint with synchronous durability.
- [x] Retry and Resume do not alter the Message-derived deadline.
- [x] Real-PostgreSQL tests exercise recovery from another application/checkpointer instance without a Run mapping or Redis authority.

## Comments

- 2026-08-30 historical review audit: renamed internal `x_application_id` variables to the domain term `tenant_id`; fixed the superseded-Turn admission race by serializing Query/Resume admission on the Conversation row and rechecking the latest Turn in `037101a`.
- Final review added real-PostgreSQL contention tests for both lock orders in `e824b3b`: Query-first blocks then supersedes Resume; Resume-first binds before the waiting Query proceeds.
- Query/Resume Run-lifecycle duplication is explicitly owned by task48; extracting a short-lived shared runtime before that deletion would add indirection. Unresolved review comments: 0.
