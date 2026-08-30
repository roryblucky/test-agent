# 48: Remove API and finalization Run lifecycle dependency

**What to build:** Complete Query, Resume, success, failure, and disconnect behavior without creating or updating an application Run lifecycle record.

**Blocked by:** 38: Resume an interrupted Answer stream; 46: Migrate Message persistence to Turn identity; 47: Migrate Artifact provenance to Turn identity.

**Status:** completed

- [x] Query and Resume execute using authorized Conversation/Turn data and LangGraph checkpoints without a Run repository.
- [x] Completion, failure, and disconnect produce compatible live SSE outcomes without persisting application Run status.
- [x] Finalization commits the assistant Message and required provenance idempotently by Turn.
- [x] No production code reads or writes owner instance, status, heartbeat, claim, lease, checkpoint pointer, or execution epoch fields.
- [x] The narrow same-thread overlap risk remains explicitly accepted rather than hidden behind a replacement lock.

## Comments

- Implemented in `13ccf96`: Query and Resume now use authorized Conversation/Turn state plus the official shared PostgreSQL checkpointer; successful terminal events persist the assistant Message idempotently by Turn, while failure and disconnect close request-owned work without Run mutation. The wire-compatible `X-Run-Id` remains an ephemeral correlation UUID only.
- Removed the obsolete Run repository, cancellation-intent adapter, fenced saver, lifecycle-coupled Message methods, and their dedicated tests. Public PostgreSQL tests cover Query/Resume success, failure, checkpoint failure, and disconnect with zero Run rows.
- Review fix in `4dede84` removed the orphaned cooperative cancellation hook and exception hierarchy. Real request cancellation remains `StreamingResponse`/iterator-owned and tested through close plus `asyncio.CancelledError` propagation.
- Deliberate concurrency disposition: the narrow same-thread network-partition overlap described in `docs/research/langgraph-streaming-delivery.md` remains accepted for this POC. Resume revalidates the latest Turn immediately before execution and terminal writes are idempotent; no replacement lock, lease, heartbeat, or detached owner was introduced.
- Final verification: Pyright strict and Ruff pass; complete `tests/` suite 252 passed against real PostgreSQL. Final Standards findings: 0. Final Spec findings: 0. Unresolved review comments: 0.
