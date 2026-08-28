# 06: Resume a minimal interrupted Run

**What to build:** Add the atomic recovery service and a default-off test tracer for `POST /v2/runs/{run_id}/resume/stream`. It resumes an interrupted or stale-running minimal Run from its exact latest checkpoint without duplicate execution. Ticket 21 adds `afterSequence` replay/live behavior; Ticket 27 integrates admission, Ticket 28 assembles the release set, and Ticket 29 enables it publicly.

**Blocked by:** 05: Wire the fenced PostgreSQL LangGraph checkpointer.

**Status:** completed

- [x] A resume request returns `404` for a missing/cross-Tenant Run and `409` for a non-stale actively owned Run.
- [x] The server locks and reads the tenant-scoped Run/claim; the HTTP client supplies no epoch. For stale `running`, the locked claim must be expired; for already `interrupted`, no active owner may exist. Either winner transitions directly to `running`, increments exactly once from the locked epoch, claims it, and creates that epoch's checkpoint namespace.
- [x] Both input states continue only from the exact stored prior checkpoint pointer. Concurrent requests serialize on the locked row; after the first transition, later requests observe an active owner and return `409` rather than incrementing again.
- [x] Concurrent tests cover stale-running and already-interrupted inputs: each produces exactly one new owner, one epoch increment, and one terminal result; the stale owner cannot commit after fencing.
- [x] At-least-once minimal-node replay does not duplicate the Run or persisted Events; Message and provider-effect idempotency are owned by the later tickets that introduce them.
- [x] This ticket does not claim final `afterSequence` behavior: its test tracer streams only Events created by the recovered execution, and default production configuration exposes no resume route.
