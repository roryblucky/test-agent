# 03: Persist minimal Run and Event records

**What to build:** Add tenant-scoped application Run/Event schema and repositories, then integrate persist-before-deliver into the still test-only v2 tracer. Public enablement remains owned by Ticket 29.

**Blocked by:** 02: Establish the application PostgreSQL foundation.

**Status:** completed

- [x] A completed tracer request leaves a tenant-scoped Run, ordered Events, and terminal outcome in PostgreSQL while execution remains direct in the receiving FastAPI instance.
- [x] Every repository method requires `tenant_id`; another Tenant receives `404` for a known Run or Event identifier.
- [x] `(tenant_id, run_id, sequence)` and `(tenant_id, run_id, event_key)` are unique. Producer-derived keys follow the spec's phase/lifecycle scheme and never derive from payload bytes.
- [x] One transaction locks the Run and compares the canonical type + step + sorted-key JSON payload: identical repeated keys return the existing Event; different content raises an invariant conflict and neither overwrites nor allocates sequence; a new key allocates the next sequence and inserts exactly once.
- [x] Lifecycle and terminal retry tests prove identical-key idempotency and different-payload conflict failure; this generic repository does not implement answer-token aggregation.
- [x] With the test-only flag enabled, every tracer Event uses persist-before-deliver and a persistence failure emits no unpersisted Event; default production configuration still exposes no v2 route.
- [x] New repository code remains in `app.langgraph_v2`; application tables use the Ticket 02 Alembic mechanism.
