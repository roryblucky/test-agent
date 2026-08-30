# 22: Persist a cancellation request

**What to build:** A test-only, default-off `POST /v2/runs/{run_id}/cancel` records a tenant-scoped cancellation intent and wakes the owning instance. Executor observation and terminal transition are a separate ticket; Ticket 28 assembles the route and Ticket 29 enables it publicly.

**Blocked by:** 21: Follow live Events across instances.

**Status:** completed

- [x] Cancellation atomically writes a tenant-scoped PostgreSQL intent and returns `202` when accepted; missing/cross-Tenant Runs return `404`, and repeats are idempotent.
- [x] Redis publishes only a best-effort low-latency wake addressed to the current owner; PostgreSQL remains authoritative.
- [x] `202` means intent accepted, not execution stopped; this ticket does not write `cancelled` or emit `stopped`.
- [x] Cancellation of an already terminal Run has one documented idempotent response and cannot mutate terminal state.
- [x] The contract is exercised only with the test flag; default production configuration exposes no cancel route.
