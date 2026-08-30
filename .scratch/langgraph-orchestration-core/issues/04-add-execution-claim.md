# 04: Add fenced direct-execution claims

**What to build:** Give each directly executing Run an expiring PostgreSQL claim and fence application writes so a stale process cannot remain authoritative. This is ownership for request-driven execution, not a queue or background pickup mechanism.

**Blocked by:** 03: Persist minimal Run and Event records.

**Status:** completed

- [x] Initial execution atomically creates a claim with `owner_instance_id`, `execution_epoch`, `heartbeat_at`, and `expires_at`, then refreshes it while running; no process polls for unowned work.
- [x] Run/Event writes compare the active `execution_epoch`; a fenced stale owner cannot commit Events or terminal state.
- [x] Owner failure leaves an expiring claim and does not require the dead process to mark itself interrupted.
- [x] Multi-connection PostgreSQL tests prove heartbeat expiry and stale-writer rejection for Run and Event writes.
