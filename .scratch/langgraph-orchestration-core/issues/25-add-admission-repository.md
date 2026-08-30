# 25: Add the admission repository

**What to build:** Add the expiring PostgreSQL admission schema and repository operations for deployment and Tenant active-Run caps. Do not integrate Run start, resume, heartbeat, or route publication yet.

**Blocked by:** 23: Observe cancellation and terminate execution.

**Status:** ready-for-agent

- [ ] The deployment-wide default admits at most 50 active execution Runs; a lower Tenant active-Run cap is enforced by the same atomic operation and does not reserve capacity.
- [ ] Atomic acquire never waits or queues: it returns a matching-epoch slot or a capacity-exceeded result with `Retry-After`, without creating other domain records.
- [ ] Renew and release require Tenant + Run + matching epoch; repeated release is idempotent and a stale epoch cannot renew or release a newer slot.
- [ ] Concurrent multi-instance repository tests prove deployment and Tenant caps cannot be oversubscribed; expired slots are reclaimable and Tenant caps do not reserve capacity.
- [ ] SSE follower connections are outside this repository and consume no active-Run slot.
- [ ] Lease expiry uses PostgreSQL time, and capacity rejection returns a positive integer `retry_after_seconds` suitable for later mapping to the HTTP `Retry-After` header.
