# 27: Integrate admission with the Run lifecycle

**What to build:** Bind admission slots to execution epochs across heartbeat, resume, terminal transitions, expired-owner interruption, and shutdown.

**Blocked by:** 21: Follow live Events across instances; 23: Observe cancellation and terminate execution; 26: Integrate admission with initial Run start.

**Status:** ready-for-agent

- [ ] Claim heartbeat renews the matching-epoch slot in the same transaction; a healthy Run held beyond slot TTL never frees capacity.
- [ ] Resume atomically reclaims a matching expired slot for stale-running or acquires free capacity for interrupted, then increments epoch and binds the slot. Capacity failure returns `429` without changing slot, epoch, Run, claim, or checkpoint pointer.
- [ ] Successful completion atomically writes the publication-safe assistant Message, sets `completed`, appends `done`, and releases matching claim/slot; SSE emits `done` afterward. Failed, cancelled, interrupted-by-expiry, and shutdown transitions release matching resources in the same fenced state/Event transaction; repeats are idempotent.
- [ ] Multi-instance tests prove healthy long Runs, concurrent resumes at full capacity, expired-owner CAS, cancellation, and shutdown never oversubscribe or leak capacity.
