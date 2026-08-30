# 23: Observe cancellation and terminate execution

**What to build:** Make the owning executor observe the authoritative cancellation intent and complete the cooperative transition to the compatible stopped stream and cancelled Run state.

**Blocked by:** 17: Persist Conversation Messages; 22: Persist a cancellation request.

**Status:** completed

- [x] The executor checks PostgreSQL cancellation intent at every graph-node boundary and immediately before the answer batch transaction; Redis wake reduces latency but bounded polling works without Redis.
- [x] An answer batch already committed is fully delivered before cancellation is applied at the next boundary. Otherwise, after observation no later phase or answer batch is committed; the owner atomically appends the stable-keyed `stopped` Event, sets `cancelled`, releases the claim, and persists no assistant Message. Ticket 27 extends this terminal transaction to release the admission slot.
- [x] Completed, failed, cancelled, and interrupted transitions are externally consistent and idempotent; the POC has no `partial` state and a fenced owner cannot overwrite a newer terminal state.
- [x] Stopping/cancellation v1 golden cases pass or are recorded as intentional v2 control-endpoint differences.
