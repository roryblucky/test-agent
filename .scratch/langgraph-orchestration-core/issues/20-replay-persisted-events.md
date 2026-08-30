# 20: Replay persisted stream Events

**What to build:** Add the tenant-scoped persisted-replay service that reads the current Event snapshot after `afterSequence` and closes without creating or following work. Exercise it through a test-only default-off route; Ticket 21 completes live behavior, Ticket 28 assembles the release set, and Ticket 29 enables it publicly.

**Blocked by:** 19: Continue a Run after SSE disconnect.

**Status:** completed

- [x] With the test-only route enabled, replay returns only Events after the requested sequence and preserves their original payloads; production/default configuration does not expose the incomplete control endpoint.
- [x] Replay from another FastAPI instance does not duplicate the Run, Message, or Event.
- [x] `afterSequence` defaults to `0`; negative/non-integer values return `422`; missing or cross-Tenant Runs return `404`.
- [x] The replay-only response closes after the current persisted snapshot, whether the Run is running or terminal.
- [x] A sequence beyond the latest Event replays nothing and closes successfully.
