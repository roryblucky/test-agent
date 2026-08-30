# 26: Integrate admission with initial Run start

**What to build:** Replace the test-only initial start path with one transaction that applies admission and creates all authoritative records required before a Run can be accepted.

**Blocked by:** 17: Persist Conversation Messages; 25: Add the admission repository.

**Status:** ready-for-agent

- [ ] One transaction resolves/creates the tenant-scoped Conversation, writes the exactly-once user Message, acquires the slot, creates the `running` Run and initial claim/epoch, and appends the first lifecycle Event.
- [ ] Any failure or capacity rejection rolls back every record; capacity rejection maps to `429` with `Retry-After` and creates no Conversation, Message, slot, Run, claim, or Event.
- [ ] An accepted test-only query immediately streams the already-persisted first Event and starts direct local execution without queueing.
- [ ] Optional `clientRequestId` validates as `[A-Za-z0-9._:-]{1,128}`; invalid or empty input returns `422`. Canonical query is NFC(CRLF→LF, trim outer Unicode whitespace) with internal whitespace unchanged.
- [ ] Idempotency lookup occurs before Conversation creation. Repeating an ID with the same canonical query and original `sessionId` presence/value reuses its bound Conversation and Run without acquiring capacity or duplicating records; adding/removing/changing `sessionId` or changing query returns `409` without mutation.
- [ ] A duplicate attaches replay/live to running, replays terminal through close, or replays interrupted through its Event and closes for explicit resume.
- [ ] Crash/race tests cover omitted IDs, identical concurrent retries, conflicting payload reuse, and cross-Tenant reuse of the same client ID.
