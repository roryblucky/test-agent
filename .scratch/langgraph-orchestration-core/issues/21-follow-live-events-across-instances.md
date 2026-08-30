# 21: Follow live Events across instances

**What to build:** Complete the still default-off replay/live and resume-stream tracers. `GET /v2/runs/{run_id}/stream?afterSequence=N` replays then follows a Run across instances; `POST /v2/runs/{run_id}/resume/stream?afterSequence=N` replays through the requested sequence boundary and follows the recovered execution. Ticket 28 assembles them and Ticket 29 enables them publicly.

**Blocked by:** 20: Replay persisted stream Events.

**Status:** completed

- [x] Local producers wake subscribers in memory; remote producers use Redis as a best-effort wake-up signal.
- [x] Every wake-up and timeout reconciles the next monotonic sequence from PostgreSQL, so missed or duplicate notifications cannot lose or duplicate Events.
- [x] The handoff from persisted history to live following is gapless under an Event inserted at the boundary.
- [x] Both test tracers implement the final `afterSequence` contract; a value beyond the latest follows a running Run or closes a terminal Run. Default production configuration still exposes no v2 control route.
- [x] Redis loss degrades to bounded polling latency, and no SSE client owns a dedicated PostgreSQL listener connection.
- [x] If a claim expires, one tenant-scoped CAS matching `running` + observed epoch + expiry increments the epoch, clears ownership, sets `interrupted`, and appends the stable-keyed Event atomically; losers reconcile the winner's Event. The transition exposes the same transaction seam that Ticket 27 extends to admission-slot release.
- [x] All followers emit the resulting interrupted Event and close; no follower resumes execution or competes with a newly resumed epoch.
