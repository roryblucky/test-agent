# 19: Continue a Run after SSE disconnect

**What to build:** Closing the client stream ends only that subscription while the already-started LangGraph Run continues in the receiving instance and reaches a persisted terminal state.

**Blocked by:** 16: Finalize the compatible Linear response.

**Status:** completed

- [x] A forced client disconnect does not cancel graph execution.
- [x] The detached Run continues persisting phase Events, checkpoint state, usage, and its terminal outcome.
- [x] The runtime keeps a strong reference to the local execution task until terminal cleanup; disconnect releases only subscriber resources.
- [x] Lifespan shutdown stops accepting work, allows a bounded checkpoint boundary, marks unfinished locally owned Runs `interrupted`, and releases subscriber and claim resources without leaks; Ticket 27 adds the same integration assertion for admission slots.
