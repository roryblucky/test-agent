# 45: Delete the transport Event journal

**What to build:** Make SSE a live projection of the request-owned LangGraph stream by deleting the persisted transport Event, replay follower, polling, sequence, and Redis wake-up implementation.

**Blocked by:** 39: Remove old public control endpoints; 40: Remove the detached runtime; 44: Delete generic PhaseResult infrastructure.

**Status:** completed

- [x] Query and Resume SSE are produced without inserting, reading, polling, sequencing, or replaying application transport Events.
- [x] Graph State no longer accumulates public transport history.
- [x] Redis Event notification and bounded PostgreSQL follower code, configuration, and tests are removed.
- [x] Citation, progress, token, completion, error, and done envelopes remain compatible during a live request.
- [x] Physical Event tables are left for the final forward-migration ticket.

## Comments

- Implemented in `7b67aed`: replaced the persisted transport journal with the request-owned LangGraph custom stream, removed replay/follower/Redis wake-up code, removed public sequence values, and retained only the transitional Run lifecycle in `runs.py`. Real PostgreSQL trigger tests prove both Query and Resume complete without inserting Event rows; migration coverage confirms the physical Event table remains.
- Review fixes in `b248b58` added the explicit progress envelope, preserved private terminal metadata for model-valued custom events, and removed the redundant cancellation observer wrapper in favor of direct PostgreSQL checks.
- Final verification: Pyright strict and Ruff pass; complete `tests/` suite 272 passed against real PostgreSQL. Final Standards findings: 0. Final Spec findings: 0. Unresolved review comments: 0.
