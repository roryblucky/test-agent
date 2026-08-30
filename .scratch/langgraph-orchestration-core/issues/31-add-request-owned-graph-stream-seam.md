# 31: Add a request-owned Graph stream seam

**What to build:** Introduce a small request-owned execution seam that consumes a LangGraph stream and produces the legacy-compatible SSE envelopes, while the existing runtime remains available until the public endpoint is switched.

**Blocked by:** 30: Align the PostgreSQL persistence ADR.

**Status:** done

- [x] The seam accepts both initial Graph input and checkpoint Resume input without creating a detached task.
- [x] LangGraph update/custom/message output is translated into existing SSE event names and payload types without exposing checkpoint or debug events.
- [x] Cancelling or closing the consumer closes and awaits the Graph iterator.
- [x] Focused tests exercise the seam directly; no public route changes in this ticket.
