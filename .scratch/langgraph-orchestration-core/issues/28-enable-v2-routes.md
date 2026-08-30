# 28: Enable the v2 route set for UAT functional testing

**What to build:** Assemble and validate the Linear, history, recovery, streaming, and cancellation route set behind the existing default-off release gate so a controlled UAT environment can perform functional testing before admission is implemented. This ticket does not make v2 production-ready or enable it by default.

**Blocked by:** 23: Observe cancellation and terminate execution.

**Status:** completed

- [x] With the UAT flag enabled, exactly the four specified v2 routes exist: query stream, replay/live stream, resume stream, and cancel; no blocking query route exists. Default production configuration remains off.
- [x] Query and resume run directly in the receiving FastAPI instance and never enter a task queue. Deployment-wide or Tenant admission limits are explicitly deferred to Tickets 25–27.
- [x] A UAT integration suite covers Tenant isolation, persist-before-deliver, the nine Linear phases, history, replay/live, resume, and cancel with deterministic dependencies.
- [x] UAT configuration documents that concurrency protection, `429` capacity behavior, production enablement, and load acceptance are not provided yet.
- [x] V1 routes and the legacy engine remain unchanged.

## Comments

- 2026-08-29: Implemented in `06b4e44` and review fixes in `8ee8251`. The documented UAT gate passed 58 tests, the full suite passed 273 tests, and independent GPT-5.6-sol Standards and Spec reviews both passed.
