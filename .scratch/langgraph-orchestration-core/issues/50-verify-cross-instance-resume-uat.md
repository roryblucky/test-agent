# 50: Verify cross-instance Resume and UAT readiness

**What to build:** Prove the redesigned Linear core is ready for functional UAT under the approved request-owned, real-time, checkpoint-resumable contract.

**Blocked by:** 49: Drop superseded Run lifecycle schema.

**Status:** completed

- [x] Contract fixtures prove the released request/SSE fields, event names, payload types, error behavior, and final response shape remain compatible.
- [x] Real Uvicorn disconnect tests prove request-owned Graph and PydanticAI iterators are cancelled and awaited.
- [x] Two application instances sharing PostgreSQL prove authorized latest-checkpoint Resume before Answer and during Answer without Redis authority.
- [x] Tests prove Message-based TTL and complete, wrong-Turn, expired, cross-Tenant, and cross-Subject Resume rejection.
- [x] Tests prove groundedness and post-moderation never change streamed text, `done.answer`, or assistant Message content.
- [x] An opt-in warmed profile proves 50 simultaneous query streams enter Graph execution without an application queue.
- [x] Ruff, the focused real-PostgreSQL suite, the full test suite, migration checks, and independent Standards/Spec review all pass.

## Comments

- Implemented in `7bd0f56`: added released wire-contract fixtures, real cross-instance PostgreSQL assertions, advisory-output HTTP invariants, a real Uvicorn disconnect test, an opt-in warmed 50-request profile, and a reproducible UAT gate. The real TCP test exposed a production gap; `_RequestOwnedStreamingResponse` now detects disconnects and cancels and awaits request-owned Graph/model streaming.
- Review fixes in `265c8c2` preserve Starlette ASGI 2.3/2.4 send-error behavior, propagate receive failures, run the disconnect through a local TCP forwarding proxy, convert the profile to 50 actual ASGI POST requests, strengthen captured-fixture and actual Resume contract checks, and make failure cleanup bounded. `c232912` closes the remaining TaskGroup and proxy teardown paths.
- Enterprise ingress disposition: the deterministic local proxy boundary is covered in the automated gate; `docs/testing.md` explicitly requires repeating the disconnect case through the deployed UAT ingress because that external proxy is not reproducible locally.
- Final verification on `c232912`: full `tests/` suite 258 passed and 1 opt-in profile skipped; the enabled 50-request profile passed separately; Ruff passed; Pyright strict reported 0 errors; diff check was clean.
- Final Standards findings: 0. Final Spec findings: 0. Unresolved review comments: 0.
