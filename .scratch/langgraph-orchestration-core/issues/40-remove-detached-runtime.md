# 40: Remove the detached runtime

**What to build:** Remove instance-local background execution and subscription supervision now that every supported Graph execution is owned by its Query or Resume request.

**Blocked by:** 36: Switch Query to request-owned execution; 39: Remove old public control endpoints.

**Status:** completed

- [x] No production path creates or registers a detached Graph task, subscriber, or automatic pickup loop.
- [x] Application startup and shutdown no longer initialize, drain, or interrupt an instance-local Run runtime.
- [x] Query and Resume still complete through the request-owned seam and cancellation cleanup remains awaited.
- [x] Runtime-specific configuration, app state, and tests are removed without changing unrelated legacy Flow Engine behavior.

## Comments

- 2026-08-30 historical Standards/Spec audit: no current findings; detached runtime and invocation seam remain fully removed. Unresolved review comments: 0.
