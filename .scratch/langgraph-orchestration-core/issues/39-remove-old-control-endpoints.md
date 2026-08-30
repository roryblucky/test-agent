# 39: Remove old public control endpoints

**What to build:** Expose only the approved request-owned Query and thread Resume streams by removing public APIs tied to application Run identifiers, exact Event replay, live following, and remote cancellation.

**Blocked by:** 38: Resume an interrupted Answer stream.

**Status:** completed

- [x] The run-addressed Resume, replay/live-follow, and remote-cancel routes are absent from routing and OpenAPI output.
- [x] Query and thread Resume remain functional and preserve their approved SSE contracts.
- [x] Removed query parameters and replay cursor behavior are no longer accepted by v2 control routes.
- [x] Route tests demonstrate that no deleted endpoint can start, follow, resume, or cancel execution.

## Comments

- 2026-08-30 historical review audit: fixed silent acceptance of removed replay cursors; Query and thread Resume now reject `afterSequence`, `after_sequence`, and `cursor` with 422 in `037101a`. Unresolved review comments: 0.
