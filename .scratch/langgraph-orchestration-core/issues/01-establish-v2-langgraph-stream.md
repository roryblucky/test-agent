# 01: Establish the minimal v2 LangGraph stream

**What to build:** A clean-room test-only `POST /v2/query/stream` tracer in the new `app.langgraph_v2` package. It accepts the legacy request, runs a minimal ingress-to-finalize LangGraph, and emits the established SSE envelope without invoking legacy orchestration. This ticket adds and pins the direct root LangGraph core dependency; all v2 routes remain behind a default-off feature flag until Ticket 29 enables them.

**Blocked by:** None (can start immediately).

**Status:** completed

- [x] With the test-only feature flag enabled, `POST /v2/query/stream` exists without an `/api` prefix, requires `X-Application-Id`, accepts optional `X-User-Groups`, and returns `200 text/event-stream` with `X-Run-Id` and `X-Conversation-Id`; production/default configuration does not expose it yet.
- [x] A legacy-shaped request (`query`, optional `sessionId`, optional additive `clientRequestId`) produces compatible start, completion, and `done` events framed as `data: <JSON>\n\n`; `done.data` retains current `model_dump()` names including `session_id`.
- [x] This ticket captures and passes only deterministic v1 request/header/framing/start/completion/minimal-`done` golden cases; it does not capture token, error, citation, final-output, stopping, or session-continuity cases.
- [x] `app.langgraph_v2` defines its own wire-compatible input model; it does not import `app.api.schemas.QueryRequest` or any module that loads legacy `FlowContext`.
- [x] LangGraph core is a pinned direct root dependency, and the minimal typed state and compiled graph can be exercised with a deterministic fake.
- [x] All new production implementation lives in `app.langgraph_v2`; existing packages contain only minimal router/lifespan registration or compatibility wiring.

## Comments

- Implemented in commits `347afd4` and `8b371cc`.
- Verification: 111 tests passed; Ruff passed; Pyright reported 0 errors.
- Two-axis review: Standards PASS; Spec PASS.
