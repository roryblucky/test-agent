# 38: Resume an interrupted Answer stream

**What to build:** Recover cleanly when the public Answer stream itself is interrupted: the UI can discard the partial Answer and receive one new complete Answer stream from the durable node boundary.

**Blocked by:** 34: Stream the real PydanticAI Answer; 37: Resume a Graph interrupted before Answer.

**Status:** completed

- [x] An orderly disconnect closes and awaits the active model stream and persists no partial assistant Message.
- [x] Resume re-enters from the latest durable checkpoint and starts a replacement Answer stream rather than attempting token-level continuation.
- [x] The replacement stream's complete output is used unchanged by advisory assessments, `done.answer`, and assistant Message persistence.
- [x] A second interruption remains recoverable only within the original Message-derived deadline.
- [x] Tests cover disconnect during Answer, process-loss recovery from another instance, and absence of duplicate assistant Messages.

## Comments

- 2026-08-30 historical Standards/Spec audit: no current findings; prior stream-cleanup helpers remain shallow and specific. Unresolved review comments: 0.
