# 16: Finalize the compatible Linear response

**What to build:** The last graph phase assembles the v1-compatible final payload and execution-analysis metadata from the completed clean-room Linear state.

**Blocked by:** 15: Add post-moderation.

**Status:** completed

- [x] The done payload preserves legacy field names, types, Documents, moderation, groundedness, usage, session identity, and Citations.
- [x] Finalization emits compatible deterministic completion fields but performs no financial analysis; volatile timing/duration is recorded only through OpenTelemetry and never changes canonical replay payloads.
- [x] Snapshot tests record every intentional additive v2 field.
- [x] Final-output golden cases pass, including the current `done.data.session_id` serialization.
- [x] Crash after final PhaseResult/Event commit before checkpoint reuses the first final payload without duplicate terminal output.
