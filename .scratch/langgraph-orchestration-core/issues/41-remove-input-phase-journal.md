# 41: Remove the input-phase journal

**What to build:** Let Query, pre-moderation, and question-refinement nodes use ordinary LangGraph State and checkpoints rather than the generic application PhaseResult journal.

**Blocked by:** 36: Switch Query to request-owned execution.

**Status:** completed

- [x] The three input phases execute without reading or writing a PhaseResult or transport Event record.
- [x] Pre-moderation still blocks flagged input before refinement or provider access.
- [x] Resume may repeat an interrupted refinement/model call and produces one valid Graph continuation.
- [x] Mixed operation remains supported while later phases still use the old journal.
- [x] Focused Graph and real-checkpointer tests remain green.

## Comments

- 2026-08-30 historical review audit: the repeated checkpoint-event assembly introduced here and expanded by later tickets was resolved by the local `_with_checkpoint_events` helper in `922cc38`. Unresolved review comments: 0.
