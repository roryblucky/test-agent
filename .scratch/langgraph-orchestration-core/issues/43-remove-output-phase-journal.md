# 43: Remove the output-phase journal

**What to build:** Move Answer, groundedness, post-moderation, and finalization off the generic PhaseResult journal while preserving real-time output, advisory audit, and exactly-once history.

**Blocked by:** 35: Make output assessments advisory; 42: Remove the Evidence-phase journal.

**Status:** completed

- [x] All four output phases execute without generic PhaseResult or transport Event persistence.
- [x] Re-executing an interrupted Answer node never creates a partial or duplicate assistant Message.
- [x] Groundedness and post-moderation remain advisory BigQuery audit records and never mutate the canonical Answer.
- [x] Finalization emits the compatible final payload and persists the complete assistant Message exactly once by Turn.
- [x] The complete nine-phase Graph runs with no phase depending on the generic journal.

## Comments

- 2026-08-30 historical review audit: corrected stale Resume/finalization descriptions, installed the production BigQuery output-assessment adapter, and added its dedicated schema/identity tests in `037101a`.
- Final review in `e824b3b` made BigQuery a direct production dependency, moved its synchronous setup/write/close work off the event loop, closes partially initialized clients on setup failure, and guarantees lifespan cleanup on normal and exceptional exits.
- The repeated checkpoint-event assembly is resolved by the shallow local helper in `922cc38`; Run lifecycle removal remains explicitly owned by task48. Unresolved review comments: 0.
