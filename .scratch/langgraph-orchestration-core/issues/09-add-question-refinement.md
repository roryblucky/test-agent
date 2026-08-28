# 09: Add question refinement

**What to build:** A role-configured PydanticAI actor produces a structured standalone question that becomes the input for later Linear phases.

**Blocked by:** 08: Add pre-moderation.

**Status:** completed

- [x] A safe query produces a typed refined question and compatible phase events.
- [x] Invalid structured output or model failure produces an explicit failed Run without invoking retrieval.
- [x] No legacy refinement handler or shared execution context is imported.
- [x] Crash after phase commit/before checkpoint reuses the first structured refinement without another model call or duplicate Events.
