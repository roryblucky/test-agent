# 18: Select sliding-window history

**What to build:** Select recent complete Conversation turns within a deterministic token budget and provide that history to refinement and answer actors.

**Blocked by:** 17: Persist Conversation Messages.

**Status:** completed

- [x] A second request with the same Conversation receives recent complete turns in stable chronological order.
- [x] Selection never exceeds the configured token budget and never includes half of a user/assistant turn.
- [x] Current input is not duplicated, and failed/cancelled outputs excluded by the Message policy never enter history.
- [x] Focused unit tests cover empty history, one oversized turn, exact-budget boundaries, and multi-turn eviction.
- [x] Session-continuity golden cases pass through the public stream.
