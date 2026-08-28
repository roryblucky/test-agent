# 08: Add pre-moderation

**What to build:** The v2 graph checks the original query before any other model or provider work and terminates flagged input using the established streaming error contract.

**Blocked by:** 07: Add the PhaseResult recovery journal.

**Status:** completed

- [x] Safe input advances to the next phase with compatible step events.
- [x] Flagged input stops before refinement and emits the legacy-shaped error outcome.
- [x] The node and its state contract live only in the dedicated v2 package.
- [x] Crash after phase commit/before checkpoint reuses the first moderation result and emits no duplicate Events.
