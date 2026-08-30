# 07: Add the PhaseResult recovery journal

**What to build:** Add a generic epoch-fenced PhaseResult seam that lets a replayed Linear node reuse its first committed normalized result and Events instead of reinvoking a model or provider.

**Blocked by:** 06: Resume a minimal interrupted Run.

**Status:** completed

- [x] A tenant-scoped PhaseResult is uniquely keyed by Run + phase name and stores only normalized structured output or Artifact references, never raw large provider payloads.
- [x] Node entry returns an existing completed PhaseResult before external invocation. Otherwise one epoch-fenced application transaction commits the normalized result and all stable phase Events; checkpointing happens afterward.
- [x] A crash after PhaseResult/Event commit but before checkpoint causes at-least-once node replay to reuse the result without another provider/model call or duplicate Event.
- [x] Volatile timestamp, duration, owner, and attempt values are excluded from canonical Event and PhaseResult content and belong only in OpenTelemetry.
- [x] Conflicting content for an existing phase key is an invariant failure; a stale epoch cannot read another Tenant's result or commit a replacement.
- [x] The allowed phase keys are exactly the nine spec names. `query` journals the canonical request plus selected history snapshot; Citations are stored inside `answer` and never create another graph phase.
