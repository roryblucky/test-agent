# 24: Add redacted OpenTelemetry coverage

**What to build:** Instrument the complete Linear request, execution, persistence, replay, recovery, and cancellation path with redacted OpenTelemetry traces and low-cardinality metrics.

**Blocked by:** 28.5: Complete the admission-aware release gate.

**Status:** deferred-awaiting-user-confirmation

- [ ] HTTP, graph phase, PydanticAI, provider, checkpoint, replay, cancellation, and recovery operations emit correlated spans.
- [ ] Traces contain only redacted identifiers and metadata; no query, credential, raw Artifact, answer body, or chain-of-thought is exported.
- [ ] Run/Conversation IDs may be span attributes but are not metric labels; errors, interruption, recovery, and cancellation remain diagnosable.
- [ ] Integration tests use an in-memory exporter to assert coverage and redaction without requiring an external collector.

## Comments

- 2026-08-28: Deferred by user until all functional tickets are complete. The Task24 implementation commits were reverted; do not restart this ticket without explicit user confirmation.
