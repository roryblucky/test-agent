# 29: Verify fifty concurrent streams and contract parity

**What to build:** An opt-in load suite demonstrates the completed Linear runtime's agreed concurrency baseline with deterministic mock dependencies, and a final acceptance suite runs the complete captured v1 golden contract. After both pass, change the release gate so default production configuration exposes v2.

**Blocked by:** 24: Add redacted OpenTelemetry coverage; 28.5: Complete the admission-aware release gate.

**Status:** ready-for-agent

- [ ] At least 20 deterministic cohorts each hold fifty accepted mock Runs active after first progress, then submit the 51st request before releasing the barrier; p95 assertions use the combined samples.
- [ ] With deterministic mocks and a warmed test app, response headers and first persisted progress Event each meet p95 ≤ 1 second; a 51st request returns `429` with `Retry-After` at p95 ≤ 500 milliseconds. The fixture overrides any lower Tenant limit.
- [ ] The suite detects Event sequence gaps, duplicate terminal output, database-pool saturation, event-loop blocking, and telemetry gaps.
- [ ] The suite reports connection-start and first-progress latency plus database/loop health, and runs separately from the default unit test suite.
- [ ] The complete captured v1 golden suite passes, with every bounded-token, additive-header/sequence, persistence, and control-endpoint difference explicitly approved.
- [ ] Only after the parity and load assertions pass does default production configuration expose the four routes assembled by Tickets 28 and 28.5; v1 remains unchanged.
