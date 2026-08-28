# 14: Add advisory groundedness

**What to build:** A role-configured evaluator assesses whether the answer is supported by its cited Documents and records the result without blocking publication in this POC.

**Blocked by:** 13: Add citation output.

**Status:** completed

- [x] Groundedness receives the answer and cited Document references and appears in phase progress and final state.
- [x] Low groundedness remains advisory and does not suppress an otherwise valid answer.
- [x] Evaluator errors produce an explicit Run failure rather than a fabricated score.
- [x] Crash after phase commit/before checkpoint reuses the first evaluator result without another evaluator call or duplicate Events.
