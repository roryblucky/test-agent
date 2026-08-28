# 11: Add reranking

**What to build:** A mock ranker receives retrieved Documents, returns a deterministic order, and makes that order visible to the answer phase and SSE clients.

**Blocked by:** 10: Add retrieval and Artifacts.

**Status:** completed

- [x] The ranker receives every retrieved Document in its original order and returns a stable reordered set.
- [x] Compatible ranking completion data reports the selected IDs and count.
- [x] Ranker failure terminates the Run without falling through to answer generation.
- [x] Crash after phase commit/before checkpoint reuses the first ranked result without another ranker call or duplicate Events.
