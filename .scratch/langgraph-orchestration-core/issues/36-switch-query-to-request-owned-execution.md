# 36: Switch Query to request-owned execution

**What to build:** Make the released v2 query stream run LangGraph directly inside the receiving FastAPI request and expose the real-time Answer contract without a detached runtime.

**Blocked by:** 31: Add a request-owned Graph stream seam; 33: Establish Turn Message identity and Resume TTL; 34: Stream the real PydanticAI Answer; 35: Make output assessments advisory.

**Status:** done

- [x] `POST /v2/query/stream` immediately enters the nine-phase Graph through the request-owned stream seam.
- [x] Existing request fields, headers, SSE framing, event names, error behavior, and final response shape remain compatible; Conversation, thread, and Turn identifiers are additive.
- [x] Disconnect closes and awaits Graph/PydanticAI execution instead of leaving hidden work running.
- [x] Successful completion persists the original complete Answer exactly once as the assistant Message and emits the same text in `done.answer`.
- [x] Route-level tests prove no detached task is required for a complete query.
