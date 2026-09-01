# 51: Redesign Conversation History

**What to build:** Replace Turn-keyed Message persistence with the minimal
Conversation/Message history model agreed for the chat UI.

**Blocked by:** None.

**Status:** completed

- [x] PostgreSQL owns UUID-formatted Conversation IDs and persists Tenant mode.
- [x] Messages reference one Conversation, pair by stable request ID, and receive
      an atomic per-Conversation sequence.
- [x] Query maps `sessionId` to Conversation ID and `clientRequestId` to request ID.
- [x] History reads only user/final-assistant Messages in sequence order.
- [x] Turn identity, Message idempotency keys, and stored checkpoint thread IDs are
      absent from History persistence.
- [x] Normative documentation and bootstrap DDL describe the implemented model.
- [x] Focused tests, Ruff, Pyright strict, and the maintained full suite pass.

## Comments

- User confirmed that `sessionId` remains the Conversation ID and
  `clientRequestId` is the stable logical request ID shared by the user Message
  and final assistant Message. Agent intermediate events remain SSE-only.
- Implemented the first-release schema cut in Alembic revision
  `0016_history_redesign`; pre-release Conversation and Message rows are
  intentionally discarded because no prior schema is deployed.
- Verification: `uv run pyright` reported zero errors; Ruff and
  `git diff --check` passed; `pytest tests -q` passed with 238 tests and one
  intentional skip against PostgreSQL. Bare repository-root pytest collection
  remains blocked by the unrelated import-time experiment scripts
  `test_stream_union.py` and `test_union2.py`.
- Review comments resolved: 0 unresolved before the independent final review.
- Review disposition: the former arbitrary-string v2 `sessionId` value domain is
  intentionally superseded by UUID Conversation IDs before first deployment;
  ADR-0002 records why no legacy-ID translation layer is required.
- Final independent review: Standards 0 findings, Spec 0 findings. The one
  audit-identity collision and four Standards comments from the first pass were
  resolved before the final full-suite verification. Unresolved comments: 0.
