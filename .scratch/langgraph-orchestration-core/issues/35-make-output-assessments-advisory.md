# 35: Make output assessments advisory

**What to build:** Preserve the real-time Answer as canonical while groundedness and post-moderation assess the completed output only for analysis and BigQuery audit.

**Blocked by:** 34: Stream the real PydanticAI Answer.

**Status:** done

- [x] Low groundedness never halts execution or changes streamed Answer text, final Answer text, or the assistant Message.
- [x] Flagged post-moderation never substitutes safe text and never changes streamed Answer text, `done.answer`, or the assistant Message.
- [x] Both assessment results are recorded through the BigQuery audit boundary with Tenant, Conversation, Turn, and assessment identity.
- [x] Pre-moderation remains the blocking input gate.
- [x] Regression tests cover flagged and evaluator-failure paths without reintroducing a publication gate.
