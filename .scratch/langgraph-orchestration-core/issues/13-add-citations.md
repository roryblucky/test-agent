# 13: Add citation output

**What to build:** Enhance the Answer node so inline references are deterministically mapped to retrieved Documents and committed as a sub-result of the existing `answer` PhaseResult, then exposed through the existing citation event and final response contracts. This is not an additional LangGraph phase.

**Blocked by:** 12: Add the PydanticAI answer phase.

**Status:** completed

- [x] Valid references produce stable Citation records linked to Document Artifacts.
- [x] Event order remains tokens, citations, then answer step completion.
- [x] Unknown or malformed references cannot create fabricated Citation targets.
- [x] Citation-related v1 golden cases pass or are listed as explicit additive v2 differences.
- [x] Crash after Answer phase commit/before checkpoint reuses its Citation sub-result without duplicate Events.
