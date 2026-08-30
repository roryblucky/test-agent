# 34: Stream the real PydanticAI Answer

**What to build:** Deliver the PydanticAI Agent's final Answer text to the UI as genuine model deltas while retaining the complete final Agent result for finalization and history.

**Blocked by:** 31: Add a request-owned Graph stream seam.

**Status:** done

- [x] Final-output text deltas reach the SSE seam as they are produced, without synthetic sentence splitting, fixed-size chunking, or pacing sleeps.
- [x] Intermediate tool/model events and reasoning content are not mistaken for public final Answer text.
- [x] Stream completion yields the complete validated PydanticAI output used by downstream Graph State.
- [x] Cancellation closes and awaits the PydanticAI stream and does not persist a partial assistant Answer.
- [x] Tests prove concatenated public deltas exactly reproduce the complete Answer.
