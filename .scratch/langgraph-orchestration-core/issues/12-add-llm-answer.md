# 12: Add the PydanticAI answer phase

**What to build:** A role-configured PydanticAI actor returns a complete structured answer from reranked Documents. Code converts it to persisted phrase-sized chunks and then streams those compatible Events without relying on legacy Agent or LLM handlers.

**Blocked by:** 11.5: Bridge existing tenant providers into v2 phases.

**Status:** completed

- [x] The actor receives reranked Document references in their stable order and no hidden legacy context.
- [x] The structured answer is non-empty. After CRLF→LF only, a pure function preserves every remaining code point, closes chunks after `.?!。！？;；\n`, hard-splits at 240 Unicode code points, and round-trips exactly when concatenated; keys are `phase:answer:token:{chunk_index}`.
- [x] One epoch-fenced transaction commits the Answer PhaseResult, usage, Citation sub-result when present, and all chunk Events before the first answer chunk is delivered. SSE uses `answer_chunk_interval_ms` (default/fake-clock 250ms; allowed 200–500ms).
- [x] Cancellation is checked immediately before the batch transaction. After commit, every chunk is delivered in sequence before cancellation may terminate at the next graph boundary.
- [x] Deterministic fake retry with identical content reuses the committed PhaseResult/Events; conflicting content for an existing phase or chunk key fails the Run and sends neither conflicting chunk nor later output.
- [x] Tests record that token Events are visible before post-moderation, matching the accepted legacy compatibility risk.
- [x] The relevant token/error golden cases pass, with bounded token chunk granularity recorded as an intentional v2 difference.
- [x] Model failure closes the stream with the compatible error contract and a failed Run.
- [x] Crash after Answer PhaseResult/Event commit but before checkpoint reuses the normalized answer/chunks without another model call; a crash before that atomic commit may repeat the model call but has delivered no answer chunk.
