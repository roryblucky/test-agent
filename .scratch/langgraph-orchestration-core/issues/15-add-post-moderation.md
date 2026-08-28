# 15: Add post-moderation

**What to build:** The generated answer passes through output moderation before finalization. Flagged output is replaced in the Run's post-moderation-safe final state; already-streamed legacy-compatible token Events cannot be retracted. Message persistence is introduced by Ticket 17.

**Blocked by:** 14: Add advisory groundedness.

**Status:** completed

- [x] Safe answers pass through unchanged with compatible moderation progress.
- [x] Flagged answers produce the compatible safe text in `done`; a regression test documents that preceding token Events may already have reached the client.
- [x] The original flagged answer is absent from the Run's publication-safe final-answer field; this ticket writes no assistant Message.
- [x] Pre-moderation remains the primary publication barrier, and the accepted post-moderation streaming risk is visible in test and operator documentation.
- [x] Crash after phase commit/before checkpoint reuses the first post-moderation decision and safe answer without another moderation call or duplicate Events.
