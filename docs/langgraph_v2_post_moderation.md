# LangGraph v2 output moderation

The v2 Linear graph always treats pre-moderation as the primary publication
barrier: a flagged user query stops before refinement, retrieval, or model
execution.

Post-moderation runs after answer generation (and groundedness when enabled).
Answer token Events may already have been streamed before the post-moderation
decision is committed, so a flagged answer cannot be retracted from a client
that has received those tokens. Post-moderation is advisory analysis: its
decision, including a flagged result or evaluator failure, is recorded through
the output-assessment audit port and never halts the graph or changes the
canonical Answer. The streamed tokens, authoritative `done.answer`, and
assistant Message all retain the completed Answer produced by the Answer phase.

Groundedness follows the same advisory rule. Setup and evaluator failures are
represented as non-terminal assessment Events and continue through
finalization. Only pre-moderation is a blocking content gate.
