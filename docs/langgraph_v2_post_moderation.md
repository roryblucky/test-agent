# LangGraph v2 output moderation

The v2 Linear graph always treats pre-moderation as the primary publication
barrier: a flagged user query stops before refinement, retrieval, or model
execution.

Post-moderation runs after answer generation (and groundedness when enabled).
Answer token Events may already have been streamed before the post-moderation
decision is committed, so a flagged answer cannot be retracted from a client
that has received those tokens. The authoritative `done` payload and the
publication-safe final state contain only the replacement text:

> The generated response was flagged by content moderation and has been removed.

Operators should use the persisted post-moderation Event and final `done`
payload as the source of truth for the answer shown after completion.
