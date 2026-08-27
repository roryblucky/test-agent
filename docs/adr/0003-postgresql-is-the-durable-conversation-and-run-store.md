# Use PostgreSQL as the durable Conversation and Run store

PostgreSQL is the system of record for Conversations, Messages, Runs, durable event indexes, and LangGraph checkpoints. Redis may carry ephemeral cache, streaming notifications, and cancellation signals, but it must not be the only copy of user-visible history or recoverable execution state.

For the POC, raw artifacts are stored in PostgreSQL behind an `ArtifactStore` interface. The interface keeps graph state limited to artifact references and permits moving raw payloads to shared object storage without changing orchestration contracts.

Conversation context supports two policies: a token-budgeted sliding window and versioned LLM context compression. A Conversation inherits its Tenant's default policy and may override it; compression summaries remain Conversation-scoped derived data, never Evidence or cross-Conversation long-term memory, and summary failure falls back to the sliding window without blocking the Run.
