# Use PostgreSQL as the durable Conversation and checkpoint store

PostgreSQL is authoritative for Tenant-owned Conversations (including owning
Subject and fixed runtime mode), Messages, and official shared LangGraph
checkpoints. Conversation IDs are database-generated UUID primary keys. The
checkpointer `thread_id` is derived from Tenant plus Conversation and is not
stored. Every history or checkpoint access authorizes Tenant and Subject first.
The stable logical `request_id` pairs one user Message with at most one final
assistant Message; an atomic per-Conversation sequence orders all Messages. No
Turn identity, Message idempotency key, public Resume API, application-owned Run,
transport Event, or retrieval Artifact store is part of this boundary. Full
retrieval chunks and Agent progress remain request-local and untracked.
