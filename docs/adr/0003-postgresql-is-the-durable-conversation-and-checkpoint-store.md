# Use PostgreSQL as the durable Conversation and checkpoint store

PostgreSQL is authoritative for the minimal Tenant-owned Conversation registry
(owning Subject, fixed runtime mode, and lifecycle) and official shared
LangGraph checkpoints. Conversation IDs are database-generated UUID primary
keys. The checkpointer `thread_id` is derived from Tenant plus Conversation and
is not stored. Every checkpoint access authorizes Tenant and Subject first; an
accepted query refreshes the Conversation's `updated_at` activity time.

The application has no separate Message History table. Checkpointed Graph state
contains only the logical user Message and final assistant Message for a
request, merged by stable request-and-role IDs with `add_messages`. Model context
is projected only from complete prior request pairs, so failed, halted, and
disconnected requests do not become later context. A repeated request ID with
the same query may re-execute without duplicating the pair; reuse with a
different query is a conflict.

No Turn identity, Message idempotency key, public Resume API,
application-owned Run, transport Event, or retrieval Artifact store is part of
this boundary. Full retrieval chunks and Agent progress remain request-local
and untracked. Distributed single-flight for concurrent requests in one
Conversation remains deferred.
