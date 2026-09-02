# Use PostgreSQL as the durable checkpoint store

PostgreSQL is authoritative only for official shared LangGraph checkpoints.
There is no application Conversation registry. The application generates a
Conversation UUID when the client omits one; any supplied valid UUID is accepted
and an unknown UUID begins with empty state. The internal `thread_id` is encoded
collision-free from trusted Tenant, Subject, runtime mode, and Conversation UUID,
so client identity cannot cross those checkpoint boundaries.

Checkpointed Graph state contains only the logical user Message and final assistant Message for a
request, merged by stable request-and-role IDs with `add_messages`. Model context
is projected only from complete prior request pairs, so failed, halted, and
disconnected requests do not become later context. A repeated request ID with
the same query may re-execute without duplicating the pair; reuse with a
different query is a conflict.

No product Message History, Turn identity, Message idempotency key, public Resume API,
application-owned Run, transport Event, or retrieval Artifact store is part of
this boundary. Full retrieval chunks and Agent progress remain request-local
and untracked. Distributed single-flight for concurrent requests in one
Conversation and checkpoint retention/deletion policy remain deferred.
