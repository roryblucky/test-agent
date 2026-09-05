# Use PostgreSQL as the durable checkpoint store

PostgreSQL is authoritative only for official shared LangGraph checkpoints.
There is no application Conversation registry. The application generates a
Conversation UUID when the client omits one; any supplied valid UUID is accepted
and an unknown UUID begins with empty state. The internal `thread_id` is encoded
collision-free from trusted Tenant, Subject, runtime mode, and Conversation UUID,
so client identity cannot cross those checkpoint boundaries.

The Conversation portion of checkpointed Graph state contains only the logical
user Message and final assistant Message for a request, merged by stable
request-and-role IDs with `add_messages`. Model context is projected only from
complete prior request pairs, so failed, halted, and disconnected requests do
not become later context. A repeated request ID with the same query may
re-execute without duplicating the pair; reuse with a different query is a
conflict. An Agent Graph may additionally checkpoint lightweight Coordination
Rounds and Task control channels that LangGraph uses during execution. Those channels are
framework state, not a product Run, Task History, or recovery contract.

No product Message History, Turn identity, Message idempotency key, public Resume API,
application-owned Run, transport Event, or retrieval Artifact store is part of
this boundary. Full Evidence bodies are staged in PydanticAI Tool-return
metadata and, after the terminal Specialist Result validates, copied into a
concurrency-safe body cache owned by the active API request and injected through
LangGraph runtime context. Parallel branches stage Task Outcomes, Evidence IDs,
and complete bounded Calculation Artifact records; only the barrier may promote
one whole immutable accepted batch. Checkpoints contain those accepted batches
but no Evidence bodies. The body cache is lookup data, while accepted-batch
references remain the eligibility authority. Raw Tool payloads, internal Agent
messages, and transport progress likewise remain request-local and untracked.

The existing request runtime guarantees that one complete derived `thread_id`
cannot have concurrent Requests. Agent mode relies on that prerequisite and does
not add advisory locks, a second admission policy, HTTP 409 double-texting
behavior, a dedicated run-session pool, or lock-session monitoring. If concurrent
same-thread Requests become possible later, their admission and recovery
semantics require a separate decision before the checkpoint contract can be
considered safe for that execution model.

The checkpointer uses strict serialization with pickle fallback and broad module
allowlists disabled. Application values cross the checkpoint boundary as
JSON-native data; only explicitly approved framework Message types or exact
symbols may deserialize. Checkpoint retention/deletion policy remains deferred.
Injecting the official checkpointer does not expose public Resume, recovery,
HITL, time-travel, or checkpoint-management behavior.

Because Evidence bodies and replay locators are not persisted, transparent
cross-process continuation of an Evidence-backed in-progress Run is unsupported.
A missing body for an accepted Evidence ID fails closed as unavailable runtime
state; a future recovery requirement must explicitly add durable Evidence
Artifacts, deterministic rehydration locators, or whole-Run restart semantics.
