# 30: Align the PostgreSQL persistence ADR

**What to build:** Make the repository's normative architecture decisions match the approved Linear redesign before more code is changed: PostgreSQL remains authoritative for Conversation, Message, Artifact, and LangGraph checkpoint data, but an application Run/Event journal is no longer required.

**Blocked by:** None (can start immediately).

**Status:** done

- [x] The persistence ADR no longer requires application-owned Run records, transport Event indexes, heartbeat, claim, lease, or execution epochs.
- [x] The ADR identifies Conversation ownership, Message-based Turn identity/TTL, shared PostgreSQL checkpoints, Artifact provenance, and cache-only Redis responsibilities without contradicting the approved spec.
- [x] No production code or database schema changes are included.
