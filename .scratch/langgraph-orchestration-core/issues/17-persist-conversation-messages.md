# 17: Persist Conversation Messages

**What to build:** Map legacy `sessionId` to a tenant-scoped durable Conversation and persist exactly-once user and assistant Messages for each Run. History selection is a separate ticket.

**Blocked by:** 16: Finalize the compatible Linear response.

**Status:** completed

- [x] Two requests with the same tenant and session identity map to one Conversation; the same identifier under another Tenant is isolated.
- [x] User Messages persist at Run start; the sanitized assistant Message persists only after successful completion. Failed, cancelled, and interrupted Runs persist no assistant Message.
- [x] Post-moderation regression proves only the Run's publication-safe final answer becomes an assistant Message; the original flagged answer never enters Message storage.
- [x] Idempotency keys prevent Message duplication during retry or resume.
- [x] Cross-Tenant Conversation and Message lookups are indistinguishable from missing resources.
- [x] Resume/retry tests prove Message idempotency now that the Message repository exists.
- [x] The repository exposes a transaction seam for Ticket 26 to resolve/create Conversation and persist the user Message atomically with admission, Run, claim, and first Event.
- [x] The repository exposes a fenced terminal transaction seam for Ticket 27 to write the publication-safe assistant Message atomically with completed Run, `done` Event, and resource release.
