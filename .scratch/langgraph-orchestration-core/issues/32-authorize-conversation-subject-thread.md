# 32: Authorize Conversation Subject and thread identity

**What to build:** Make a persistent Conversation the authorization boundary that maps one Tenant and trusted Gateway Subject to a stable LangGraph thread identity.

**Blocked by:** 30: Align the PostgreSQL persistence ADR.

**Status:** done

- [x] Conversation persistence records the owning Subject and stable thread identity with Tenant-scoped uniqueness.
- [x] Creating or continuing a Conversation takes Subject identity only from the trusted request context supplied behind the API Gateway.
- [x] Query, history, and thread lookup return the same not-found response for missing and cross-Tenant/cross-Subject Conversations.
- [x] Possession of a thread identifier alone never authorizes Message or checkpoint access.
- [x] A forward database migration and real-PostgreSQL authorization tests are included.
