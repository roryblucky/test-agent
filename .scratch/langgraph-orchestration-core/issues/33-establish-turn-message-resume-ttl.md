# 33: Establish Turn Message identity and Resume TTL

**What to build:** Represent each user interaction as one Turn shared by Message records and LangGraph State, and derive its fixed Resume window from the user Message creation time.

**Blocked by:** 32: Authorize Conversation Subject and thread identity.

**Status:** done

- [x] The exactly-once user Message, eventual assistant Message, and Graph State share one Tenant/Conversation-scoped Turn identifier.
- [x] Resume expiry is calculated from the authoritative user Message creation time plus deployment configuration.
- [x] Query retry and Resume never renew or replace the original Turn creation time.
- [x] Redis is optional cache-only state and cannot grant access or extend a deadline.
- [x] Real-PostgreSQL tests cover duplicate Turn creation and expiry calculations without changing a public Resume route yet.
