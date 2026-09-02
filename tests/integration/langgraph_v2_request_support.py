"""Shared authorized request setup for graph integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from app.langgraph_v2.authorization import TrustedRequestContext


@dataclass(frozen=True)
class RequestScope:
    """Trusted Conversation request used by graph tests."""

    context: TrustedRequestContext
    conversation_id: UUID
    request_id: str


def create_request_scope(
    *,
    request_id: UUID | str | None = None,
    context: TrustedRequestContext | None = None,
) -> RequestScope:
    """Create one trusted request scope with application identities."""
    resolved_context = context or TrustedRequestContext(
        tenant_id="tenant-a", subject_id="subject-a"
    )
    resolved_request_id = str(request_id or uuid4())
    return RequestScope(
        context=resolved_context,
        conversation_id=uuid4(),
        request_id=resolved_request_id,
    )
