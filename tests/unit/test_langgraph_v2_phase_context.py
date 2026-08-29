from typing import cast
from uuid import uuid4

import pytest

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository


def _context(
    *,
    message_repository: ConversationMessageRepository | None = None,
    request_context: TrustedRequestContext | None = None,
) -> PhaseExecutionContext:
    return PhaseExecutionContext(
        repository=cast(PhaseResultRepository, object()),
        tenant_id="tenant-a",
        run_id=uuid4(),
        owner_instance_id="instance-a",
        execution_epoch=1,
        message_repository=message_repository,
        request_context=request_context,
    )


def test_phase_context_requires_trusted_context_for_message_history() -> None:
    with pytest.raises(ValueError, match="request_context is required"):
        _context(message_repository=cast(ConversationMessageRepository, object()))


def test_phase_context_rejects_a_context_for_another_tenant() -> None:
    with pytest.raises(ValueError, match="tenant_id must match"):
        _context(
            request_context=TrustedRequestContext(
                tenant_id="tenant-b", subject_id="subject-a"
            )
        )
