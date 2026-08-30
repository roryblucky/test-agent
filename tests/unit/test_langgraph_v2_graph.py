from typing import cast

import pytest

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.graph import build_tracer_graph


def test_graph_requires_trusted_context_for_message_history() -> None:
    with pytest.raises(ValueError, match="request_context is required"):
        build_tracer_graph(
            message_repository=cast(ConversationMessageRepository, object())
        )


def test_graph_rejects_a_context_for_another_tenant() -> None:
    with pytest.raises(ValueError, match="tenant_id must match"):
        build_tracer_graph(
            tenant_id="tenant-a",
            request_context=TrustedRequestContext(
                tenant_id="tenant-b", subject_id="subject-a"
            ),
        )
