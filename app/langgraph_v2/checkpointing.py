"""LangGraph PostgreSQL checkpoint identity helpers."""

from __future__ import annotations

import base64
import json
from typing import Any, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.langgraph_v2.conversation_context import validate_request_identity


def thread_id_for(
    tenant_id: str,
    subject_id: str,
    runtime_mode: str,
    conversation_id: str,
) -> str:
    """Encode trusted scope and Conversation into a collision-free thread ID."""
    return _encode_parts(
        "thread",
        tenant_id,
        subject_id,
        runtime_mode,
        conversation_id,
    )


def thread_checkpoint_config(*, thread_id: str, checkpoint_ns: str) -> RunnableConfig:
    """Build the checkpoint config for one Graph thread invocation."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }


async def read_conversation_messages(
    checkpointer: BaseCheckpointSaver[Any],
    config: RunnableConfig,
) -> list[BaseMessage]:
    """Read the typed Conversation Message channel from the latest checkpoint."""
    checkpoint_tuple = await checkpointer.aget_tuple(config)
    if checkpoint_tuple is None:
        return []
    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
    raw_messages = cast(object, channel_values.get("conversation_messages", []))
    if not isinstance(raw_messages, list) or not all(
        isinstance(message, BaseMessage)
        for message in cast(list[object], raw_messages)
    ):
        raise TypeError("checkpoint conversation_messages are invalid")
    return cast(list[BaseMessage], raw_messages)


async def validate_checkpoint_request_identity(
    checkpointer: BaseCheckpointSaver[Any],
    config: RunnableConfig,
    *,
    request_id: str,
    query: str,
) -> None:
    """Validate a logical request against checkpointed Conversation Messages."""
    messages = await read_conversation_messages(checkpointer, config)
    validate_request_identity(messages, request_id=request_id, query=query)

def _encode_parts(*parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")
