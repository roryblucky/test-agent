"""LangGraph PostgreSQL checkpoint identity helpers."""

from __future__ import annotations

import base64
import json

from langchain_core.runnables import RunnableConfig


def thread_id_for(tenant_id: str, conversation_id: str) -> str:
    """Encode Tenant and Conversation into a collision-free thread ID."""
    return _encode_parts("thread", tenant_id, conversation_id)


def initial_checkpoint_config(*, thread_id: str, checkpoint_ns: str) -> RunnableConfig:
    """Build the config used only when creating the first checkpoint."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }


def exact_checkpoint_config(
    *,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
) -> RunnableConfig:
    """Build a read/resume config pinned to one application-known checkpoint."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }


def _encode_parts(*parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")
