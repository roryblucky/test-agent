"""Fenced LangGraph PostgreSQL checkpoint integration for v2 Runs."""

from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from opentelemetry import trace

CheckpointPointerWriter = Callable[[str, str], Awaitable[None]]
_TRACER = trace.get_tracer(__name__)


def thread_id_for(tenant_id: str, conversation_id: str) -> str:
    """Encode Tenant and Conversation into a collision-free thread ID."""
    return _encode_parts("thread", tenant_id, conversation_id)


def checkpoint_namespace_for(
    tenant_id: str,
    run_id: str,
    execution_epoch: int,
) -> str:
    """Encode Tenant, Run, and execution epoch into a checkpoint namespace."""
    return _encode_parts("checkpoint", tenant_id, run_id, str(execution_epoch))


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


class FencedAsyncPostgresSaver(AsyncPostgresSaver):
    """Official saver that records its committed checkpoint in the Run."""

    def __init__(
        self,
        conn: Any,
        *,
        checkpoint_namespace: str,
        pointer_writer: CheckpointPointerWriter,
    ) -> None:
        super().__init__(conn)
        self._checkpoint_namespace = checkpoint_namespace
        self._pointer_writer = pointer_writer

    def _scoped_config(self, config: RunnableConfig) -> RunnableConfig:
        """Keep the application namespace while LangGraph runs a root graph.

        LangGraph reserves the root graph namespace as ``""``.  The saver
        still stores that root graph in the application run namespace by
        translating only the checkpoint persistence config.
        """
        configurable = dict(config.get("configurable", {}))
        configurable["checkpoint_ns"] = self._checkpoint_namespace
        return {**config, "configurable": configurable}

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read the current Run namespace from the official saver."""
        with _TRACER.start_as_current_span("langgraph_v2.checkpoint.read") as span:
            span.set_attribute(
                "checkpoint.exact",
                bool(config.get("configurable", {}).get("checkpoint_id")),
            )
            return await super().aget_tuple(self._scoped_config(config))

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Commit through the official saver before fencing the Run pointer."""
        with _TRACER.start_as_current_span("langgraph_v2.checkpoint.write") as span:
            span.set_attribute(
                "checkpoint.exact_parent",
                bool(config.get("configurable", {}).get("checkpoint_id")),
            )
            next_config = await super().aput(
                self._scoped_config(config),
                checkpoint,
                metadata,
                new_versions,
            )
            configurable = next_config.get("configurable", {})
            await self._pointer_writer(
                str(configurable["checkpoint_id"]),
                str(configurable["checkpoint_ns"]),
            )
            return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist intermediate writes in the same fenced namespace."""
        with _TRACER.start_as_current_span(
            "langgraph_v2.checkpoint.write_intermediate"
        ):
            await super().aput_writes(
                self._scoped_config(config), writes, task_id, task_path
            )


def _encode_parts(*parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")
