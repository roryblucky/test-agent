"""LangGraph PostgreSQL checkpoint identity helpers."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping
from typing import Any, Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.langgraph_v2.contracts import LinearQueryResponse
from app.langgraph_v2.conversation_context import validate_request_identity
from app.langgraph_v2.pre_moderation import ModerationDecision
from app.models.domain import GroundednessResult
from app.models.workflow import CitationReference


class CheckpointStateAdapter(Protocol):
    """Validate one runtime's persisted channels before application use."""

    def validate_checkpoint_state(
        self,
        channel_values: Mapping[str, object],
    ) -> list[BaseMessage]:
        """Return validated Conversation Messages from owned checkpoint state."""
        ...


class LinearCheckpointStateAdapter:
    """Strict projection for every persisted Linear Graph channel."""

    _string_channels = frozenset({"query", "conversation_id", "request_id"})
    _nullable_string_channels = frozenset(
        {
            "refined_query",
            "refinement_error",
            "retrieval_error",
            "reranking_error",
            "answer",
            "answer_error",
            "groundedness_error",
            "post_moderation_error",
        }
    )
    _usage_channels = frozenset(
        {"refinement_usage", "answer_usage", "groundedness_usage"}
    )
    _nullable_json_object_channels = frozenset(
        {"moderation", "post_moderation", "groundedness", "final_response"}
    )
    _owned_channels = (
        _string_channels
        | _nullable_string_channels
        | _usage_channels
        | _nullable_json_object_channels
        | {"citations"}
        | {"conversation_messages", "halted"}
    )

    def validate_checkpoint_state(
        self,
        channel_values: Mapping[str, object],
    ) -> list[BaseMessage]:
        """Validate all Linear channels before Conversation projection."""
        for channel in self._string_channels:
            if channel in channel_values and not isinstance(
                channel_values[channel], str
            ):
                raise TypeError(f"checkpoint {channel} is invalid")
        for channel in self._nullable_string_channels:
            if (
                channel in channel_values
                and channel_values[channel] is not None
                and not isinstance(channel_values[channel], str)
            ):
                raise TypeError(f"checkpoint {channel} is invalid")
        if "halted" in channel_values and not isinstance(
            channel_values["halted"], bool
        ):
            raise TypeError("checkpoint halted is invalid")
        for channel in self._usage_channels:
            if channel in channel_values:
                _validate_json_object(channel_values[channel], channel=channel)
        for channel in self._nullable_json_object_channels:
            if channel in channel_values and channel_values[channel] is not None:
                _validate_json_object(channel_values[channel], channel=channel)
        if "citations" in channel_values:
            _validate_json_array(channel_values["citations"], channel="citations")

        _validate_optional_model(
            channel_values.get("moderation"),
            model=ModerationDecision,
            channel="moderation",
        )
        _validate_optional_model(
            channel_values.get("post_moderation"),
            model=ModerationDecision,
            channel="post_moderation",
        )
        _validate_optional_model(
            channel_values.get("groundedness"),
            model=GroundednessResult,
            channel="groundedness",
        )
        _validate_optional_model(
            channel_values.get("final_response"),
            model=LinearQueryResponse,
            channel="final_response",
        )
        for citation in cast(list[object], channel_values.get("citations", [])):
            _validate_model(citation, model=CitationReference, channel="citations")

        raw_messages = channel_values.get("conversation_messages", [])
        if not isinstance(raw_messages, list) or not all(
            isinstance(message, HumanMessage | AIMessage)
            for message in cast(list[object], raw_messages)
        ):
            raise TypeError("checkpoint conversation_messages are invalid")
        return cast(list[BaseMessage], raw_messages)


def _validate_json_value(value: object) -> None:
    if value is None or isinstance(value, str | int | bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("checkpoint JSON number is invalid")
        return
    if isinstance(value, list):
        for item in cast(list[object], value):
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        json_object = cast(dict[object, object], value)
        if (
            json_object.get("lc") == 2
            and json_object.get("type") == "constructor"
            and "id" in json_object
        ):
            raise TypeError("checkpoint constructor payload is invalid")
        for key, item in json_object.items():
            if not isinstance(key, str):
                raise TypeError("checkpoint JSON object key is invalid")
            _validate_json_value(item)
        return
    raise TypeError("checkpoint value is not JSON-native")


def _validate_json_object(value: object, *, channel: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint {channel} is invalid")
    _validate_json_value(cast(object, value))


def _validate_json_array(value: object, *, channel: str) -> None:
    if not isinstance(value, list):
        raise TypeError(f"checkpoint {channel} is invalid")
    _validate_json_value(cast(object, value))


def _validate_optional_model(
    value: object,
    *,
    model: type[LinearQueryResponse | GroundednessResult | ModerationDecision],
    channel: str,
) -> None:
    if value is not None:
        _validate_model(value, model=model, channel=channel)


def _validate_model(
    value: object,
    *,
    model: type[
        CitationReference | LinearQueryResponse | GroundednessResult | ModerationDecision
    ],
    channel: str,
) -> None:
    try:
        model.model_validate(value)
    except ValueError as error:
        raise TypeError(f"checkpoint {channel} is invalid") from error


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


def thread_checkpoint_config(*, thread_id: str) -> RunnableConfig:
    """Build the checkpoint config for one Graph thread invocation."""
    return {"configurable": {"thread_id": thread_id}}


async def read_conversation_messages(
    checkpointer: BaseCheckpointSaver[Any],
    config: RunnableConfig,
    *,
    state_adapter: CheckpointStateAdapter | None = None,
) -> list[BaseMessage]:
    """Read the typed Conversation Message channel from the latest checkpoint."""
    checkpoint_tuple = await checkpointer.aget_tuple(config)
    if checkpoint_tuple is None:
        return []
    checkpoint = cast(object, checkpoint_tuple.checkpoint)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint payload is invalid")
    checkpoint_fields = cast(Mapping[str, object], checkpoint)
    channel_values = checkpoint_fields.get("channel_values", {})
    if not isinstance(channel_values, Mapping):
        raise TypeError("checkpoint channel_values are invalid")
    adapter = state_adapter or LinearCheckpointStateAdapter()
    return adapter.validate_checkpoint_state(cast(Mapping[str, object], channel_values))


async def validate_checkpoint_request_identity(
    checkpointer: BaseCheckpointSaver[Any],
    config: RunnableConfig,
    *,
    request_id: str,
    query: str,
    state_adapter: CheckpointStateAdapter | None = None,
) -> None:
    """Validate a logical request against checkpointed Conversation Messages."""
    messages = await read_conversation_messages(
        checkpointer,
        config,
        state_adapter=state_adapter,
    )
    validate_request_identity(messages, request_id=request_id, query=query)

def _encode_parts(*parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")
