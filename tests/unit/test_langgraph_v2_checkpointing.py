import json
from dataclasses import dataclass
from typing import cast

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app.langgraph_v2.checkpointing import (
    LinearCheckpointStateAdapter,
    thread_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.graph import build_linear_graph
from app.langgraph_v2.postgres import strict_checkpoint_serializer


def test_thread_checkpoint_config_only_sets_thread_id() -> None:
    assert thread_checkpoint_config(thread_id="thread-1") == {
        "configurable": {"thread_id": "thread-1"}
    }


def test_thread_id_is_collision_free_across_trusted_scope_parts() -> None:
    conversation_id = "00000000-0000-0000-0000-000000000001"
    identities = {
        thread_id_for("tenant:a", "subject", "linear", conversation_id),
        thread_id_for("tenant", "a:subject", "linear", conversation_id),
        thread_id_for("tenant:a", "other-subject", "linear", conversation_id),
        thread_id_for("tenant:a", "subject", "agent", conversation_id),
        thread_id_for(
            "tenant:a",
            "subject",
            "linear",
            "00000000-0000-0000-0000-000000000002",
        ),
    }

    assert len(identities) == 5


def test_linear_checkpoint_adapter_accepts_all_owned_channels() -> None:
    messages = [HumanMessage(content="saved conversation")]
    channel_values = {
        "query": "question",
        "conversation_id": "conversation-1",
        "request_id": "request-1",
        "conversation_messages": messages,
        "halted": False,
        "moderation": {"is_flagged": False},
        "refined_query": "question",
        "refinement_usage": {"requests": 1},
        "refinement_error": None,
        "retrieval_error": None,
        "reranking_error": None,
        "answer": "answer",
        "answer_usage": {"requests": 1},
        "citations": [],
        "answer_error": None,
        "groundedness": {"is_grounded": True, "score": 1.0, "details": None},
        "groundedness_usage": {"requests": 1},
        "groundedness_error": None,
        "post_moderation": {"is_flagged": False},
        "post_moderation_error": None,
        "final_response": {"query": "question", "conversation_id": "conversation-1"},
        "branch:to:query": None,
        "__start__": {},
    }

    assert (
        LinearCheckpointStateAdapter().validate_checkpoint_state(channel_values)
        == messages
    )


@pytest.mark.parametrize(
    "channel_values",
    [
        {"conversation_messages": [HumanMessage(content="lost type").model_dump()]},
        {
            "conversation_messages": [
                {
                    "lc": 2,
                    "type": "constructor",
                    "id": ["unknown", "Message"],
                    "args": ["unsafe"],
                }
            ]
        },
        {"query": {"unexpected": "object"}},
        {"groundedness": {"score": float("nan")}},
        {"groundedness": {"score": float("inf")}},
        {"citations": {"not": "a list"}},
        {"groundedness": {"unknown": "Pydantic object"}},
    ],
)
def test_linear_checkpoint_adapter_rejects_invalid_owned_channels(
    channel_values: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="checkpoint"):
        LinearCheckpointStateAdapter().validate_checkpoint_state(channel_values)


@dataclass
class _UnknownDataclass:
    value: str


class _UnknownPydanticModel(BaseModel):
    value: str


@pytest.mark.parametrize(
    "value",
    [_UnknownDataclass(value="unsafe"), _UnknownPydanticModel(value="unsafe")],
)
def test_strict_serializer_downgraded_custom_values_fail_typed_channel(
    value: object,
) -> None:
    serializer = strict_checkpoint_serializer()
    downgraded = serializer.loads_typed(serializer.dumps_typed(value))

    assert isinstance(downgraded, dict)
    with pytest.raises(TypeError, match="checkpoint groundedness is invalid"):
        LinearCheckpointStateAdapter().validate_checkpoint_state(
            {"groundedness": cast(object, downgraded)}
        )


def test_constructor_payload_fails_typed_channel() -> None:
    serializer = strict_checkpoint_serializer()
    payload = {
        "lc": 2,
        "type": "constructor",
        "id": ["unknown", "Constructor"],
        "args": ["unsafe"],
    }
    downgraded = serializer.loads_typed(("json", json.dumps(payload).encode()))

    assert isinstance(downgraded, dict)
    with pytest.raises(TypeError, match="checkpoint constructor payload is invalid"):
        LinearCheckpointStateAdapter().validate_checkpoint_state(
            {"groundedness": cast(object, downgraded)}
        )


@pytest.mark.asyncio
async def test_linear_graph_reuses_strict_json_native_checkpoint_state() -> None:
    checkpointer = InMemorySaver(serde=strict_checkpoint_serializer())
    graph = build_linear_graph(checkpointer)
    config = thread_checkpoint_config(thread_id="linear-thread")

    await graph.ainvoke(
        {
            "query": "first question",
            "conversation_id": "conversation-1",
            "request_id": "request-1",
            "conversation_messages": [],
        },
        config,
    )
    await graph.ainvoke(
        {
            "query": "follow-up question",
            "conversation_id": "conversation-1",
            "request_id": "request-2",
            "conversation_messages": [],
        },
        config,
    )

    checkpoint = await checkpointer.aget_tuple(config)
    assert checkpoint is not None
    values = checkpoint.checkpoint["channel_values"]
    assert isinstance(values["final_response"], dict)
    assert all(isinstance(message, HumanMessage) for message in values["conversation_messages"])
