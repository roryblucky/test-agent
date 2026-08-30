"""Replay-safe final response assembly for the v2 Linear graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast
from uuid import UUID

from app.langgraph_v2.artifacts import ArtifactScope, ArtifactStore
from app.langgraph_v2.contracts import LiveStreamEvent, TracerQueryResponse
from app.models.domain import Document


def _steps(state: Mapping[str, Any]) -> list[str]:
    steps = [
        "query",
        "pre_moderation",
        "question_refinement",
        "retrieval",
        "reranking",
    ]
    if "answer" in state:
        steps.append("answer")
    if "groundedness" in state or "groundedness_error" in state:
        steps.append("groundedness")
    if "post_moderation" in state or "post_moderation_error" in state:
        steps.append("moderation:post")
    steps.append("finalization")
    return steps


def _legacy_usage(usage: Mapping[str, Any]) -> dict[str, Any] | None:
    """Map answer usage to the legacy metadata shape."""
    if not usage:
        return None
    input_tokens = int(usage.get("input_tokens", usage.get("request_tokens", 0)))
    output_tokens = int(usage.get("output_tokens", usage.get("response_tokens", 0)))
    return {
        "requests": int(usage.get("requests", 1)),
        "request_tokens": input_tokens,
        "response_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens", input_tokens + output_tokens)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _combine_usage(usages: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Aggregate usage from each model-backed phase."""
    non_empty = [usage for usage in usages if usage]
    if not non_empty:
        return None
    combined = {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for usage in non_empty:
        combined["requests"] += int(usage.get("requests", 1))
        combined["input_tokens"] += int(
            usage.get("input_tokens", usage.get("request_tokens", 0))
        )
        combined["output_tokens"] += int(
            usage.get("output_tokens", usage.get("response_tokens", 0))
        )
        combined["total_tokens"] += int(
            usage.get(
                "total_tokens",
                int(usage.get("input_tokens", usage.get("request_tokens", 0)))
                + int(usage.get("output_tokens", usage.get("response_tokens", 0))),
            )
        )
    return _legacy_usage(combined)


def _legacy_moderation(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Render moderation fields with the legacy nullable keys present."""
    if value is None:
        return None
    result = dict(value)
    result.setdefault("categories", {})
    result.setdefault("reason", None)
    return result


async def run_finalization(
    state: Mapping[str, Any],
    *,
    scope: ArtifactScope,
    artifacts: ArtifactStore,
) -> tuple[list[LiveStreamEvent], TracerQueryResponse]:
    """Assemble the response and its checkpoint-owned finalization events."""
    documents = []
    if "answer" in state:
        documents = [
            Document.model_validate(
                (
                    await artifacts.get(
                        scope=scope,
                        artifact_id=UUID(ref["artifact_id"]),
                    )
                ).payload
            )
            for ref in state.get("ranked_refs", state.get("artifact_refs", []))
            if ref.get("artifact_type") == "document"
        ]
    response = TracerQueryResponse(
        query=state["query"],
        conversation_id=state["conversation_id"],
        metadata={"steps_executed": _steps(state)},
        refined_query=state.get("refined_query"),
        answer=state.get("answer"),
        documents=[document.model_dump(mode="json") for document in documents],
        moderation=(
            _legacy_moderation(state.get("moderation")) if "answer" in state else None
        ),
        groundedness=state.get("groundedness"),
        citations=state.get("citations", []),
    )
    events = [
        LiveStreamEvent(
            type="step_start",
            step="finalization",
        ),
        LiveStreamEvent(
            type="step_completed",
            step="finalization",
            data={"status": "completed"},
        ),
    ]
    normalized = response.model_dump(mode="json")
    usages: list[Mapping[str, Any]] = [
        state.get("answer_usage", {}),
        state.get("refinement_usage", {}),
    ]
    groundedness_usage = state.get("groundedness_usage", {})
    if isinstance(groundedness_usage, Mapping):
        usages.append(cast(Mapping[str, Any], groundedness_usage))
    usage = _combine_usage(usages)
    if usage is not None:
        normalized["metadata"]["usage"] = usage
    response = TracerQueryResponse.model_validate(normalized)
    return events, response


def finalize_in_memory(
    state: Mapping[str, Any],
) -> tuple[list[LiveStreamEvent], TracerQueryResponse]:
    """Assemble the same response shape for a non-persistent graph."""
    response = TracerQueryResponse(
        query=state["query"],
        conversation_id=state["conversation_id"],
        metadata={"steps_executed": _steps(state)},
        refined_query=state.get("refined_query"),
        answer=state.get("answer"),
        moderation=(
            _legacy_moderation(state.get("moderation")) if "answer" in state else None
        ),
        groundedness=state.get("groundedness"),
        citations=state.get("citations", []),
        documents=[],
    )
    normalized = response.model_dump(mode="json")
    usage = _combine_usage(
        [state.get("answer_usage", {}), state.get("refinement_usage", {})]
    )
    if usage is not None:
        normalized["metadata"]["usage"] = usage
    response = TracerQueryResponse.model_validate(normalized)
    return [
        LiveStreamEvent(type="step_start", step="finalization"),
        LiveStreamEvent(
            type="step_completed",
            step="finalization",
            data={"status": "completed"},
        ),
    ], response
