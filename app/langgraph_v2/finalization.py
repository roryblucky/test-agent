"""Replay-safe final response assembly for the v2 Linear graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from app.langgraph_v2.artifacts import ArtifactStore
from app.langgraph_v2.contracts import TracerQueryResponse, TracerStreamEvent
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultInput
from app.langgraph_v2.run_events import EventInput, EventRecord
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
    context: PhaseExecutionContext,
    artifacts: ArtifactStore,
) -> tuple[list[EventRecord], TracerQueryResponse]:
    """Assemble and journal the deterministic publication payload once."""

    async def invoke() -> PhaseResultInput:
        documents = []
        if "answer" in state:
            documents = [
                Document.model_validate(
                    (
                        await artifacts.get(
                            tenant_id=context.tenant_id,
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
                _legacy_moderation(state.get("moderation"))
                if "answer" in state
                else None
            ),
            groundedness=state.get("groundedness"),
            citations=state.get("citations", []),
        )
        events = (
            EventInput(
                event_key="phase:finalization:step_start:1",
                type="step_start",
                step="finalization",
            ),
            EventInput(
                event_key="phase:finalization:step_completed:1",
                type="step_completed",
                step="finalization",
                data={"status": "completed"},
            ),
        )
        normalized = response.model_dump(mode="json")
        usages: list[Mapping[str, Any]] = [state.get("answer_usage", {})]
        for phase_name in ("question_refinement", "groundedness"):
            phase = await context.repository.get_completed(
                context.tenant_id, context.run_id, phase_name
            )
            if phase is not None and isinstance(phase.normalized_result, Mapping):
                usages.append(phase.normalized_result.get("usage", {}))
        usage = _combine_usage(usages)
        if usage is not None:
            normalized["metadata"]["usage"] = usage
        return PhaseResultInput(
            phase_name="finalization",
            normalized_result=normalized,
            events=events,
        )

    result = await context.repository.get_or_invoke(
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        owner_instance_id=context.owner_instance_id,
        execution_epoch=context.execution_epoch,
        phase_name="finalization",
        invoke=invoke,
    )
    return list(result.events), TracerQueryResponse.model_validate(
        result.normalized_result
    )


def finalize_in_memory(state: Mapping[str, Any]) -> dict[str, Any]:
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
    done_data = response.model_dump(by_alias=True)
    usage = _legacy_usage(state.get("answer_usage", {}))
    if usage is not None:
        done_data["metadata"]["usage"] = usage
    events = [
        TracerStreamEvent(
            event_key="phase:finalization:step_start:1",
            type="step_start",
            step="finalization",
            sequence=len(state["events"]) + 1,
        ).model_dump(exclude_none=True),
        TracerStreamEvent(
            event_key="phase:finalization:step_completed:1",
            type="step_completed",
            step="finalization",
            data={"status": "completed"},
            sequence=len(state["events"]) + 2,
        ).model_dump(exclude_none=True),
        TracerStreamEvent(
            event_key="lifecycle:completed:0",
            type="done",
            data=done_data,
            sequence=len(state["events"]) + 3,
        ).model_dump(exclude_none=True),
    ]
    return {"events": [*state["events"], *events]}
