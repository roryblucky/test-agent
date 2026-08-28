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
    if "groundedness" in state:
        steps.append("groundedness")
    if "post_moderation" in state:
        steps.append("moderation:post")
    steps.append("finalization")
    return steps


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
            moderation=state.get("moderation") if "answer" in state else None,
            groundedness=state.get("groundedness"),
            usage=state.get("answer_usage", {}),
            citations=state.get("citations", []),
        )
        done_data = response.model_dump(by_alias=True)
        if "answer" not in state:
            done_data.pop("usage", None)
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
            EventInput(
                event_key="lifecycle:completed:0",
                type="done",
                data=done_data,
            ),
        )
        return PhaseResultInput(
            phase_name="finalization",
            normalized_result=response.model_dump(mode="json"),
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
        moderation=state.get("moderation") if "answer" in state else None,
        groundedness=state.get("groundedness"),
        usage=state.get("answer_usage", {}),
        citations=state.get("citations", []),
        documents=[],
    )
    done_data = response.model_dump(by_alias=True)
    if "answer" not in state:
        done_data.pop("usage", None)
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
