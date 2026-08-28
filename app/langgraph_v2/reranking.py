"""Replay-safe document reranking phase for the v2 linear graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from app.langgraph_v2.artifacts import ArtifactRef, ArtifactWriter
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultInput
from app.langgraph_v2.run_events import EventInput, EventRecord
from app.models.domain import Document


class RerankingResult(BaseModel):
    """Validated ordered documents returned by a ranker."""

    documents: list[Document] = Field(default_factory=list)


class Ranker(Protocol):
    """Provider seam for deterministic or model-backed ranking."""

    async def rank(self, documents: list[Document]) -> RerankingResult:
        """Return the same documents in the selected order."""
        ...


class MockRanker:
    """Deterministic POC ranker that reverses the retrieved order."""

    async def rank(self, documents: list[Document]) -> RerankingResult:
        """Return a stable reordered copy without changing document contents."""
        return RerankingResult(documents=list(reversed(documents)))


async def run_reranking(
    state: Mapping[str, Any],
    *,
    context: PhaseExecutionContext,
    artifacts: ArtifactWriter,
    ranker: Ranker,
) -> tuple[list[EventRecord], list[ArtifactRef], bool, str | None]:
    """Hydrate retrieved Documents, journal ranking, and return ordered refs."""

    async def invoke() -> PhaseResultInput:
        try:
            refs = [
                ref
                for ref in state.get("artifact_refs", [])
                if ref.get("artifact_type") == "document"
            ]
            documents = [
                Document.model_validate(
                    (
                        await artifacts.get(
                            tenant_id=context.tenant_id,
                            artifact_id=UUID(ref["artifact_id"]),
                        )
                    ).payload
                )
                for ref in refs
            ]
            ranked = await ranker.rank(documents)
            input_ids = [document.id for document in documents]
            output_ids = [document.id for document in ranked.documents]
            if len(output_ids) != len(input_ids) or set(output_ids) != set(input_ids):
                raise ValueError("ranker must return every retrieved document exactly once")
            ref_by_id = dict(zip(input_ids, refs, strict=True))
            ordered_refs = [ref_by_id[document_id] for document_id in output_ids]
            return PhaseResultInput(
                phase_name="reranking",
                normalized_result={"document_ids": output_ids},
                artifact_refs=ordered_refs,
                events=(
                    EventInput(
                        event_key="phase:reranking:step_start:1",
                        type="step_start",
                        step="reranker",
                    ),
                    EventInput(
                        event_key="phase:reranking:step_completed:1",
                        type="step_completed",
                        step="reranker",
                        data={
                            "document_count": len(output_ids),
                            "selected_ids": output_ids,
                        },
                    ),
                ),
            )
        except Exception as exc:
            message = str(exc) or "Reranking failed."
            return PhaseResultInput(
                phase_name="reranking",
                normalized_result={"failed": True, "error": message},
                events=(
                    EventInput(
                        event_key="phase:reranking:step_start:1",
                        type="step_start",
                        step="reranker",
                    ),
                    EventInput(
                        event_key="phase:reranking:error:1",
                        type="error",
                        data=message,
                    ),
                ),
                terminal_status="failed",
            )

    result = await context.repository.get_or_invoke(
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        owner_instance_id=context.owner_instance_id,
        execution_epoch=context.execution_epoch,
        phase_name="reranking",
        invoke=invoke,
    )
    if result.normalized_result.get("failed") is True:
        return list(result.events), [], True, str(result.normalized_result["error"])
    return list(result.events), list(result.artifact_refs), False, None
