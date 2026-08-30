"""Mock retrieval actor and replay-safe v2 retrieval phase."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, Field

from app.langgraph_v2.artifacts import ArtifactRef, ArtifactWriter
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultInput
from app.langgraph_v2.run_events import EventInput, EventRecord
from app.models.domain import Document


class RetrievalResult(BaseModel):
    """Production-shaped retrieval result before Artifact persistence."""

    documents: list[Document] = Field(default_factory=list[Document])
    raw_payload: dict[str, Any] = Field(default_factory=dict[str, Any])


class Retriever(Protocol):
    """Provider seam for retrieval."""

    async def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve documents for a refined query."""
        ...


class MockRetriever:
    """Deterministic retrieval provider for the v2 POC."""

    async def retrieve(self, query: str) -> RetrievalResult:
        """Return one stable mock document."""
        return RetrievalResult(
            documents=[
                Document(id="mock-doc-1", content=f"Evidence for {query}", score=1.0)
            ],
            raw_payload={"query": query, "source": "mock"},
        )


async def run_retrieval(
    state: Mapping[str, Any],
    *,
    context: PhaseExecutionContext,
    artifacts: ArtifactWriter,
    retriever: Retriever,
) -> tuple[list[EventRecord], list[ArtifactRef], list[Document], bool, str | None]:
    """Journal retrieval output and persist referenced Artifacts."""

    async def invoke() -> PhaseResultInput:
        try:
            result = await retriever.retrieve(
                state.get("refined_query", state["query"])
            )
            refs: list[ArtifactRef] = []
            for index, document in enumerate(result.documents):
                artifact = await artifacts.create(
                    tenant_id=context.tenant_id,
                    artifact_type="document",
                    payload=document.model_dump(exclude_none=True),
                    artifact_id=_artifact_id(context, "document", index),
                )
                refs.append(
                    {
                        "artifact_id": str(artifact.artifact_id),
                        "artifact_type": "document",
                    }
                )
            raw = await artifacts.create(
                tenant_id=context.tenant_id,
                artifact_type="retrieval_raw",
                payload=result.raw_payload,
                artifact_id=_artifact_id(context, "retrieval_raw", 0),
            )
            refs.append(
                {"artifact_id": str(raw.artifact_id), "artifact_type": "retrieval_raw"}
            )
            events = (
                EventInput(
                    event_key="phase:retrieval:step_start:1",
                    type="step_start",
                    step="retriever",
                ),
                EventInput(
                    event_key="phase:retrieval:step_completed:1",
                    type="step_completed",
                    step="retriever",
                    data={
                        "document_count": len(result.documents),
                        "documents": [
                            {"id": document.id, "score": document.score}
                            for document in result.documents
                        ],
                        "artifact_ids": [ref["artifact_id"] for ref in refs],
                    },
                ),
            )
            return PhaseResultInput(
                phase_name="retrieval",
                normalized_result={
                    "document_ids": [document.id for document in result.documents]
                },
                artifact_refs=refs,
                events=events,
            )
        except Exception as exc:
            message = str(exc) or "Retrieval failed."
            return PhaseResultInput(
                phase_name="retrieval",
                normalized_result={"failed": True, "error": message},
                events=(
                    EventInput(
                        event_key="phase:retrieval:step_start:1",
                        type="step_start",
                        step="retriever",
                    ),
                    EventInput(
                        event_key="phase:retrieval:error:1", type="error", data=message
                    ),
                ),
                terminal_status="failed",
            )

    result = await context.repository.get_or_invoke(
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        owner_instance_id=context.owner_instance_id,
        execution_epoch=context.execution_epoch,
        phase_name="retrieval",
        invoke=invoke,
    )
    if result.normalized_result.get("failed") is True:
        return list(result.events), [], [], True, str(result.normalized_result["error"])
    return list(result.events), list(result.artifact_refs), [], False, None


def _artifact_id(
    context: PhaseExecutionContext, artifact_type: str, index: int
) -> UUID:
    """Derive a stable id so a retry after a checkpoint gap is idempotent."""
    return uuid5(
        NAMESPACE_URL,
        f"langgraph-v2:{context.tenant_id}:{context.run_id}:retrieval:{artifact_type}:{index}",
    )
