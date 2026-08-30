"""Mock retrieval actor and replay-safe v2 retrieval phase."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, Field

from app.langgraph_v2.artifacts import ArtifactRef, ArtifactScope, ArtifactWriter
from app.langgraph_v2.contracts import LiveStreamEvent
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
    scope: ArtifactScope,
    artifacts: ArtifactWriter,
    retriever: Retriever,
) -> tuple[list[LiveStreamEvent], list[ArtifactRef], list[Document], bool, str | None]:
    """Persist retrieved Artifacts and return checkpoint-owned State data."""
    try:
        refined_query = state.get("refined_query")
        result = await retriever.retrieve(
            refined_query if isinstance(refined_query, str) else state["query"]
        )
        refs: list[ArtifactRef] = []
        for document in result.documents:
            payload = document.model_dump(mode="json", exclude_none=True)
            artifact = await artifacts.create(
                scope=scope,
                artifact_type="document",
                payload=payload,
                artifact_id=_artifact_id(
                    scope=scope,
                    artifact_type="document",
                    payload=payload,
                ),
            )
            refs.append(
                {
                    "artifact_id": str(artifact.artifact_id),
                    "artifact_type": "document",
                }
            )
        raw = await artifacts.create(
            scope=scope,
            artifact_type="retrieval_raw",
            payload=result.raw_payload,
            artifact_id=_artifact_id(
                scope=scope,
                artifact_type="retrieval_raw",
                payload=result.raw_payload,
            ),
        )
        refs.append(
            {"artifact_id": str(raw.artifact_id), "artifact_type": "retrieval_raw"}
        )
        return (
            [
                LiveStreamEvent(
                    type="step_start",
                    step="retriever",
                ),
                LiveStreamEvent(
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
            ],
            refs,
            result.documents,
            False,
            None,
        )
    except Exception as exc:
        message = str(exc) or "Retrieval failed."
        return (
            [
                LiveStreamEvent(
                    type="step_start",
                    step="retriever",
                ),
                LiveStreamEvent(
                    type="error",
                    data=message,
                    checkpoint_terminal=True,
                ),
            ],
            [],
            [],
            True,
            message,
        )


def _artifact_id(
    *,
    scope: ArtifactScope,
    artifact_type: str,
    payload: Any,
) -> UUID:
    """Address immutable retrieval data consistently across Resume Runs."""
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return uuid5(
        NAMESPACE_URL,
        ":".join(
            (
                "langgraph-v2",
                scope.context.tenant_id,
                scope.conversation_id,
                "turn",
                str(scope.turn_id),
                "retrieval",
                artifact_type,
                canonical_payload,
            )
        ),
    )
