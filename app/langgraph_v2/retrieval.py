"""Mock retrieval actor and request-local v2 retrieval phase."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field

from app.langgraph_v2.contracts import LiveStreamEvent
from app.langgraph_v2.evidence import Evidence
from app.models.domain import Document


class RetrievalResult(BaseModel):
    """Production-shaped request-local retrieval result."""

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
    retriever: Retriever,
) -> tuple[list[LiveStreamEvent], list[Evidence], bool, str | None]:
    """Return evidence that remains local to the active Graph invocation."""
    try:
        refined_query = state.get("refined_query")
        result = await retriever.retrieve(
            refined_query if isinstance(refined_query, str) else state["query"]
        )
        turn_id = str(state.get("turn_id", ""))
        evidence = [
            Evidence(
                evidence_id=_evidence_id(turn_id=turn_id, document=document),
                document=document,
            )
            for document in result.documents
        ]
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
                        "evidence_ids": [item.evidence_id for item in evidence],
                    },
                ),
            ],
            evidence,
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
            True,
            message,
        )


def _evidence_id(*, turn_id: str, document: Document) -> str:
    """Create a stable citation ID without persisting the document payload."""
    canonical_payload = json.dumps(
        document.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return str(
        uuid5(
            NAMESPACE_URL,
            ":".join(("langgraph-v2", "turn", turn_id, "retrieval", canonical_payload)),
        )
    )
