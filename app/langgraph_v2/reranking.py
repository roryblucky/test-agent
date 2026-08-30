"""Replay-safe document reranking phase for the v2 linear graph."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from app.langgraph_v2.artifacts import ArtifactRef, ArtifactStore
from app.langgraph_v2.run_events import EventInput
from app.models.domain import Document


class RerankingResult(BaseModel):
    """Validated ordered documents returned by a ranker."""

    documents: list[Document] = Field(default_factory=list[Document])


class Ranker(Protocol):
    """Provider seam for deterministic or model-backed ranking."""

    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        """Return the same documents in the selected order."""
        ...


class MockRanker:
    """Deterministic POC ranker that reverses the retrieved order."""

    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        """Return a stable reordered copy without changing document contents."""
        del query
        return RerankingResult(documents=list(reversed(documents)))


async def run_reranking(
    state: Mapping[str, Any],
    *,
    tenant_id: str,
    artifacts: ArtifactStore,
    ranker: Ranker,
) -> tuple[list[EventInput], list[ArtifactRef], bool, str | None]:
    """Hydrate retrieved Documents and return checkpoint-owned ranked refs."""
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
                        tenant_id=tenant_id,
                        artifact_id=UUID(ref["artifact_id"]),
                    )
                ).payload
            )
            for ref in refs
        ]
        ranked = await ranker.rank(
            state.get("refined_query", state["query"]), documents
        )
        output_ids = [document.id for document in ranked.documents]
        input_keys = [_document_key(document) for document in documents]
        output_keys = [_document_key(document) for document in ranked.documents]
        if Counter(output_keys) != Counter(input_keys):
            raise ValueError("ranker must return every retrieved document exactly once")
        refs_by_key: dict[str, list[ArtifactRef]] = {}
        for document, ref in zip(documents, refs, strict=True):
            refs_by_key.setdefault(_document_key(document), []).append(ref)
        ordered_refs = [refs_by_key[key].pop(0) for key in output_keys]
        return (
            [
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
            ],
            ordered_refs,
            False,
            None,
        )
    except Exception as exc:
        message = str(exc) or "Reranking failed."
        return (
            [
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
            ],
            [],
            True,
            message,
        )


def _document_key(document: Document) -> str:
    """Canonical payload key that preserves distinct Documents sharing an id."""
    return json.dumps(
        document.model_dump(exclude_none=True), ensure_ascii=False, sort_keys=True
    )
