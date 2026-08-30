"""Request-local document reranking phase for the v2 linear graph."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.langgraph_v2.contracts import LiveStreamEvent
from app.langgraph_v2.evidence import Evidence
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
    ranker: Ranker,
) -> tuple[list[LiveStreamEvent], list[Evidence], bool, str | None]:
    """Rank request-local evidence without writing document payloads."""
    try:
        evidence = [Evidence.model_validate(item) for item in state["evidence"]]
        documents = [item.document for item in evidence]
        refined_query = state.get("refined_query")
        ranked = await ranker.rank(
            refined_query if isinstance(refined_query, str) else state["query"],
            documents,
        )
        output_ids = [document.id for document in ranked.documents]
        input_keys = [_document_key(document) for document in documents]
        output_keys = [_document_key(document) for document in ranked.documents]
        if Counter(output_keys) != Counter(input_keys):
            raise ValueError("ranker must return every retrieved document exactly once")
        evidence_by_key: dict[str, list[Evidence]] = {}
        for document, item in zip(documents, evidence, strict=True):
            evidence_by_key.setdefault(_document_key(document), []).append(item)
        ranked_evidence = [evidence_by_key[key].pop(0) for key in output_keys]
        return (
            [
                LiveStreamEvent(
                    type="step_start",
                    step="reranker",
                ),
                LiveStreamEvent(
                    type="step_completed",
                    step="reranker",
                    data={
                        "document_count": len(output_ids),
                        "selected_ids": output_ids,
                    },
                ),
            ],
            ranked_evidence,
            False,
            None,
        )
    except Exception as exc:
        message = str(exc) or "Reranking failed."
        return (
            [
                LiveStreamEvent(
                    type="step_start",
                    step="reranker",
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


def _document_key(document: Document) -> str:
    """Canonical payload key that preserves distinct Documents sharing an id."""
    return json.dumps(
        document.model_dump(exclude_none=True), ensure_ascii=False, sort_keys=True
    )
