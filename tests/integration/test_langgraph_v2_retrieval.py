from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from app.langgraph_v2.graph import LinearGraphState, build_linear_graph
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)


@pytest.mark.asyncio
async def test_retrieved_chunks_are_not_checkpointed() -> None:
    class Retriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(
                documents=[Document(id="d1", content=f"large chunk for {query}")],
                raw_payload={"provider_response": "also not persisted"},
            )

    checkpointer = MemorySaver()
    graph = build_linear_graph(checkpointer=checkpointer, retriever=Retriever())
    config: RunnableConfig = {"configurable": {"thread_id": "transient-evidence"}}
    state: LinearGraphState = {
        "query": "hello",
        "conversation_id": "c1",
        "turn_id": str(uuid4()),
        "client_request_id": None,
    }

    result = await graph.ainvoke(
        state,
        config=config,
        interrupt_after=["retrieval"],
        durability="sync",
    )
    snapshot = await graph.aget_state(config)

    assert "evidence" in result
    assert result["evidence"][0].document.content == "large chunk for hello"
    assert "evidence" not in snapshot.values
    assert "ranked_evidence" not in snapshot.values
    assert "provider_response" not in repr(snapshot.values)
    assert "large chunk for hello" not in repr(snapshot.values)


def test_empty_retrieval_is_explicit_on_public_stream(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class EmptyRetriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(raw_payload={"query": query, "source": "empty"})

    app = persistent_linear_app(
        langgraph_v2_migrated_database_url, retriever=EmptyRetriever()
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "empty"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
    events = parse_sse(response.text)
    retrieval = next(
        event
        for event in events
        if event.get("step") == "retriever" and event["type"] == "step_completed"
    )
    assert response.status_code == 200
    assert retrieval["data"]["document_count"] == 0
    assert retrieval["data"]["documents"] == []


def test_failed_retrieval_is_error_without_finalization_on_public_stream(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class FailingRetriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            raise RuntimeError(f"provider unavailable for {query}")

    app = persistent_linear_app(
        langgraph_v2_migrated_database_url, retriever=FailingRetriever()
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "fail"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
    events = parse_sse(response.text)
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert events[-1]["data"] == "provider unavailable for fail"
    assert all(event.get("step") != "finalization" for event in events)
