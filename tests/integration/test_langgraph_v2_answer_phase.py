from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult
from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.graph import build_tracer_graph
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import RunEventRepository
from app.models.domain import Document
from tests.integration.test_langgraph_v2_tracer import parse_sse, persistent_tracer_app


class _AnswerActor:
    calls = 0

    async def answer(self, query: str, documents: list[Document]) -> AnswerResult:
        self.calls += 1
        assert query == "hello"
        assert [document.id for document in documents] == ["d1"]
        return AnswerResult(answer="One. Two\nThree; four", usage={"output_tokens": 4})


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id="d1", content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        return RerankingResult(documents=documents)


@pytest.mark.asyncio
async def test_answer_receives_ranked_documents_and_replays_without_model_call(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(langgraph_v2_migrated_database_url, min_size=1, max_size=2) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a", run_id=uuid4(), conversation_id="c1", owner_instance_id="i1"
        )
        actor = _AnswerActor()
        context = PhaseExecutionContext(
            repository=PhaseResultRepository(pool),
            artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            phase_context=context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=actor,
        )
        state = {"query": "hello", "conversation_id": "c1", "client_request_id": None, "events": []}

        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)

    assert actor.calls == 1
    assert first["answer"] == "One. Two\nThree; four"
    assert second["answer"] == first["answer"]
    answer_events = [event for event in first["events"] if event.get("step") == "llm:answer"]
    assert [event["type"] for event in answer_events] == [
        "step_start", "token", "token", "token", "token", "step_completed"
    ]
    assert [event["data"] for event in answer_events[1:-1]] == [
        "One.", " Two\n", "Three;", " four"
    ]


def test_answer_chunks_are_streamed_before_finalization(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_tracer_app(
        langgraph_v2_migrated_database_url,
        answer_actor=_AnswerActor(),
        retriever=_Retriever(),
        ranker=_Ranker(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream", json={"query": "hello"}, headers={"X-Application-Id": "tenant-a"}
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    token_positions = [index for index, event in enumerate(events) if event["type"] == "token"]
    finalization_position = next(index for index, event in enumerate(events) if event.get("step") == "finalization")
    assert token_positions
    assert max(token_positions) < finalization_position
