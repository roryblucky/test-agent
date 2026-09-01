from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import (
    AnswerCitation,
    AnswerResult,
    AnswerStreamChunk,
)
from app.langgraph_v2.graph import LinearGraphState, build_linear_graph
from app.langgraph_v2.history import ConversationExchange
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.stream import stream_graph
from app.models.domain import Document
from app.models.workflow import AggregatedEvidence
from app.services.citation_extractor import build_citations
from app.services.events import EventEmitter
from tests.integration.langgraph_v2_request_support import seed_request_scope
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)


class _AnswerActor:
    calls = 0

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        self.calls += 1
        assert query == "hello"
        assert [document.id for document in documents] == ["d1"]
        return AnswerResult(answer="One. Two\nThree; four", usage={"output_tokens": 4})

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id="d1", content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        return RerankingResult(documents=documents)


class _StreamingAnswerActor:
    async def answer_stream(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AsyncIterator[AnswerStreamChunk]:
        del query, documents, history
        result = AnswerResult(answer="One. Two.", usage={"output_tokens": 2})
        yield AnswerStreamChunk(delta="One. ")
        yield AnswerStreamChunk(delta="Two.")
        yield AnswerStreamChunk(result=result)


class _FailingAnswer:
    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        del query, documents, history
        raise RuntimeError("answer model unavailable")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _CitingAnswer:
    calls = 0

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        self.calls += 1
        del query, documents, history
        return AnswerResult(
            answer="Cited answer.",
            citations=[
                AnswerCitation(
                    index=1,
                    quoted_text="hello",
                ),
                AnswerCitation(index=2, quoted_text="nope"),
            ],
        )

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _InlineCitationAnswer:
    calls = 0

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        self.calls += 1
        del query, documents, history
        return AnswerResult(
            answer="Supported claim [1]. Malformed [x] [0] []. Unknown claim [99]."
        )

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _RankedInlineAnswer:
    calls = 0

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        self.calls += 1
        del query, documents, history
        return AnswerResult(answer="World claim [1].")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _MalformedCitationAnswer:
    calls = 0

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        self.calls += 1
        del query, documents, history
        return AnswerResult(
            answer="Malformed [x] [0] [] and unmatched [1",
            citations=[AnswerCitation(index=1, quoted_text="hello")],
        )

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


@pytest.mark.asyncio
async def test_answer_receives_ranked_documents_on_each_execution(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        actor = _AnswerActor()
        scope = await seed_request_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            request_context=scope.context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=actor,
        )
        state: LinearGraphState = {
            "query": "hello",
            "conversation_id": "c1",
            "request_id": "request-1",
        }

        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)

    assert actor.calls == 2
    assert "answer" in first
    assert first["answer"] == "One. Two\nThree; four"
    assert "answer" in first
    assert "answer" in second
    assert second["answer"] == first["answer"]


@pytest.mark.asyncio
async def test_compiled_graph_projects_answer_deltas_through_custom_stream(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        scope = await seed_request_scope(pool)
        graph = build_linear_graph(
            checkpointer=MemorySaver(),
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            request_context=scope.context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_StreamingAnswerActor(),
        )
        state: LinearGraphState = {
            "query": "hello",
            "conversation_id": "c1",
            "request_id": "request-1",
        }

        config: RunnableConfig = {"configurable": {"thread_id": "ticket34-stream"}}
        frames = [frame async for frame in stream_graph(graph, state, config=config)]
        result = await graph.aget_state(config)

    events = [parse_sse(frame)[0] for frame in frames]
    token_events = [event for event in events if event["type"] == "token"]
    assert [event["data"] for event in token_events] == ["One. ", "Two."]
    assert "".join(event["data"] for event in token_events) == result.values["answer"]
    assert not any(event.get("data") == "private reasoning" for event in events)


def test_answer_model_failure_fails_the_public_run(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        answer_actor=_FailingAnswer(),
        retriever=_Retriever(),
        ranker=_Ranker(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    events = parse_sse(response.text)
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    fixture = json.loads(
        (
            Path(__file__).parents[1]
            / "fixtures"
            / "langgraph_v2"
            / "v1_answer_wire.json"
        ).read_text()
    )
    assert {key: events[-1][key] for key in ("type", "data")} == fixture["error_event"]
    assert not any(event["type"] == "done" for event in events)


def test_answer_chunks_are_streamed_before_finalization(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        answer_actor=_AnswerActor(),
        retriever=_Retriever(),
        ranker=_Ranker(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    token_positions = [
        index for index, event in enumerate(events) if event["type"] == "token"
    ]
    finalization_position = next(
        index
        for index, event in enumerate(events)
        if event.get("step") == "finalization"
    )
    assert token_positions
    assert max(token_positions) < finalization_position


def test_new_request_does_not_inherit_answer_when_answer_actor_is_unavailable(
    langgraph_v2_migrated_database_url: str,
) -> None:
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        retriever=_Retriever(),
        ranker=_Ranker(),
    )
    app.state.langgraph_v2_answer_actor = _StreamingAnswerActor()
    headers = {
        "X-Application-Id": "tenant-a",
        "X-Subject-Id": "subject-a",
    }

    with TestClient(app) as client:
        first = client.post(
            "/v2/query/stream",
            json={"query": "first", "clientRequestId": "first-request"},
            headers=headers,
        )
        conversation_id = first.headers["x-conversation-id"]
        del app.state.langgraph_v2_answer_actor
        second = client.post(
            "/v2/query/stream",
            json={
                "query": "second",
                "sessionId": conversation_id,
                "clientRequestId": "second-request",
            },
            headers=headers,
        )

    first_events = parse_sse(first.text)
    second_events = parse_sse(second.text)
    assert first_events[-1]["type"] == "done"
    assert first_events[-1]["data"]["answer"] == "One. Two."
    assert second_events[-1]["type"] == "done"
    assert second_events[-1]["data"]["query"] == "second"
    assert second_events[-1]["data"]["answer"] is None
    assert second_events[-1]["data"]["documents"] == []
    assert second_events[-1]["data"]["metadata"]["steps_executed"] == [
        "query",
        "pre_moderation",
        "question_refinement",
        "retrieval",
        "reranking",
        "finalization",
    ]
    assert not any(event["type"] == "token" for event in second_events)


@pytest.mark.asyncio
async def test_answer_citation_subresult_is_bound_on_each_execution(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        actor = _CitingAnswer()
        scope = await seed_request_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            request_context=scope.context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=actor,
        )
        state: LinearGraphState = {
            "query": "hello",
            "conversation_id": "c1",
            "request_id": "request-1",
        }
        first = await graph.ainvoke(state)
        second = await graph.ainvoke(state)

    assert actor.calls == 2
    assert "citations" in first
    assert "citations" in second
    assert first["citations"] == second["citations"]
    assert "citations" in first
    assert first["citations"][0].evidence_id
    assert "citations" in first
    assert first["citations"][0].quoted_text == "hello"


@pytest.mark.asyncio
async def test_answer_inline_citations_map_ranked_documents_and_ignore_unknown_indices(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        actor = _InlineCitationAnswer()
        scope = await seed_request_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            request_context=scope.context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=actor,
        )
        state: LinearGraphState = {
            "query": "hello",
            "conversation_id": "c1",
            "request_id": "request-1",
        }
        result = await graph.ainvoke(state)

    assert actor.calls == 1
    assert "citations" in result
    assert [citation.index for citation in result["citations"]] == [1]
    assert "citations" in result
    assert result["citations"][0].evidence_id
    citation_fixture = json.loads(
        (
            Path(__file__).parents[1]
            / "fixtures"
            / "langgraph_v2"
            / "v1_citations_wire.json"
        ).read_text()
    )
    citation_data = result["citations"][0].model_dump(mode="json")
    legacy_citations, _ = await build_citations(
        "Supported claim [1].",
        [
            AggregatedEvidence(
                evidence_id="legacy:evidence:1",
                source="d1",
                tool_call_id="legacy",
                content="hello",
                citation_index=1,
            )
        ],
    )
    assert legacy_citations
    assert {
        key: legacy_citations[0].model_dump(mode="json")[key]
        for key in citation_fixture["event"]["data"][0]
    } == citation_fixture["event"]["data"][0]
    legacy_emitter = EventEmitter()
    await legacy_emitter.emit_citations(
        [citation.model_dump(mode="json") for citation in legacy_citations]
    )
    await legacy_emitter.close()
    legacy_frames = [frame async for frame in legacy_emitter]
    legacy_wire = json.loads(legacy_frames[0].removeprefix("data: ").strip())
    assert legacy_wire["type"] == citation_fixture["event"]["type"]
    assert legacy_wire["data"] == [
        citation.model_dump(mode="json") for citation in legacy_citations
    ]
    assert citation_data["index"] == legacy_wire["data"][0]["index"]
    assert citation_data["source"] == legacy_wire["data"][0]["source"]
    assert citation_data["evidence_id"] != legacy_wire["data"][0]["evidence_id"]
    assert "highlight_spans" in citation_data


@pytest.mark.asyncio
async def test_inline_citation_uses_reranked_evidence_position(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class TwoRetriever:
        async def retrieve(self, query: str) -> RetrievalResult:
            del query
            return RetrievalResult(
                documents=[
                    Document(id="d1", content="hello"),
                    Document(id="d2", content="world"),
                ]
            )

    class ReverseRanker:
        async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
            del query
            return RerankingResult(documents=list(reversed(documents)))

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        actor = _RankedInlineAnswer()
        scope = await seed_request_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            request_context=scope.context,
            retriever=TwoRetriever(),
            ranker=ReverseRanker(),
            answer_actor=actor,
        )
        result = await graph.ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "request_id": "request-1",
            }
        )

    assert actor.calls == 1
    assert "ranked_evidence" in result
    assert "citations" in result
    assert (
        result["citations"][0].evidence_id == result["ranked_evidence"][0].evidence_id
    )


@pytest.mark.asyncio
async def test_malformed_inline_references_do_not_fallback_to_structured_citations(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        actor = _MalformedCitationAnswer()
        scope = await seed_request_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_request_id=scope.request_id,
            request_context=scope.context,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=actor,
        )
        result = await graph.ainvoke(
            {
                "query": "hello",
                "conversation_id": "c1",
                "request_id": "request-1",
            }
        )

    assert actor.calls == 1
    assert "citations" in result
    assert result["citations"] == []
