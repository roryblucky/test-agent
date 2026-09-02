from __future__ import annotations

# Public-stream coverage complements the pure selector unit tests.
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from app.langgraph_v2.answer import AnswerResult, AnswerStreamChunk
from app.langgraph_v2.checkpointing import (
    read_conversation_messages,
    thread_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.conversation_context import ConversationExchange
from app.langgraph_v2.pre_moderation import MockModerationProvider
from app.langgraph_v2.question_refinement import (
    QuestionRefinementResult,
    V2ResolvedQuery,
)
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)

_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "langgraph_v2"
    / "v1_session_continuity_wire.json"
)


class _RefinementActor:
    def __init__(self) -> None:
        self.histories: list[list[ConversationExchange]] = []

    async def refine(
        self,
        query: str,
        history: Sequence[ConversationExchange],
    ) -> QuestionRefinementResult:
        self.histories.append(list(history))
        return QuestionRefinementResult(
            resolved_query=V2ResolvedQuery(original_query=query, standalone_query=query)
        )


class _AnswerActor:
    def __init__(self) -> None:
        self.histories: list[list[ConversationExchange]] = []

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AnswerResult:
        del documents
        self.histories.append(list(history))
        return AnswerResult(answer=f"answer for {query}")

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationExchange]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id=query, content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        del query
        return RerankingResult(documents=documents)


def test_three_public_turns_restore_complete_checkpoint_context(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "00000000-0000-0000-0000-000000000021"

    refinement = _RefinementActor()
    answer = _AnswerActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=refinement,
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=answer,
    )

    with TestClient(app) as client:
        first = client.post(
            "/v2/query/stream",
            json={"query": "first question", "sessionId": conversation_id},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        second = client.post(
            "/v2/query/stream",
            json={"query": "follow up", "sessionId": conversation_id},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        third = client.post(
            "/v2/query/stream",
            json={"query": "third reference", "sessionId": conversation_id},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        assert client.portal is not None
        isolated_messages = [
            client.portal.call(
                read_conversation_messages,
                app.state.langgraph_v2_checkpointer,
                thread_checkpoint_config(
                    thread_id=thread_id_for(
                        tenant_id,
                        subject_id,
                        runtime_mode,
                        conversation_id,
                    ),
                ),
            )
            for tenant_id, subject_id, runtime_mode in (
                ("tenant-b", "subject-a", "linear"),
                ("tenant-a", "subject-b", "linear"),
                ("tenant-a", "subject-a", "agent"),
            )
        ]

    expected_history = [
        ConversationExchange(user="first question", assistant="answer for first question")
    ]
    fixture = json.loads(_FIXTURE_PATH.read_text())
    assert [request["query"] for request in fixture["requests"]] == [
        "first question",
        "follow up",
    ]
    second_history = [
        *expected_history,
        ConversationExchange(user="follow up", assistant="answer for follow up"),
    ]
    assert first.status_code == second.status_code == third.status_code == 200
    assert isolated_messages == [[], [], []]
    assert refinement.histories == [[], expected_history, second_history]
    assert answer.histories == [[], expected_history, second_history]
    second_done = parse_sse(second.text)[-1]
    assert second_done["type"] == fixture["second_done"]["type"]
    assert second_done["data"]["answer"] == fixture["second_done"]["data"]["answer"]


def test_halted_request_is_not_used_as_later_model_context(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "00000000-0000-0000-0000-000000000023"
    refinement = _RefinementActor()
    answer = _AnswerActor()
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=refinement,
        retriever=_Retriever(),
        ranker=_Ranker(),
        moderation_provider=MockModerationProvider(),
        answer_actor=answer,
    )

    with TestClient(app) as client:
        halted = client.post(
            "/v2/query/stream",
            json={"query": "blocked request", "sessionId": conversation_id},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        completed = client.post(
            "/v2/query/stream",
            json={"query": "safe request", "sessionId": conversation_id},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    assert halted.status_code == completed.status_code == 200
    assert parse_sse(halted.text)[-1]["type"] == "error"
    assert refinement.histories == [[]]
    assert answer.histories == [[]]


def test_client_request_id_retry_and_conflict_are_explicit(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "00000000-0000-0000-0000-000000000024"
    app = persistent_linear_app(
        langgraph_v2_migrated_database_url,
        refinement_actor=_RefinementActor(),
        retriever=_Retriever(),
        ranker=_Ranker(),
        answer_actor=_AnswerActor(),
    )
    request = {
        "query": "stable query",
        "sessionId": conversation_id,
        "clientRequestId": "stable-request",
    }
    headers = {"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"}

    with TestClient(app) as client:
        first = client.post("/v2/query/stream", json=request, headers=headers)
        retry = client.post("/v2/query/stream", json=request, headers=headers)
        assert client.portal is not None
        messages = client.portal.call(
            read_conversation_messages,
            app.state.langgraph_v2_checkpointer,
            thread_checkpoint_config(
                thread_id=thread_id_for(
                    "tenant-a", "subject-a", "linear", conversation_id
                ),
            ),
        )
        conflict = client.post(
            "/v2/query/stream",
            json={**request, "query": "different query"},
            headers=headers,
        )

    assert first.status_code == retry.status_code == 200
    assert first.headers["x-request-id"] == retry.headers["x-request-id"]
    assert [message.id for message in messages].count("stable-request:user") == 1
    assert [message.id for message in messages].count("stable-request:assistant") == 1
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == (
        "clientRequestId was already used for a different query"
    )
