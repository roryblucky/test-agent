from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic_ai.usage import RunUsage

from app.api.dependencies import TenantContext, get_tenant
from app.api.router import router as legacy_router
from app.langgraph_v2.answer import AnswerResult, AnswerStreamChunk
from app.langgraph_v2.api import register_v2_routes
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.graph import LinearGraphState, build_linear_graph
from app.langgraph_v2.groundedness import GroundednessAssessment
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.postgres import V2PostgresConfig, postgres_lifespan
from app.langgraph_v2.pre_moderation import ModerationDecision
from app.langgraph_v2.question_refinement import (
    QuestionRefinementResult,
    V2ResolvedQuery,
)
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document, GroundednessResult, ModerationResult
from app.models.workflow import CitationReference
from app.services.events import EventEmitter
from app.services.flow_context import FlowContext
from app.services.tenant_manager import TenantManager
from tests.integration.langgraph_v2_turn_support import seed_turn_scope
from tests.integration.test_langgraph_v2_linear_core import (
    configure_linear_tenant,
    parse_sse,
    seed_subject_conversation,
)


class _FailingTerminalCheckpointSaver(AsyncPostgresSaver):
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        if checkpoint.get("channel_values", {}).get("final_response") is not None:
            raise RuntimeError("forced terminal checkpoint failure")
        return await super().aput(config, checkpoint, metadata, new_versions)


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(
            documents=[
                Document(
                    id="d1",
                    content=query,
                    source_url="https://example.test/d1",
                    source_type="mock",
                )
            ]
        )


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        return RerankingResult(documents=documents)


class _Answer:
    calls = 0

    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        del history
        self.calls += 1
        assert query == "hello"
        assert [document.id for document in documents] == ["d1"]
        return AnswerResult(answer="grounded answer [1]", usage={"output_tokens": 3})

    async def answer_stream(
        self, query: str, documents: list[Document], history: Sequence[ConversationTurn]
    ) -> AsyncIterator[AnswerStreamChunk]:
        result = await self.answer(query, documents, history)
        yield AnswerStreamChunk(delta=result.answer)
        yield AnswerStreamChunk(result=result)


class _UsageRefinement:
    async def refine(
        self, query: str, history: Sequence[ConversationTurn]
    ) -> QuestionRefinementResult:
        del history
        return QuestionRefinementResult(
            resolved_query=V2ResolvedQuery(
                original_query=query, standalone_query=query
            ),
            usage={"input_tokens": 2, "output_tokens": 3},
        )


class _Groundedness:
    async def evaluate(
        self, answer: str, documents: list[Document]
    ) -> GroundednessAssessment:
        assert answer == "grounded answer [1]"
        assert [document.id for document in documents] == ["d1"]
        return GroundednessAssessment(
            is_grounded=True,
            score=0.9,
            details="supported",
            usage={"input_tokens": 5, "output_tokens": 2},
        )


class _Moderation:
    calls = 0

    async def check(self, text: str) -> ModerationDecision:
        self.calls += 1
        assert text in {"hello", "grounded answer [1]"}
        return ModerationDecision(is_flagged=False)


class _LegacyFlow:
    async def execute(
        self,
        query: str,
        *,
        emitter: EventEmitter,
        session_id: str,
        message_history: list[object],
    ) -> FlowContext:
        del message_history
        await emitter.emit_step_start("query")
        await emitter.emit_step_completed("query", {"query": query})
        context = FlowContext(query=query, session_id=session_id, emitter=emitter)
        context.refined_query = query
        context.llm_response = "grounded answer [1]"
        document = Document(
            id="d1",
            content="hello",
            source_url="https://example.test/d1",
            source_type="mock",
        )
        context.documents = [document]
        context.ranked_documents = [document]
        context.moderation_result = ModerationResult(is_flagged=False)
        context.groundedness_result = GroundednessResult(
            is_grounded=True, score=0.9, details="supported"
        )
        context.metadata.update(
            {
                "steps_executed": [
                    "query",
                    "pre_moderation",
                    "question_refinement",
                    "retrieval",
                    "reranking",
                    "answer",
                    "groundedness",
                    "moderation:post",
                    "finalization",
                ]
            }
        )
        context.metadata["citations"] = [
            CitationReference(
                index=1,
                evidence_id="__artifact_id__",
                source="https://example.test/d1",
                source_type="mock",
                url="https://example.test/d1",
                snippet="hello",
                highlight_content="hello",
                metadata={"document_id": "d1"},
            ).model_dump(mode="json")
        ]
        context.total_usage = RunUsage(requests=3, input_tokens=7, output_tokens=8)
        return context


class _LegacyManager:
    def __init__(self) -> None:
        self.flow = _LegacyFlow()

    def get_flow_engine(self, app_id: str) -> _LegacyFlow:
        del app_id
        return self.flow


def _state() -> LinearGraphState:
    return {
        "query": "hello",
        "conversation_id": "c1",
        "client_request_id": None,
    }


@pytest.mark.asyncio
async def test_final_payload_preserves_documents_moderation_usage_and_session(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        answer = _Answer()
        moderation = _Moderation()
        scope = await seed_turn_scope(pool)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_turn_id=scope.turn_id,
            request_context=scope.context,
            moderation_provider=moderation,
            refinement_actor=_UsageRefinement(),
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=answer,
            groundedness_actor=_Groundedness(),
        )
        result = await graph.ainvoke(_state())
    assert "final_response" in result
    final_response = result["final_response"]
    assert final_response is not None
    done = final_response.model_dump(by_alias=True)
    assert done["session_id"] == "c1"
    assert done["answer"] == "grounded answer [1]"
    assert done["documents"][0]["id"] == "d1"
    assert done["moderation"]["is_flagged"] is False
    assert done["groundedness"] == {
        "is_grounded": True,
        "score": 0.9,
        "details": "supported",
    }
    assert done["citations"][0]["index"] == 1
    assert done["metadata"]["usage"] == {
        "requests": 3,
        "request_tokens": 7,
        "response_tokens": 8,
        "total_tokens": 15,
        "input_tokens": 7,
        "output_tokens": 8,
    }
    expected = json.loads(
        (
            Path(__file__).parents[1]
            / "fixtures"
            / "langgraph_v2"
            / "v1_finalization_wire.json"
        ).read_text()
    )
    legacy_app = FastAPI()
    legacy_manager = _LegacyManager()
    typed_legacy_manager = cast(TenantManager, legacy_manager)
    legacy_app.include_router(legacy_router)
    legacy_app.state.tenant_manager = legacy_manager
    legacy_app.dependency_overrides[get_tenant] = lambda: TenantContext(
        app_id="tenant-a", manager=typed_legacy_manager
    )
    with TestClient(legacy_app) as client:
        legacy_http = client.post(
            "/api/v1/query/stream",
            json={"query": "hello", "sessionId": "c1"},
            headers={"X-Application-Id": "tenant-a"},
        )
    legacy_frames = [
        json.loads(frame.removeprefix("data: "))
        for frame in legacy_http.text.strip().split("\n\n")
    ]
    assert legacy_http.status_code == 200
    assert legacy_http.headers["content-type"].startswith("text/event-stream")
    assert legacy_frames == expected["legacy_events"]
    assert legacy_frames[-1] == expected["event"]
    stable_done = json.loads(json.dumps(done))
    stable_done["citations"][0]["evidence_id"] = "__artifact_id__"
    assert stable_done == expected["event"]["data"]
    assert final_response.model_dump(by_alias=True) == done
    assert answer.calls == 1
    assert moderation.calls == 2


@pytest.mark.asyncio
async def test_graph_finalization_does_not_own_assistant_message_persistence(
    langgraph_v2_migrated_database_url: str,
) -> None:
    request_context = TrustedRequestContext(
        tenant_id="tenant-a", subject_id="subject-a"
    )
    turn_id = uuid4()
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        messages = ConversationMessageRepository(pool)
        await messages.resolve_conversation(
            context=request_context, conversation_id="c1"
        )
        await messages.create_turn(
            context=request_context,
            conversation_id="c1",
            turn_id=turn_id,
            content="hello",
            idempotency_key=f"turn:{turn_id}:user",
        )
        answer = _Answer()
        await seed_turn_scope(pool, turn_id=turn_id, context=request_context)
        graph = build_linear_graph(
            tenant_id="tenant-a",
            current_turn_id=turn_id,
            message_repository=messages,
            request_context=request_context,
            moderation_provider=_Moderation(),
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=answer,
        )
        state: LinearGraphState = {**_state(), "turn_id": str(turn_id)}

        await graph.ainvoke(state)
        await graph.ainvoke(state)
        retained = await messages.list_messages(
            context=request_context, conversation_id="c1"
        )

    assert answer.calls == 2
    assert [
        (message.role, message.content)
        for message in retained
        if message.turn_id == turn_id
    ] == [("user", "hello")]


def test_public_v2_sse_matches_final_output_golden(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async def seed() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            await ConversationMessageRepository(pool).resolve_conversation(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="c1",
            )

    asyncio.run(seed())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=langgraph_v2_migrated_database_url),
        ):
            yield

    app = FastAPI(lifespan=lifespan)
    configure_linear_tenant(app)
    register_v2_routes(
        app,
        enabled=True,
        refinement_actor=_UsageRefinement(),
        retriever=_Retriever(),
        ranker=_Ranker(),
        moderation_provider=_Moderation(),
        answer_actor=_Answer(),
        groundedness_actor=_Groundedness(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": "c1"},
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )

    async def read_messages() -> list[tuple[str, str]]:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            messages = await ConversationMessageRepository(pool).list_messages(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="c1",
            )
            return [(message.role, message.content) for message in messages]

    fixture = json.loads(
        (
            Path(__file__).parents[1]
            / "fixtures"
            / "langgraph_v2"
            / "v1_finalization_wire.json"
        ).read_text()
    )
    v2_fixture = json.loads(
        (
            Path(__file__).parents[1]
            / "fixtures"
            / "langgraph_v2"
            / "v2_finalization_wire.json"
        ).read_text()
    )
    events = [
        json.loads(frame.removeprefix("data: "))
        for frame in response.text.strip().split("\n\n")
    ]
    stable_done = events[-1]
    stable_done["data"]["citations"][0]["evidence_id"] = "__artifact_id__"
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "x-run-id" not in response.headers
    assert response.headers["x-conversation-id"] == "c1"
    assert v2_fixture["intentional_additive_fields"] == []
    assert all("sequence" not in event for event in events), events
    assert set(events[-1]["data"]) == set(v2_fixture["done_data_fields"])
    for header in v2_fixture["required_response_headers"]:
        assert header in response.headers
    assert stable_done == fixture["event"]
    assert asyncio.run(read_messages()) == [
        ("user", "hello"),
        ("assistant", "grounded answer [1]"),
    ]


def test_terminal_checkpoint_failure_does_not_persist_assistant(
    langgraph_v2_migrated_database_url: str,
) -> None:
    conversation_id = "terminal-checkpoint-failure"
    asyncio.run(
        seed_subject_conversation(
            langgraph_v2_migrated_database_url,
            conversation_id,
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with postgres_lifespan(
            app,
            config=V2PostgresConfig(database_url=langgraph_v2_migrated_database_url),
            checkpointer_factory=_FailingTerminalCheckpointSaver,
        ):
            yield

    app = FastAPI(lifespan=lifespan)
    configure_linear_tenant(app)
    register_v2_routes(
        app,
        enabled=True,
        retriever=_Retriever(),
        ranker=_Ranker(),
        moderation_provider=_Moderation(),
        answer_actor=_Answer(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/query/stream",
            json={"query": "hello", "sessionId": conversation_id},
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "subject-a",
            },
        )

    async def read_roles() -> list[str]:
        async with AsyncConnectionPool(
            langgraph_v2_migrated_database_url, min_size=1, max_size=2
        ) as pool:
            messages = await ConversationMessageRepository(pool).list_messages(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id=conversation_id,
            )
            return [message.role for message in messages]

    delivered = parse_sse(response.text)
    assert all(event["type"] != "done" for event in delivered)
    assert delivered[-1]["type"] == "error"
    assert "forced terminal checkpoint failure" in delivered[-1]["data"]
    assert asyncio.run(read_roles()) == ["user"]
