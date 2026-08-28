from __future__ import annotations

from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult
from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.graph import build_tracer_graph
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.pre_moderation import ModerationDecision
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import RunEventRepository
from app.models.domain import Document


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

    async def answer(self, query: str, documents: list[Document]) -> AnswerResult:
        self.calls += 1
        assert query == "hello"
        assert [document.id for document in documents] == ["d1"]
        return AnswerResult(answer="grounded answer", usage={"output_tokens": 3})


class _Moderation:
    calls = 0

    async def check(self, text: str) -> ModerationDecision:
        self.calls += 1
        assert text in {"hello", "grounded answer"}
        return ModerationDecision(is_flagged=False)


def _state() -> dict[str, object]:
    return {
        "query": "hello",
        "conversation_id": "c1",
        "client_request_id": None,
        "events": [],
    }


def _context(
    pool: AsyncConnectionPool, run_id, epoch: int, repository: PhaseResultRepository
) -> PhaseExecutionContext:
    return PhaseExecutionContext(
        repository=repository,
        artifact_repository=ArtifactRepository(pool),
        tenant_id="tenant-a",
        run_id=run_id,
        owner_instance_id="i1",
        execution_epoch=epoch,
    )


@pytest.mark.asyncio
async def test_final_payload_preserves_documents_moderation_usage_and_session(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        answer = _Answer()
        moderation = _Moderation()
        graph = build_tracer_graph(
            phase_context=_context(
                pool, run.run_id, run.execution_epoch, PhaseResultRepository(pool)
            ),
            moderation_provider=moderation,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=answer,
        )
        result = await graph.ainvoke(_state())

    done = result["events"][-1]["data"]
    assert done["session_id"] == "c1"
    assert done["answer"] == "grounded answer"
    assert done["documents"][0]["id"] == "d1"
    assert done["moderation"]["is_flagged"] is False
    assert done["usage"] == {"output_tokens": 3}
    assert result["final_response"].model_dump(by_alias=True) == done
    assert answer.calls == 1
    assert moderation.calls == 2


@pytest.mark.asyncio
async def test_finalization_replays_after_commit_window_without_duplicate_done(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class CrashAfterFinalizationCommit(PhaseResultRepository):
        crashed = False

        async def commit(self, **kwargs):  # type: ignore[no-untyped-def]
            result = await super().commit(**kwargs)
            if kwargs["phase"].phase_name == "finalization" and not self.crashed:
                self.crashed = True
                raise RuntimeError("crash after finalization commit")
            return result

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        run = await RunEventRepository(pool).create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        answer = _Answer()
        moderation = _Moderation()
        graph = build_tracer_graph(
            phase_context=_context(
                pool,
                run.run_id,
                run.execution_epoch,
                CrashAfterFinalizationCommit(pool),
            ),
            moderation_provider=moderation,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=answer,
        )
        with pytest.raises(RuntimeError, match="crash after finalization commit"):
            await graph.ainvoke(_state())
        recovered = await graph.ainvoke(_state())

    assert answer.calls == 1
    assert moderation.calls == 2
    assert recovered["events"][-1]["type"] == "done"
    assert sum(event["type"] == "done" for event in recovered["events"]) == 1
    assert len({event["event_key"] for event in recovered["events"]}) == len(
        recovered["events"]
    )
