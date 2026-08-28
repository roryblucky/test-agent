from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerResult
from app.langgraph_v2.artifacts import ArtifactRepository
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.graph import build_tracer_graph
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.pre_moderation import ModerationDecision
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.langgraph_v2.run_events import EventInput, RunEventRepository
from app.models.domain import Document


class _Retriever:
    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(documents=[Document(id="d1", content=query)])


class _Ranker:
    async def rank(self, query: str, documents: list[Document]) -> RerankingResult:
        return RerankingResult(documents=documents)


class _Answer:
    async def answer(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AnswerResult:
        del history
        del query, documents
        return AnswerResult(answer="generated answer")


class _SafeModeration:
    calls = 0

    async def check(self, text: str) -> ModerationDecision:
        self.calls += 1
        assert text in {"hello", "generated answer"}
        return ModerationDecision(is_flagged=False)


class _FlaggingModeration:
    calls = 0

    async def check(self, text: str) -> ModerationDecision:
        self.calls += 1
        assert text in {"hello", "generated answer"}
        return ModerationDecision(
            is_flagged=text == "generated answer", reason="unsafe output"
        )


def _state() -> dict[str, object]:
    return {
        "query": "hello",
        "conversation_id": "c1",
        "client_request_id": None,
        "events": [],
    }


@pytest.mark.asyncio
async def test_safe_answer_passes_post_moderation_unchanged(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        messages = ConversationMessageRepository(pool)
        await messages.resolve_conversation(tenant_id="tenant-a", conversation_id="c1")
        await messages.persist_user_message(
            tenant_id="tenant-a",
            conversation_id="c1",
            run_id=run.run_id,
            content="hello",
            idempotency_key=f"run:{run.run_id}:user",
        )
        moderation = _SafeModeration()
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
            moderation_provider=moderation,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
        )
        result = await graph.ainvoke(_state())
        done = result["events"][-1]
        await runs.complete_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            event=EventInput(
                event_key=done["event_key"],
                type=done["type"],
                data=done["data"],
            ),
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        await messages.persist_assistant_message_after_completion(
            tenant_id="tenant-a",
            conversation_id="c1",
            run_id=run.run_id,
            content=result["answer"],
            idempotency_key=f"run:{run.run_id}:assistant",
        )
        persisted_messages = await messages.list_messages("tenant-a", "c1")

    assert moderation.calls == 2
    assert result["answer"] == "generated answer"
    assert result["post_moderation"]["is_flagged"] is False
    assert [
        event["step"]
        for event in result["events"]
        if event.get("step") == "moderation:post"
    ] == ["moderation:post", "moderation:post"]
    assert result["events"][-1]["type"] == "done"
    assert result["events"][-1]["data"]["answer"] == "generated answer"
    assert [message.content for message in persisted_messages] == [
        "hello",
        "generated answer",
    ]


@pytest.mark.asyncio
async def test_flagged_answer_is_replaced_only_in_final_state(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        messages = ConversationMessageRepository(pool)
        await messages.resolve_conversation(tenant_id="tenant-a", conversation_id="c1")
        await messages.persist_user_message(
            tenant_id="tenant-a",
            conversation_id="c1",
            run_id=run.run_id,
            content="hello",
            idempotency_key=f"run:{run.run_id}:user",
        )
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
            moderation_provider=_FlaggingModeration(),
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
        )
        result = await graph.ainvoke(_state())
        done = result["events"][-1]
        await runs.complete_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            event=EventInput(
                event_key=done["event_key"],
                type=done["type"],
                data=done["data"],
            ),
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        await messages.persist_assistant_message_after_completion(
            tenant_id="tenant-a",
            conversation_id="c1",
            run_id=run.run_id,
            content=result["answer"],
            idempotency_key=f"run:{run.run_id}:assistant",
        )
        persisted_messages = await messages.list_messages("tenant-a", "c1")

    assert result["answer"] == (
        "The generated response was flagged by content moderation and has been removed."
    )
    assert any(
        event["type"] == "token" and event["data"] == "generated answer"
        for event in result["events"]
    )
    assert result["events"][-1]["type"] == "done"
    assert result["events"][-1]["data"]["answer"] == result["answer"]
    assert [message.content for message in persisted_messages] == [
        "hello",
        "The generated response was flagged by content moderation and has been removed.",
    ]
    assert all(message.content != "generated answer" for message in persisted_messages)


@pytest.mark.asyncio
async def test_post_moderation_replays_after_commit_window_crash(
    langgraph_v2_migrated_database_url: str,
) -> None:
    class CrashAfterPostModerationCommit(PhaseResultRepository):
        crashed = False

        async def commit(self, **kwargs):  # type: ignore[no-untyped-def]
            result = await super().commit(**kwargs)
            if kwargs["phase"].phase_name == "post_moderation" and not self.crashed:
                self.crashed = True
                raise RuntimeError("crash after post moderation commit")
            return result

    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        runs = RunEventRepository(pool)
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="c1",
            owner_instance_id="i1",
        )
        moderation = _FlaggingModeration()
        context = PhaseExecutionContext(
            repository=CrashAfterPostModerationCommit(pool),
            artifact_repository=ArtifactRepository(pool),
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
        )
        graph = build_tracer_graph(
            phase_context=context,
            moderation_provider=moderation,
            retriever=_Retriever(),
            ranker=_Ranker(),
            answer_actor=_Answer(),
        )
        with pytest.raises(RuntimeError, match="crash after post moderation commit"):
            await graph.ainvoke(_state())
        recovered = await graph.ainvoke(_state())
        events = await runs.list_events("tenant-a", run.run_id)

    assert moderation.calls == 2
    assert recovered["answer"].startswith("The generated response was flagged")
    assert len({event.event_key for event in events}) == len(events)
    assert (
        sum(
            event.event_key == "phase:post_moderation:step_completed:1"
            for event in events
        )
        == 1
    )
