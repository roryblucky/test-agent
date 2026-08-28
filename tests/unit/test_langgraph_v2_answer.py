import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic_ai.usage import RunUsage

from app.langgraph_v2.answer import (
    AnswerCancelled,
    AnswerOutput,
    PydanticAIAnswerActor,
    build_answer_actor,
    run_answer,
    split_answer_chunks,
)
from app.langgraph_v2.api import _stream_unseen_events
from app.langgraph_v2.phase_results import PhaseExecutionContext
from app.langgraph_v2.run_events import EventRecord
from app.models.domain import Document


class _FakeAgent:
    def __init__(self) -> None:
        self.prompt: str | None = None

    async def run(self, prompt: str) -> SimpleNamespace:
        self.prompt = prompt
        return SimpleNamespace(
            output=AnswerOutput(answer="structured answer"),
            usage=lambda: RunUsage(input_tokens=2, output_tokens=3),
        )


@pytest.mark.asyncio
async def test_pydantic_ai_answer_actor_passes_only_ordered_evidence() -> None:
    agent = _FakeAgent()
    actor = PydanticAIAnswerActor(agent)  # type: ignore[arg-type]

    result = await actor.answer("question", [Document(id="d1", content="evidence")])

    assert result.answer == "structured answer"
    assert result.usage["input_tokens"] == 2
    assert agent.prompt == "Question: question\n\nEvidence:\n[1] evidence"


def test_build_answer_actor_uses_registry_model_and_output_type() -> None:
    agent = _FakeAgent()

    class Registry:
        def create_agent(self, model_name: str, **kwargs: object) -> _FakeAgent:
            assert model_name == "pro"
            assert kwargs["output_type"] is AnswerOutput
            assert kwargs["instructions"] == "custom"
            return agent

    built = build_answer_actor(Registry(), instructions="custom")

    assert isinstance(built, PydanticAIAnswerActor)


@pytest.mark.asyncio
async def test_answer_checks_cancellation_before_publication() -> None:
    class Repository:
        async def get_or_invoke(self, **kwargs: object) -> object:
            raise AssertionError("cancelled answer must not invoke the repository")

    context = PhaseExecutionContext(
        repository=Repository(),  # type: ignore[arg-type]
        tenant_id="tenant-a",
        run_id=uuid4(),
        owner_instance_id="instance-a",
        execution_epoch=1,
        cancellation_check=lambda: asyncio.sleep(0, result=True),
    )

    with pytest.raises(AnswerCancelled):
        await run_answer(
            {"query": "hello"}, context=context, artifacts=object(), actor=object()  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_live_answer_chunks_use_the_configured_fake_clock_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = UUID("00000000-0000-0000-0000-000000000001")
    created_at = datetime.now(UTC)
    events = [
        EventRecord(
            tenant_id="tenant-a", run_id=run_id, sequence=index + 1,
            event_key=f"phase:answer:token:{index}", type="token",
            step="llm:answer", data=chunk, created_at=created_at,
        )
        for index, chunk in enumerate(("one", "two"))
    ]

    class Repository:
        async def list_events(self, tenant_id: str, requested_run_id: UUID) -> list[EventRecord]:
            assert tenant_id == "tenant-a"
            assert requested_run_id == run_id
            return events

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("app.langgraph_v2.api.asyncio.sleep", fake_sleep)
    sent: set[str] = set()
    chunks = [
        frame
        async for frame in _stream_unseen_events(
            Repository(), tenant_id="tenant-a", run_id=run_id,
            sent_keys=sent, answer_chunk_count=[0], answer_chunk_interval_ms=250,
        )
    ]

    assert len(chunks) == 2
    assert sleeps == [0.25]


def test_split_answer_chunks_preserves_text_and_prefers_boundaries() -> None:
    answer = "First sentence. Second sentence\nThird; final"

    chunks = split_answer_chunks(answer)

    assert chunks == ["First sentence.", " Second sentence\n", "Third;", " final"]
    assert "".join(chunks) == answer


def test_split_answer_chunks_hard_splits_at_unicode_codepoint_limit() -> None:
    answer = "界" * 241

    chunks = split_answer_chunks(answer)

    assert [len(chunk) for chunk in chunks] == [240, 1]
    assert "".join(chunks) == answer


def test_split_answer_chunks_normalizes_only_crlf() -> None:
    answer = "A\r\nB\rC"

    chunks = split_answer_chunks(answer)

    assert chunks == ["A\n", "B\rC"]
    assert "".join(chunks) == "A\nB\rC"
