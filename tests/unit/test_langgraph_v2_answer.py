import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Self
from uuid import UUID, uuid4

import pytest
from pydantic_ai.usage import RunUsage

from app.langgraph_v2.answer import (
    AnswerCancelled,
    AnswerCitation,
    AnswerOutput,
    PydanticAIAnswerActor,
    bind_answer_citations,
    build_answer_actor,
    run_answer,
    split_answer_chunks,
)
from app.langgraph_v2.api import _stream_unseen_events
from app.langgraph_v2.history import ConversationTurn
from app.langgraph_v2.phase_results import PhaseExecutionContext
from app.langgraph_v2.run_events import EventRecord
from app.models.domain import Document


class _FakeStructuredStream:
    def __init__(
        self,
        snapshots: list[AnswerOutput],
        output: AnswerOutput,
        *,
        block_after_snapshots: bool = False,
    ) -> None:
        self.snapshots = snapshots
        self.output = output
        self.block_after_snapshots = block_after_snapshots
        self.entered = False
        self.exited = False
        self.debounce_by: float | None = 0.1
        self.released = asyncio.Event()

    async def __aenter__(self) -> Self:
        self.entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exited = True
        self.released.set()

    async def stream_output(self, *, debounce_by: float | None):
        self.debounce_by = debounce_by
        for snapshot in self.snapshots:
            yield snapshot
        if self.block_after_snapshots:
            await self.released.wait()

    async def get_output(self) -> AnswerOutput:
        return self.output

    def usage(self) -> RunUsage:
        return RunUsage(input_tokens=5, output_tokens=7)


class _FakeStructuredAgent:
    def __init__(self, stream: _FakeStructuredStream) -> None:
        self.stream = stream
        self.prompt: str | None = None
        self.kwargs: dict[str, object] = {}

    def run_stream(self, prompt: str, **kwargs: object) -> _FakeStructuredStream:
        self.prompt = prompt
        self.kwargs = kwargs
        return self.stream


class _FakePhaseRepository:
    def __init__(self) -> None:
        self.committed = False
        self.candidate: object | None = None

    async def get_or_invoke(self, **kwargs: object) -> object:
        phase = await kwargs["invoke"]()  # type: ignore[index]
        self.candidate = phase
        self.committed = True
        return SimpleNamespace(
            normalized_result=phase.normalized_result,  # type: ignore[attr-defined]
            events=[],
        )


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


@pytest.mark.asyncio
async def test_pydantic_ai_answer_actor_streams_answer_field_deltas_and_final_result() -> (
    None
):
    final = AnswerOutput(
        answer="Hello, world!",
        citations=[AnswerCitation(index=1, quoted_text="evidence")],
    )
    stream = _FakeStructuredStream(
        [AnswerOutput(answer="Hel"), AnswerOutput(answer="Hello, "), final],
        final,
    )
    agent = _FakeStructuredAgent(stream)
    actor = PydanticAIAnswerActor(agent)  # type: ignore[arg-type]
    history = [ConversationTurn(user="previous", assistant="answer")]

    chunks = [
        chunk
        async for chunk in actor.answer_stream(
            "question", [Document(id="d1", content="evidence")], history
        )
    ]

    assert [chunk.delta for chunk in chunks if chunk.delta] == ["Hel", "lo, ", "world!"]
    result = next(chunk.result for chunk in chunks if chunk.result is not None)
    assert result.answer == final.answer
    assert result.citations == final.citations
    assert result.usage["input_tokens"] == 5
    assert stream.debounce_by is None
    assert agent.kwargs["message_history"]


@pytest.mark.asyncio
async def test_pydantic_ai_answer_actor_stream_cancellation_closes_agent_context() -> (
    None
):
    stream = _FakeStructuredStream(
        [AnswerOutput(answer="partial")],
        AnswerOutput(answer="partial"),
        block_after_snapshots=True,
    )
    actor = PydanticAIAnswerActor(_FakeStructuredAgent(stream))  # type: ignore[arg-type]
    answer_stream = actor.answer_stream("question", [])

    first = await answer_stream.__anext__()
    assert first.delta == "partial"
    pending = asyncio.create_task(answer_stream.__anext__())
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert stream.entered is True
    assert stream.exited is True


@pytest.mark.asyncio
async def test_run_answer_writes_real_deltas_and_returns_same_complete_answer() -> None:
    final = AnswerOutput(answer="Hello, world!")
    stream = _FakeStructuredStream(
        [AnswerOutput(answer="Hello"), final],
        final,
    )
    actor = PydanticAIAnswerActor(_FakeStructuredAgent(stream))  # type: ignore[arg-type]
    repository = _FakePhaseRepository()
    context = PhaseExecutionContext(
        repository=repository,  # type: ignore[arg-type]
        tenant_id="tenant-a",
        run_id=uuid4(),
        owner_instance_id="instance-a",
        execution_epoch=1,
    )
    public_events: list[dict[str, object]] = []

    _, result, halted, error = await run_answer(
        {"query": "question"},
        context=context,
        artifacts=object(),  # type: ignore[arg-type]
        actor=actor,
        stream_writer=public_events.append,
    )

    assert halted is False
    assert error is None
    assert result is not None
    assert result.answer == "Hello, world!"
    assert [event["data"] for event in public_events if event["type"] == "token"] == [
        "Hello",
        ", world!",
    ]
    assert (
        "".join(event["data"] for event in public_events if event["type"] == "token")
        == result.answer
    )
    assert repository.committed is True


@pytest.mark.asyncio
async def test_run_answer_cancellation_after_delta_does_not_commit_partial_result() -> (
    None
):
    final = AnswerOutput(answer="Hello, world!")
    stream = _FakeStructuredStream(
        [AnswerOutput(answer="Hello"), final],
        final,
    )
    actor = PydanticAIAnswerActor(_FakeStructuredAgent(stream))  # type: ignore[arg-type]
    repository = _FakePhaseRepository()
    checks = 0

    async def cancellation_check() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    context = PhaseExecutionContext(
        repository=repository,  # type: ignore[arg-type]
        tenant_id="tenant-a",
        run_id=uuid4(),
        owner_instance_id="instance-a",
        execution_epoch=1,
        cancellation_check=cancellation_check,
    )

    with pytest.raises(AnswerCancelled):
        await run_answer(
            {"query": "question"},
            context=context,
            artifacts=object(),  # type: ignore[arg-type]
            actor=actor,
            stream_writer=lambda _: None,
        )

    assert repository.committed is False
    assert stream.exited is True


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
            {"query": "hello"},
            context=context,
            artifacts=object(),
            actor=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_live_answer_chunks_use_the_configured_fake_clock_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = UUID("00000000-0000-0000-0000-000000000001")
    created_at = datetime.now(UTC)
    events = [
        EventRecord(
            tenant_id="tenant-a",
            run_id=run_id,
            sequence=index + 1,
            event_key=f"phase:answer:token:{index}",
            type="token",
            step="llm:answer",
            data=chunk,
            created_at=created_at,
        )
        for index, chunk in enumerate(("one", "two"))
    ]

    class Repository:
        async def list_events(
            self, tenant_id: str, requested_run_id: UUID
        ) -> list[EventRecord]:
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
            Repository(),
            tenant_id="tenant-a",
            run_id=run_id,
            sent_keys=sent,
            answer_chunk_count=[0],
            answer_chunk_interval_ms=250,
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


def test_answer_citation_quote_must_match_bound_document() -> None:
    citations = bind_answer_citations(
        [AnswerCitation(index=1, quoted_text="not in evidence")],
        [{"artifact_id": "artifact-1", "artifact_type": "document"}],
        [Document(id="doc-1", content="actual evidence")],
    )

    assert citations[0].evidence_id == "artifact-1"
    assert citations[0].attribution_status == "unlocated"
    assert citations[0].quoted_text is None


def test_answer_chunk_golden_case() -> None:
    fixture = json.loads(
        (
            Path(__file__).parents[1]
            / "fixtures"
            / "langgraph_v2"
            / "v2_answer_wire.json"
        ).read_text()
    )

    assert split_answer_chunks(fixture["answer"]) == fixture["chunks"]


@pytest.mark.asyncio
async def test_answer_tokens_are_delivered_before_post_moderation_boundary() -> None:
    run_id = UUID("00000000-0000-0000-0000-000000000002")
    created_at = datetime.now(UTC)
    events = [
        EventRecord(
            tenant_id="tenant-a",
            run_id=run_id,
            sequence=1,
            event_key="phase:answer:token:0",
            type="token",
            data="answer",
            created_at=created_at,
        ),
        EventRecord(
            tenant_id="tenant-a",
            run_id=run_id,
            sequence=2,
            event_key="phase:moderation:post:step_start:1",
            type="step_start",
            step="moderation:post",
            created_at=created_at,
        ),
    ]

    class Repository:
        async def list_events(
            self, tenant_id: str, requested_run_id: UUID
        ) -> list[EventRecord]:
            assert tenant_id == "tenant-a"
            assert requested_run_id == run_id
            return events

    frames = [
        frame
        async for frame in _stream_unseen_events(
            Repository(),
            tenant_id="tenant-a",
            run_id=run_id,
            sent_keys=set(),
            answer_chunk_count=[0],
            answer_chunk_interval_ms=250,
        )
    ]

    assert frames[0].index('"type": "token"') < frames[1].index(
        '"step": "moderation:post"'
    )
