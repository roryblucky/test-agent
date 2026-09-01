import asyncio
from typing import Any, Self

import pytest
from pydantic_ai.usage import RunUsage

from app.langgraph_v2.answer import (
    AnswerCitation,
    AnswerOutput,
    PydanticAIAnswerActor,
    bind_answer_citations,
    build_answer_actor,
    run_answer,
)
from app.langgraph_v2.conversation_context import ConversationExchange
from app.langgraph_v2.evidence import Evidence
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
        self.cleanup_completed = False
        self.debounce_by: float | None = 0.1
        self.released = asyncio.Event()

    async def __aenter__(self) -> Self:
        self.entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exited = True
        await asyncio.sleep(0)
        self.cleanup_completed = True
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

    def run_stream(self, prompt: str, **kwargs: Any) -> _FakeStructuredStream:
        self.prompt = prompt
        self.kwargs = kwargs
        return self.stream


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
    history = [ConversationExchange(user="previous", assistant="answer")]

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
    pending = asyncio.ensure_future(answer_stream.__anext__())
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert stream.entered is True
    assert stream.exited is True
    assert stream.cleanup_completed is True


@pytest.mark.asyncio
async def test_run_answer_writes_real_deltas_and_returns_same_complete_answer() -> None:
    final = AnswerOutput(answer="Hello, world!")
    stream = _FakeStructuredStream(
        [AnswerOutput(answer="Hello"), final],
        final,
    )
    actor = PydanticAIAnswerActor(_FakeStructuredAgent(stream))  # type: ignore[arg-type]
    public_events: list[dict[str, object]] = []

    _, result, halted, error = await run_answer(
        {"query": "question", "ranked_evidence": []},
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
        "".join(
            str(event["data"]) for event in public_events if event["type"] == "token"
        )
        == result.answer
    )


def test_build_answer_actor_uses_registry_model_and_output_type() -> None:
    agent = _FakeStructuredAgent(
        _FakeStructuredStream([], AnswerOutput(answer="answer"))
    )

    class Registry:
        def create_agent(self, model_name: str, **kwargs: Any) -> _FakeStructuredAgent:
            assert model_name == "pro"
            assert kwargs["output_type"] is AnswerOutput
            assert kwargs["instructions"] == "custom"
            return agent

    built = build_answer_actor(Registry(), instructions="custom")

    assert isinstance(built, PydanticAIAnswerActor)


def test_answer_citation_quote_must_match_bound_document() -> None:
    citations = bind_answer_citations(
        [AnswerCitation(index=1, quoted_text="not in evidence")],
        [
            Evidence(
                evidence_id="evidence-1",
                document=Document(id="doc-1", content="actual evidence"),
            )
        ],
    )

    assert citations[0].evidence_id == "evidence-1"
    assert citations[0].attribution_status == "unlocated"
    assert citations[0].quoted_text is None
