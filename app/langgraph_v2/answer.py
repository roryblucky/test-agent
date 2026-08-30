"""Structured PydanticAI answer actor and replay-safe answer phase."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.langgraph_v2.artifacts import ArtifactRef, ArtifactStore
from app.langgraph_v2.history import ConversationTurn, to_model_message_history
from app.langgraph_v2.run_events import CancellationObserved, EventInput
from app.models.domain import Document
from app.models.workflow import CitationReference


class AnswerCitation(BaseModel):
    """A model citation request referring to ordered evidence by index."""

    index: int = Field(ge=1)
    quoted_text: str | None = None


class AnswerOutput(BaseModel):
    """Validated structured answer returned by the PydanticAI actor."""

    answer: str = Field(min_length=1)
    citations: list[AnswerCitation] = Field(default_factory=list[AnswerCitation])


class AnswerResult(BaseModel):
    """Answer output plus stable model usage metadata."""

    answer: str = Field(min_length=1)
    usage: dict[str, Any] = Field(default_factory=dict)
    citations: list[AnswerCitation] = Field(default_factory=list[AnswerCitation])


@dataclass(frozen=True)
class AnswerStreamChunk:
    """One final-answer delta, with the validated result on stream completion."""

    delta: str = ""
    result: AnswerResult | None = None


class BoundAnswerResult(BaseModel):
    """Answer result with citations bound to durable Artifact provenance."""

    answer: str = Field(min_length=1)
    usage: dict[str, Any] = Field(default_factory=dict)
    citations: list[CitationReference] = Field(default_factory=list[CitationReference])


def bind_answer_citations(
    citations: list[AnswerCitation],
    refs: list[ArtifactRef],
    documents: list[Document],
) -> list[CitationReference]:
    """Bind indexed model citations to ranked Artifacts with quote validation."""
    bound: list[CitationReference] = []
    seen_indices: set[int] = set()
    for citation in citations:
        if citation.index > len(refs) or citation.index in seen_indices:
            continue
        seen_indices.add(citation.index)
        ref = refs[citation.index - 1]
        document = documents[citation.index - 1]
        quote = citation.quoted_text
        located = bool(quote and quote in document.content)
        bound.append(
            CitationReference(
                index=citation.index,
                evidence_id=ref["artifact_id"],
                source=document.source_url or document.id,
                source_type=document.source_type,
                title=document.section_title,
                url=document.source_url,
                snippet=quote if located else None,
                quoted_text=quote if located else None,
                quoted_passages=[quote] if located and quote else [],
                page_number=document.page_number,
                section=document.section_title,
                attribution_status="located" if located else "unlocated",
                metadata={"artifact_id": ref["artifact_id"]},
            )
        )
    return bound


async def build_inline_citations(
    answer: str,
    refs: list[ArtifactRef],
    documents: list[Document],
) -> list[CitationReference]:
    """Extract ``[n]`` references and map them through ranked ArtifactRefs."""
    citations: list[CitationReference] = []
    for match in re.finditer(r"\[([1-9]\d*)\]", answer):
        index = int(match.group(1))
        if index > len(refs) or any(citation.index == index for citation in citations):
            continue
        ref = refs[index - 1]
        document = documents[index - 1]
        source = str(
            document.metadata.get("source") or document.source_url or document.id
        )
        citations.append(
            CitationReference(
                index=index,
                evidence_id=ref["artifact_id"],
                source=source,
                source_type=document.source_type,
                title=document.section_title,
                url=document.source_url,
                snippet=document.content[:300],
                page_number=document.page_number,
                section=document.section_title,
                highlight_content=document.content,
                metadata={"artifact_id": ref["artifact_id"]},
            )
        )
    return citations


class AnswerActor(Protocol):
    """PydanticAI-backed seam for streaming an answer from evidence."""

    def answer_stream(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn],
    ) -> AsyncIterator[AnswerStreamChunk]:
        """Yield final-answer deltas and the complete validated result."""
        ...


class AnswerCancelled(CancellationObserved):
    """Raised when cancellation is observed before answer publication."""


class PydanticAIAnswerActor:
    """Adapt a PydanticAI Agent with structured answer output."""

    def __init__(self, agent: Agent[Any, AnswerOutput]) -> None:
        self._agent = agent

    @staticmethod
    def _prompt(
        query: str,
        documents: list[Document],
    ) -> str:
        evidence = "\n\n".join(
            f"[{index}] {document.content}"
            for index, document in enumerate(documents, 1)
        )
        return f"Question: {query}\n\nEvidence:\n{evidence}"

    @staticmethod
    def _usage_payload(result: Any) -> dict[str, Any]:
        usage = result.usage()
        if is_dataclass(usage) and not isinstance(usage, type):
            return asdict(usage)
        return dict(vars(usage))

    async def answer_stream(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[AnswerStreamChunk]:
        """Stream final ``AnswerOutput.answer`` deltas and return its full result.

        PydanticAI's structured stream yields progressively validated snapshots,
        rather than text deltas.  The answer field is append-only for a normal
        structured output stream, so each public chunk is the newly produced
        suffix.  Tool, reasoning, and other model events never enter this path.
        """
        prompt = self._prompt(query, documents)
        run_kwargs: dict[str, Any] = {}
        if history:
            run_kwargs["message_history"] = to_model_message_history(history)

        async with self._agent.run_stream(prompt, **run_kwargs) as stream:
            previous = ""
            async for partial in stream.stream_output(debounce_by=None):
                answer = getattr(partial, "answer", None)
                if not isinstance(answer, str):
                    continue
                if not answer.startswith(previous):
                    raise ValueError("streamed Answer field is not append-only")
                delta = answer[len(previous) :]
                previous = answer
                if delta:
                    yield AnswerStreamChunk(delta=delta)

            output = AnswerOutput.model_validate(await stream.get_output())
            result = AnswerResult(
                answer=output.answer,
                usage=self._usage_payload(stream),
                citations=output.citations,
            )
            yield AnswerStreamChunk(result=result)


def build_answer_actor(
    registry: Any,
    *,
    model_name: str = "pro",
    instructions: str | None = None,
) -> PydanticAIAnswerActor:
    """Create a role-configured PydanticAI answer actor."""
    agent = registry.create_agent(
        model_name,
        output_type=AnswerOutput,
        instructions=instructions
        or "Answer using only the supplied evidence and return structured JSON.",
    )
    return PydanticAIAnswerActor(agent)


async def run_answer(
    state: Mapping[str, Any],
    *,
    tenant_id: str,
    cancellation_check: Callable[[], Awaitable[bool]] | None,
    artifacts: ArtifactStore,
    actor: AnswerActor,
    stream_writer: Any | None = None,
) -> tuple[list[EventInput], BoundAnswerResult | None, bool, str | None]:
    """Hydrate ranked evidence and stream one checkpoint-owned Answer result."""

    def write_event(event: EventInput) -> None:
        if stream_writer is not None:
            stream_writer(
                {
                    **event.model_dump(exclude_none=True),
                    "journal_policy": "checkpoint_only",
                }
            )

    if cancellation_check is not None and await cancellation_check():
        raise AnswerCancelled("answer generation cancelled before publication")
    try:
        refs = [
            ref
            for ref in state.get("ranked_refs", state.get("artifact_refs", []))
            if ref.get("artifact_type") == "document"
        ]
        documents = [
            Document.model_validate(
                (
                    await artifacts.get(
                        tenant_id=tenant_id,
                        artifact_id=UUID(ref["artifact_id"]),
                    )
                ).payload
            )
            for ref in refs
        ]
        history = [
            ConversationTurn.model_validate(turn) for turn in state.get("history", [])
        ]
        answer_query = state.get("refined_query", state["query"])
        events: list[EventInput] = [
            EventInput(
                event_key="phase:answer:step_start:1",
                type="step_start",
                step="llm:answer",
            )
        ]

        chunks: list[str] = []
        streamed_result: AnswerResult | None = None
        answer_started = False
        answer_iterator = actor.answer_stream(answer_query, documents, history)
        try:
            async for chunk in answer_iterator:
                if chunk.result is not None:
                    streamed_result = chunk.result
                if not chunk.delta:
                    continue
                if (
                    cancellation_check is not None
                    and await cancellation_check()
                ):
                    raise AnswerCancelled(
                        "answer generation cancelled before publication"
                    )
                if not answer_started:
                    answer_started = True
                    write_event(events[0])
                chunks.append(chunk.delta)
                event = EventInput(
                    event_key=f"phase:answer:token:{len(chunks) - 1}",
                    type="token",
                    data=chunk.delta,
                )
                events.append(event)
                write_event(event)
        finally:
            close = getattr(answer_iterator, "aclose", None)
            if close is not None:
                close_task = asyncio.ensure_future(close())
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError:
                    await asyncio.shield(close_task)
                    raise
        if streamed_result is None:
            raise ValueError("answer stream did not return a final result")
        validated = AnswerResult.model_validate(streamed_result)
        normalized_answer = validated.answer
        if "".join(chunks) != normalized_answer:
            raise ValueError("streamed Answer deltas do not match final Answer")
        citations = await build_inline_citations(normalized_answer, refs, documents)
        if "[" not in normalized_answer and "]" not in normalized_answer:
            citations = bind_answer_citations(validated.citations, refs, documents)
        if citations:
            event = EventInput(
                event_key="phase:answer:citations:1",
                type="citations",
                data=[citation.model_dump(mode="json") for citation in citations],
            )
            events.append(event)
            write_event(event)
        event = EventInput(
            event_key="phase:answer:step_completed:1",
            type="step_completed",
            step="llm:answer",
            data={"chunk_count": len(chunks)},
        )
        events.append(event)
        write_event(event)
        if (
            cancellation_check is not None
            and await cancellation_check()
        ):
            raise AnswerCancelled("answer generation cancelled before publication")
        return (
            events,
            BoundAnswerResult(
                answer=normalized_answer,
                usage=validated.usage,
                citations=citations,
            ),
            False,
            None,
        )
    except AnswerCancelled:
        raise
    except Exception as exc:
        message = str(exc) or "Answer generation failed."
        start_event = EventInput(
            event_key="phase:answer:step_start:1",
            type="step_start",
            step="llm:answer",
        )
        error_event = EventInput(
            event_key="phase:answer:error:1",
            type="error",
            data=message,
        )
        write_event(start_event)
        write_event(error_event)
        return (
            [
                start_event,
                error_event,
            ],
            None,
            True,
            message,
        )
