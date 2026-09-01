"""Structured PydanticAI answer actor and replay-safe answer phase."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.langgraph_v2.contracts import LiveStreamEvent
from app.langgraph_v2.evidence import Evidence
from app.langgraph_v2.history import ConversationExchange, to_model_message_history
from app.langgraph_v2.model_usage import model_usage_payload
from app.langgraph_v2.stream import await_task_completion
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
    """Answer result with citations bound to request-local evidence."""

    answer: str = Field(min_length=1)
    usage: dict[str, Any] = Field(default_factory=dict)
    citations: list[CitationReference] = Field(default_factory=list[CitationReference])


def bind_answer_citations(
    citations: list[AnswerCitation],
    evidence: list[Evidence],
) -> list[CitationReference]:
    """Bind indexed model citations to ranked evidence with quote validation."""
    bound: list[CitationReference] = []
    seen_indices: set[int] = set()
    for citation in citations:
        if citation.index > len(evidence) or citation.index in seen_indices:
            continue
        seen_indices.add(citation.index)
        item = evidence[citation.index - 1]
        document = item.document
        quote = citation.quoted_text
        located = bool(quote and quote in document.content)
        bound.append(
            CitationReference(
                index=citation.index,
                evidence_id=item.evidence_id,
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
                metadata={"document_id": document.id},
            )
        )
    return bound


async def build_inline_citations(
    answer: str,
    evidence: list[Evidence],
) -> list[CitationReference]:
    """Extract ``[n]`` references and map them through ranked evidence."""
    citations: list[CitationReference] = []
    for match in re.finditer(r"\[([1-9]\d*)\]", answer):
        index = int(match.group(1))
        if index > len(evidence) or any(
            citation.index == index for citation in citations
        ):
            continue
        item = evidence[index - 1]
        document = item.document
        source = str(
            document.metadata.get("source") or document.source_url or document.id
        )
        citations.append(
            CitationReference(
                index=index,
                evidence_id=item.evidence_id,
                source=source,
                source_type=document.source_type,
                title=document.section_title,
                url=document.source_url,
                snippet=document.content[:300],
                page_number=document.page_number,
                section=document.section_title,
                highlight_content=document.content,
                metadata={"document_id": document.id},
            )
        )
    return citations


class AnswerActor(Protocol):
    """PydanticAI-backed seam for streaming an answer from evidence."""

    def answer_stream(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange],
    ) -> AsyncIterator[AnswerStreamChunk]:
        """Yield final-answer deltas and the complete validated result."""
        ...


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

    async def answer_stream(
        self,
        query: str,
        documents: list[Document],
        history: Sequence[ConversationExchange] = (),
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
                usage=model_usage_payload(stream),
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
    actor: AnswerActor,
    stream_writer: Any | None = None,
) -> tuple[list[LiveStreamEvent], BoundAnswerResult | None, bool, str | None]:
    """Stream one Answer from request-local ranked evidence."""

    def write_event(event: LiveStreamEvent) -> None:
        if stream_writer is not None:
            stream_writer(event.to_stream_payload())

    answer_started = False
    try:
        evidence = [Evidence.model_validate(item) for item in state["ranked_evidence"]]
        documents = [item.document for item in evidence]
        history = [
            ConversationExchange.model_validate(exchange) for exchange in state.get("history", [])
        ]
        refined_query = state.get("refined_query")
        answer_query = (
            refined_query if isinstance(refined_query, str) else state["query"]
        )
        events: list[LiveStreamEvent] = [
            LiveStreamEvent(
                type="step_start",
                step="llm:answer",
            )
        ]

        chunks: list[str] = []
        streamed_result: AnswerResult | None = None
        answer_iterator = actor.answer_stream(answer_query, documents, history)
        try:
            async for chunk in answer_iterator:
                if chunk.result is not None:
                    streamed_result = chunk.result
                if not chunk.delta:
                    continue
                if not answer_started:
                    answer_started = True
                    write_event(events[0])
                chunks.append(chunk.delta)
                event = LiveStreamEvent(
                    type="token",
                    data=chunk.delta,
                )
                events.append(event)
                write_event(event)
        finally:
            close = getattr(answer_iterator, "aclose", None)
            if close is not None:
                close_task = asyncio.ensure_future(close())
                cancelled = await await_task_completion(close_task)
                try:
                    await close_task
                except BaseException:
                    if cancelled:
                        raise asyncio.CancelledError
                    raise
                if cancelled:
                    raise asyncio.CancelledError
        if streamed_result is None:
            raise ValueError("answer stream did not return a final result")
        validated = AnswerResult.model_validate(streamed_result)
        normalized_answer = validated.answer
        if "".join(chunks) != normalized_answer:
            raise ValueError("streamed Answer deltas do not match final Answer")
        citations = await build_inline_citations(normalized_answer, evidence)
        if "[" not in normalized_answer and "]" not in normalized_answer:
            citations = bind_answer_citations(validated.citations, evidence)
        if citations:
            event = LiveStreamEvent(
                type="citations",
                data=[citation.model_dump(mode="json") for citation in citations],
            )
            events.append(event)
            write_event(event)
        event = LiveStreamEvent(
            type="step_completed",
            step="llm:answer",
            data={"chunk_count": len(chunks)},
        )
        events.append(event)
        write_event(event)
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
    except Exception as exc:
        message = str(exc) or "Answer generation failed."
        start_event = LiveStreamEvent(
            type="step_start",
            step="llm:answer",
        )
        error_event = LiveStreamEvent(
            type="error",
            data=message,
            checkpoint_terminal=True,
        )
        if not answer_started:
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
