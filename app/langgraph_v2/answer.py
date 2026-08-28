"""Structured PydanticAI answer actor and replay-safe answer phase."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.langgraph_v2.artifacts import ArtifactStore
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultInput
from app.langgraph_v2.run_events import EventInput, EventRecord
from app.models.domain import Document
from app.models.workflow import CitationReference

ANSWER_CHUNK_INTERVAL_MS = 250
ANSWER_CHUNK_MAX_CODEPOINTS = 240
_ANSWER_BOUNDARIES = frozenset(".?!。！？;；\n")


class AnswerOutput(BaseModel):
    """Validated structured answer returned by the PydanticAI actor."""

    answer: str = Field(min_length=1)
    citations: list[AnswerCitation] = Field(default_factory=list)


class AnswerCitation(BaseModel):
    """A model citation request referring to ordered evidence by index."""

    index: int = Field(ge=1)
    quoted_text: str | None = None


class AnswerResult(BaseModel):
    """Answer output plus stable model usage metadata."""

    answer: str = Field(min_length=1)
    usage: dict[str, Any] = Field(default_factory=dict)
    citations: list[AnswerCitation] = Field(default_factory=list)


class BoundAnswerResult(BaseModel):
    """Answer result with citations bound to durable Artifact provenance."""

    answer: str = Field(min_length=1)
    usage: dict[str, Any] = Field(default_factory=dict)
    citations: list[CitationReference] = Field(default_factory=list)


class AnswerActor(Protocol):
    """PydanticAI-backed seam for generating an answer from evidence."""

    async def answer(self, query: str, documents: list[Document]) -> AnswerResult:
        """Return a validated answer for the ordered evidence."""
        ...


class AnswerCancelled(RuntimeError):
    """Raised when cancellation is observed before answer publication."""


class PydanticAIAnswerActor:
    """Adapt a PydanticAI Agent with structured answer output."""

    def __init__(self, agent: Agent[Any, AnswerOutput]) -> None:
        self._agent = agent

    async def answer(self, query: str, documents: list[Document]) -> AnswerResult:
        """Run the model with only the query and ordered Documents as context."""
        evidence = "\n\n".join(
            f"[{index}] {document.content}" for index, document in enumerate(documents, 1)
        )
        prompt = f"Question: {query}\n\nEvidence:\n{evidence}"
        result = await self._agent.run(prompt)
        usage = result.usage()
        usage_payload = asdict(usage) if is_dataclass(usage) else dict(vars(usage))
        output = AnswerOutput.model_validate(result.output)
        return AnswerResult(
            answer=output.answer,
            usage=usage_payload,
            citations=output.citations,
        )


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


def split_answer_chunks(
    answer: str,
    *,
    max_codepoints: int = ANSWER_CHUNK_MAX_CODEPOINTS,
) -> list[str]:
    """Split text at sentence boundaries while preserving its normalized content."""
    if max_codepoints < 1:
        raise ValueError("max_codepoints must be positive")
    normalized = answer.replace("\r\n", "\n")
    chunks: list[str] = []
    current: list[str] = []
    for character in normalized:
        current.append(character)
        if character in _ANSWER_BOUNDARIES or len(current) >= max_codepoints:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    return chunks


async def run_answer(
    state: Mapping[str, Any],
    *,
    context: PhaseExecutionContext,
    artifacts: ArtifactStore,
    actor: AnswerActor,
) -> tuple[list[EventRecord], BoundAnswerResult | None, bool, str | None]:
    """Hydrate ranked evidence, journal the answer and all chunks atomically."""

    async def invoke() -> PhaseResultInput:
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
                            tenant_id=context.tenant_id,
                            artifact_id=UUID(ref["artifact_id"]),
                        )
                    ).payload
                )
                for ref in refs
            ]
            answer = await actor.answer(
                state.get("refined_query", state["query"]), documents
            )
            validated = AnswerResult.model_validate(answer)
            chunks = split_answer_chunks(validated.answer)
            normalized_answer = "".join(chunks)
            citations: list[CitationReference] = []
            seen_citations: set[int] = set()
            for citation in validated.citations:
                if citation.index > len(refs) or citation.index in seen_citations:
                    continue
                seen_citations.add(citation.index)
                ref = refs[citation.index - 1]
                document = documents[citation.index - 1]
                source = document.source_url or document.id
                citations.append(
                    CitationReference(
                        index=citation.index,
                        evidence_id=ref["artifact_id"],
                        source=source,
                        source_type=document.source_type,
                        title=document.section_title,
                        url=document.source_url,
                        snippet=citation.quoted_text,
                        quoted_text=citation.quoted_text,
                        page_number=document.page_number,
                        section=document.section_title,
                        attribution_status=(
                            "located" if citation.quoted_text else "unlocated"
                        ),
                        metadata={"artifact_id": ref["artifact_id"]},
                    )
                )
            events: list[EventInput] = [
                EventInput(
                    event_key="phase:answer:step_start:1",
                    type="step_start",
                    step="llm:answer",
                )
            ]
            events.extend(
                EventInput(
                    event_key=f"phase:answer:token:{index}",
                    type="token",
                    data=chunk,
                )
                for index, chunk in enumerate(chunks)
            )
            if citations:
                events.append(
                    EventInput(
                        event_key="phase:answer:citations:1",
                        type="citations",
                        data=[
                            citation.model_dump(mode="json") for citation in citations
                        ],
                    )
                )
            events.append(
                EventInput(
                    event_key="phase:answer:step_completed:1",
                    type="step_completed",
                    step="llm:answer",
                    data={"chunk_count": len(chunks)},
                )
            )
            return PhaseResultInput(
                phase_name="answer",
                normalized_result={
                    "answer": normalized_answer,
                    "usage": validated.usage,
                    "citations": [citation.model_dump(mode="json") for citation in citations],
                },
                events=tuple(events),
                terminal_status=None,
            )
        except Exception as exc:
            message = str(exc) or "Answer generation failed."
            return PhaseResultInput(
                phase_name="answer",
                normalized_result={"failed": True, "error": message},
                events=(
                    EventInput(
                        event_key="phase:answer:step_start:1",
                        type="step_start",
                        step="llm:answer",
                    ),
                    EventInput(
                        event_key="phase:answer:error:1",
                        type="error",
                        data=message,
                    ),
                ),
                terminal_status="failed",
            )

    async def check_before_commit() -> None:
        if context.cancellation_check is not None and await context.cancellation_check():
            raise AnswerCancelled("answer generation cancelled before publication")

    if context.cancellation_check is not None and await context.cancellation_check():
        raise AnswerCancelled("answer generation cancelled before publication")
    result = await context.repository.get_or_invoke(
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        owner_instance_id=context.owner_instance_id,
        execution_epoch=context.execution_epoch,
        phase_name="answer",
        invoke=invoke,
        before_commit=check_before_commit,
    )
    if result.normalized_result.get("failed") is True:
        return list(result.events), None, True, str(result.normalized_result["error"])
    return (
        list(result.events),
        BoundAnswerResult.model_validate(result.normalized_result),
        False,
        None,
    )
