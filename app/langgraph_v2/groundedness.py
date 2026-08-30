"""Advisory groundedness evaluator for the v2 linear graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.langgraph_v2.contracts import LiveStreamEvent
from app.langgraph_v2.evidence import Evidence
from app.langgraph_v2.model_usage import model_usage_payload
from app.langgraph_v2.output_assessments import (
    OutputAssessmentAudit,
    build_output_assessment_scope,
    record_output_assessment,
)
from app.models.domain import Document, GroundednessResult


class GroundednessOutput(BaseModel):
    """Validated evaluator output with a bounded score."""

    is_grounded: bool
    score: float = Field(ge=0, le=1)
    details: str | None = None


class GroundednessAssessment(GroundednessOutput):
    """V2-owned evaluator result carrying model usage metadata."""

    usage: dict[str, Any] = Field(default_factory=dict)


class GroundednessActor(Protocol):
    """PydanticAI-backed seam for advisory answer evaluation."""

    async def evaluate(
        self,
        answer: str,
        documents: list[Document],
    ) -> GroundednessAssessment:
        """Return a structured groundedness assessment."""
        ...


class UnavailableGroundednessActor:
    """Advisory actor used when groundedness setup cannot complete."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def evaluate(
        self,
        answer: str,
        documents: list[Document],
    ) -> GroundednessAssessment:
        """Surface setup failure through the normal advisory phase boundary."""
        del answer, documents
        raise self._error


class PydanticAIGroundednessActor:
    """Adapt a PydanticAI Agent to the groundedness actor protocol."""

    def __init__(self, agent: Agent[Any, GroundednessOutput]) -> None:
        self._agent = agent

    async def evaluate(
        self, answer: str, documents: list[Document]
    ) -> GroundednessAssessment:
        """Evaluate the answer against the supplied evidence text."""
        evidence = "\n\n".join(document.content for document in documents)
        result = await self._agent.run(f"Answer:\n{answer}\n\nEvidence:\n{evidence}")
        return GroundednessAssessment(
            **result.output.model_dump(),
            usage=model_usage_payload(result),
        )


def build_groundedness_actor(
    registry: Any,
    *,
    model_name: str = "fast",
    instructions: str | None = None,
) -> PydanticAIGroundednessActor:
    """Create a role-configured groundedness evaluator."""
    agent = registry.create_agent(
        model_name,
        output_type=GroundednessOutput,
        instructions=instructions
        or "Assess whether the answer is supported by the evidence.",
    )
    return PydanticAIGroundednessActor(agent)


async def run_groundedness(
    state: Mapping[str, Any],
    *,
    tenant_id: str,
    current_turn_id: UUID | None,
    output_assessment_audit: OutputAssessmentAudit | None,
    actor: GroundednessActor,
) -> tuple[
    list[LiveStreamEvent], GroundednessResult | None, dict[str, Any], str | None
]:
    """Evaluate the canonical Answer and record the advisory audit result."""
    assessment_scope = build_output_assessment_scope(
        tenant_id=tenant_id,
        conversation_id=(
            state.get("conversation_id")
            if isinstance(state.get("conversation_id"), str)
            else None
        ),
        turn_id=current_turn_id or state.get("turn_id"),
    )

    try:
        citations = state.get("citations", [])
        cited_ids = {
            citation.evidence_id
            if hasattr(citation, "evidence_id")
            else citation.get("evidence_id")
            for citation in citations
        }
        evidence = [Evidence.model_validate(item) for item in state["ranked_evidence"]]
        documents = [
            item.document for item in evidence if item.evidence_id in cited_ids
        ]
        answer = state.get("answer")
        actor_result = await actor.evaluate(
            answer if isinstance(answer, str) else "",
            documents,
        )
        raw_result = actor_result.model_dump()
        output = GroundednessOutput.model_validate(raw_result)
        result = GroundednessResult.model_validate(output.model_dump())
        normalized_result = {
            **result.model_dump(),
            "usage": raw_result.get("usage", {}),
        }
        await record_output_assessment(
            output_assessment_audit,
            scope=assessment_scope,
            assessment_type="groundedness",
            result=normalized_result,
        )
        return (
            [
                LiveStreamEvent(
                    type="step_start",
                    step="groundedness",
                ),
                LiveStreamEvent(
                    type="step_completed",
                    step="groundedness",
                    data=result.model_dump(),
                ),
            ],
            result,
            actor_result.usage,
            None,
        )
    except Exception as exc:
        message = str(exc) or "Groundedness evaluation failed."
        failed_result = {"failed": True, "error": message}
        await record_output_assessment(
            output_assessment_audit,
            scope=assessment_scope,
            assessment_type="groundedness",
            result=failed_result,
        )
        return (
            [
                LiveStreamEvent(
                    type="step_start",
                    step="groundedness",
                ),
                LiveStreamEvent(
                    type="step_completed",
                    step="groundedness",
                    data=failed_result,
                ),
            ],
            None,
            {},
            message,
        )
