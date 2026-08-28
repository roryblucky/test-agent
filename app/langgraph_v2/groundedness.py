"""Advisory groundedness evaluator for the v2 linear graph."""

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
from app.models.domain import Document, GroundednessResult


class GroundednessOutput(BaseModel):
    """Validated evaluator output with a bounded score."""

    is_grounded: bool
    score: float = Field(ge=0, le=1)
    details: str | None = None


class GroundednessActor(Protocol):
    """PydanticAI-backed seam for advisory answer evaluation."""

    async def evaluate(
        self,
        answer: str,
        documents: list[Document],
    ) -> GroundednessResult:
        """Return a structured groundedness assessment."""
        ...


class PydanticAIGroundednessActor:
    """Adapt a PydanticAI Agent to the groundedness actor protocol."""

    def __init__(self, agent: Agent[Any, GroundednessOutput]) -> None:
        self._agent = agent

    async def evaluate(self, answer: str, documents: list[Document]) -> GroundednessResult:
        """Evaluate the answer against the supplied evidence text."""
        evidence = "\n\n".join(document.content for document in documents)
        result = await self._agent.run(f"Answer:\n{answer}\n\nEvidence:\n{evidence}")
        usage = result.usage()
        usage_payload = asdict(usage) if is_dataclass(usage) else dict(vars(usage))
        return GroundednessResult(
            **result.output.model_dump(),
            usage=usage_payload,
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
        instructions=instructions or "Assess whether the answer is supported by the evidence.",
    )
    return PydanticAIGroundednessActor(agent)


async def run_groundedness(
    state: Mapping[str, Any],
    *,
    context: PhaseExecutionContext,
    artifacts: ArtifactStore,
    actor: GroundednessActor,
) -> tuple[list[EventRecord], GroundednessResult | None, bool, str | None]:
    """Evaluate an answer once and journal the advisory result atomically."""

    async def invoke() -> PhaseResultInput:
        try:
            citations = state.get("citations", [])
            cited_ids = {
                citation.evidence_id
                if hasattr(citation, "evidence_id")
                else citation.get("evidence_id")
                for citation in citations
            }
            refs = [
                ref
                for ref in state.get("ranked_refs", state.get("artifact_refs", []))
                if ref.get("artifact_type") == "document"
                and ref.get("artifact_id") in cited_ids
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
            actor_result = await actor.evaluate(state.get("answer", ""), documents)
            raw_result = actor_result.model_dump()
            output = GroundednessOutput.model_validate(raw_result)
            result = GroundednessResult.model_validate(
                {**output.model_dump(), "usage": raw_result.get("usage", {})}
            )
            return PhaseResultInput(
                phase_name="groundedness",
                normalized_result=result.model_dump(),
                events=(
                    EventInput(
                        event_key="phase:groundedness:step_start:1",
                        type="step_start",
                        step="groundedness",
                    ),
                    EventInput(
                        event_key="phase:groundedness:step_completed:1",
                        type="step_completed",
                        step="groundedness",
                        data=result.model_dump(),
                    ),
                ),
            )
        except Exception as exc:
            message = str(exc) or "Groundedness evaluation failed."
            return PhaseResultInput(
                phase_name="groundedness",
                normalized_result={"failed": True, "error": message},
                events=(
                    EventInput(
                        event_key="phase:groundedness:step_start:1",
                        type="step_start",
                        step="groundedness",
                    ),
                    EventInput(
                        event_key="phase:groundedness:error:1",
                        type="error",
                        data=message,
                    ),
                ),
                terminal_status="failed",
            )

    result = await context.repository.get_or_invoke(
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        owner_instance_id=context.owner_instance_id,
        execution_epoch=context.execution_epoch,
        phase_name="groundedness",
        invoke=invoke,
    )
    if result.normalized_result.get("failed") is True:
        return list(result.events), None, True, str(result.normalized_result["error"])
    return list(result.events), GroundednessResult.model_validate(result.normalized_result), False, None
