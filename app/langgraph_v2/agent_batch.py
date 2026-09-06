"""First-batch Task contracts, validation, execution, and acceptance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.langgraph_v2.agent_scope import SpecialistDescriptor

_SPECIALIST_FINDING_MAX_BYTES = 16 * 1024


class StructuredOutputInvalid(ValueError):
    """Reject a returned Specialist draft before it becomes a contribution."""


def _stable_id(prefix: str, value: Mapping[str, object]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}_{hashlib.sha256(canonical).hexdigest()[:32]}"


class TaskProposal(BaseModel):
    """Model-authored bounded request for one registered Specialist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    specialist_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    context_task_ids: tuple[str, ...] = ()


class DispatchBatch(BaseModel):
    """Coordinator decision that requests one first-round Task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["dispatch"]
    tasks: tuple[TaskProposal, ...] = Field(min_length=1, max_length=1)


class SpecialistFindingDraft(BaseModel):
    """Minimal model-authored Specialist terminal output for this ticket."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()

    def canonical_json_size(self) -> int:
        """Return canonical UTF-8 size used by the acceptance boundary."""
        return len(
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )

    def require_canonical_size(self) -> None:
        """Reject an over-limit model output before it becomes a contribution."""
        if self.canonical_json_size() > _SPECIALIST_FINDING_MAX_BYTES:
            raise StructuredOutputInvalid("Specialist finding exceeds 16 KiB")


class SpecialistResult(BaseModel):
    """Accepted minimal Result without Tool metadata or diagnostics."""

    model_config = ConfigDict(frozen=True)

    summary: str
    evidence_ids: tuple[str, ...] = ()


class TaskSucceeded(BaseModel):
    """Platform-owned terminal Task Outcome for accepted Specialist work."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["succeeded"] = "succeeded"
    task_id: str
    result: SpecialistResult


class BatchContribution(BaseModel):
    """One immutable terminal result staged by a Specialist branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str
    task_id: str
    attempt: Literal[1]
    outcome: TaskSucceeded


class AcceptedBatch(BaseModel):
    """One immutable batch promoted only by the barrier."""

    model_config = ConfigDict(frozen=True)

    id: str
    outcomes: tuple[TaskSucceeded, ...]


@dataclass(frozen=True)
class AcceptedTask:
    """Graph-owned identity plus trusted Specialist selection."""

    id: str
    objective: str
    specialist_id: str


@dataclass(frozen=True)
class ActiveBatch:
    """Scalar manifest for the sole Ticket 04 first-round batch."""

    id: str
    tasks: tuple[AcceptedTask, ...]

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Expose stable manifest order to the barrier."""
        return tuple(task.id for task in self.tasks)


class SpecialistActor(Protocol):
    """Execute one bounded Specialist Task without exposing Tools to graph code."""

    async def run(self, input: SpecialistTaskInput) -> SpecialistFindingDraft:
        """Return the model-authored terminal finding draft."""
        ...


@dataclass(frozen=True)
class SpecialistRegistration:
    """Typed code registration for one Specialist actor."""

    id: str
    actor: SpecialistActor


@dataclass(frozen=True)
class SpecialistRegistry:
    """Resolve registered and Tenant-eligible Specialists at one deep seam."""

    registrations: Sequence[SpecialistRegistration]
    tenant_eligible_ids: frozenset[str]

    def resolve(
        self,
        specialist_id: str,
        *,
        scope_descriptors: Sequence[SpecialistDescriptor],
    ) -> SpecialistRegistration:
        """Require registration plus Tenant and current-Scope eligibility."""
        registered = {
            registration.id: registration for registration in self.registrations
        }
        if (
            specialist_id not in registered
            or specialist_id not in self.tenant_eligible_ids
            or specialist_id not in {descriptor.id for descriptor in scope_descriptors}
        ):
            raise ValueError("Specialist is not eligible")
        return registered[specialist_id]


@dataclass(frozen=True)
class SpecialistTaskInput:
    """The only input needed by a first no-Tool Specialist invocation."""

    task_id: str
    objective: str


def accept_initial_dispatch(
    decision: DispatchBatch,
    *,
    request_id: str,
    registry: SpecialistRegistry,
    scope_descriptors: Sequence[SpecialistDescriptor],
) -> ActiveBatch:
    """Validate one first Dispatch and assign graph-owned stable Task identity."""
    proposal = decision.tasks[0]
    if not proposal.objective.strip():
        raise ValueError("Specialist objective must not be blank")
    if proposal.context_task_ids:
        raise ValueError("initial Dispatch cannot select prior Task context")
    registry.resolve(proposal.specialist_id, scope_descriptors=scope_descriptors)
    task_id = _stable_id(
        "task",
        {"request_id": request_id, "round": 1, "dispatch_order": 0},
    )
    batch_id = _stable_id("batch", {"request_id": request_id, "round": 1})
    return ActiveBatch(
        id=batch_id,
        tasks=(
            AcceptedTask(
                id=task_id,
                objective=proposal.objective.strip(),
                specialist_id=proposal.specialist_id,
            ),
        ),
    )


async def execute_specialist(
    task: AcceptedTask,
    *,
    batch_id: str,
    registry: SpecialistRegistry,
    scope_descriptors: Sequence[SpecialistDescriptor],
) -> BatchContribution:
    """Run one registered no-Tool Specialist and stage only its terminal outcome."""
    registration = registry.resolve(
        task.specialist_id, scope_descriptors=scope_descriptors
    )
    draft = await registration.actor.run(
        SpecialistTaskInput(task_id=task.id, objective=task.objective)
    )
    draft.require_canonical_size()
    return BatchContribution(
        batch_id=batch_id,
        task_id=task.id,
        attempt=1,
        outcome=TaskSucceeded(
            task_id=task.id,
            result=SpecialistResult(
                summary=draft.summary,
                evidence_ids=draft.evidence_ids,
            ),
        ),
    )


def promote_batch(
    batch: ActiveBatch,
    contributions: Mapping[str, BatchContribution],
) -> AcceptedBatch:
    """Validate exact manifest membership before atomically promoting a batch."""
    if set(contributions) != set(batch.task_ids):
        raise ValueError("Batch contribution manifest is invalid")
    ordered = tuple(contributions[task_id] for task_id in batch.task_ids)
    if any(
        contribution.batch_id != batch.id
        or contribution.task_id != contribution.outcome.task_id
        or contribution.task_id not in batch.task_ids
        or contribution.attempt != 1
        for contribution in ordered
    ):
        raise ValueError("Batch contribution manifest is invalid")
    return AcceptedBatch(id=batch.id, outcomes=tuple(item.outcome for item in ordered))
