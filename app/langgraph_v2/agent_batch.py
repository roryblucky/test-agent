"""First-batch Task contracts, validation, execution, and acceptance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    EvidenceProvider,
    RequestEvidenceCatalog,
    bind_evidence_tool,
)
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
    evidence_ids: tuple[str, ...] = Field(max_length=16, default=())

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
    evidence_ids: tuple[str, ...] = Field(max_length=16, default=())


class SpecialistAttempt(BaseModel):
    """Actor-local terminal output, including app-only Tool metadata."""

    model_config = ConfigDict(frozen=True)

    finding: SpecialistFindingDraft
    evidence: tuple[EvidenceEnvelope, ...] = ()


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

    async def run(self, input: SpecialistTaskInput) -> SpecialistAttempt:
        """Return one terminal finding plus app-only Tool metadata."""
        ...


EvidenceTool = Callable[[str, str], object]
ToolTelemetry = Callable[[str, str], None]


class SpecialistActorFactory(Protocol):
    """Build one Specialist actor after its Tool set is frozen."""

    def __call__(
        self,
        tools: tuple[EvidenceTool, ...],
        returned_evidence: list[EvidenceEnvelope],
    ) -> SpecialistActor:
        """Return an actor limited to exactly the supplied Tool bindings."""
        ...


@dataclass(frozen=True)
class EvidenceToolRegistration:
    """One code-registered Evidence Tool and its maximum source authority."""

    id: str
    provider: EvidenceProvider
    allowed_sources: frozenset[str]
    allowed_queries: frozenset[str] = frozenset()
    audit: ToolTelemetry | None = None


@dataclass(frozen=True)
class SpecialistRegistration:
    """Typed code registration for one Specialist actor."""

    id: str
    actor: SpecialistActor | None = None
    actor_factory: SpecialistActorFactory | None = None
    allowed_tool_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SpecialistRegistry:
    """Resolve registered and Tenant-eligible Specialists at one deep seam."""

    registrations: Sequence[SpecialistRegistration]
    tenant_eligible_ids: frozenset[str]
    tool_registrations: Sequence[EvidenceToolRegistration] = ()
    tenant_eligible_tool_ids: frozenset[str] = frozenset()

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

    def effective_tool_ids(
        self,
        registration: SpecialistRegistration,
        *,
        scope_tool_ids: frozenset[str],
    ) -> frozenset[str]:
        """Freeze registered, Tenant, Scope, and Specialist Tool authority."""
        registered_ids = frozenset(tool.id for tool in self.tool_registrations)
        return (
            registered_ids
            & self.tenant_eligible_tool_ids
            & scope_tool_ids
            & registration.allowed_tool_ids
        )

    def bind_actor(
        self,
        registration: SpecialistRegistration,
        *,
        scope_tool_ids: frozenset[str],
        scope_sources: frozenset[str],
        scope_queries: frozenset[str],
        tenant_id: str,
        request_id: str,
        task_id: str,
        tool_telemetry: ToolTelemetry | None = None,
    ) -> SpecialistActor:
        """Create one actor with a frozen Scope-narrowed Tool surface."""
        effective_ids = self.effective_tool_ids(
            registration, scope_tool_ids=scope_tool_ids
        )
        if not effective_ids and registration.actor_factory is None:
            if registration.actor is not None:
                return registration.actor
            raise ValueError("Specialist Tool actor factory is not configured")
        actor_factory = registration.actor_factory
        if actor_factory is None:
            raise AssertionError("Specialist actor factory is required")
        registered = {tool.id: tool for tool in self.tool_registrations}
        returned_evidence: list[EvidenceEnvelope] = []
        tools: list[EvidenceTool] = []
        for tool_id in sorted(effective_ids):
            tool = registered[tool_id]

            def report_tool_status(
                status: str,
                *,
                tool_id: str = tool_id,
                audit: ToolTelemetry | None = tool.audit,
            ) -> None:
                if audit is not None:
                    try:
                        audit(tool_id, status)
                    except BaseException:
                        pass
                if tool_telemetry is not None:
                    tool_telemetry(tool_id, status)

            binding = bind_evidence_tool(
                tool.provider,
                allowed_sources=tool.allowed_sources & scope_sources,
                allowed_queries=tool.allowed_queries & scope_queries,
                tenant_id=tenant_id,
                request_id=request_id,
                task_id=task_id,
                returned_evidence=returned_evidence,
                telemetry=report_tool_status,
            )
            binding.__name__ = tool_id
            tools.append(binding)
        return actor_factory(tuple(tools), returned_evidence)


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
    catalog: RequestEvidenceCatalog | None = None,
    tenant_id: str | None = None,
    request_id: str | None = None,
    scope_tool_ids: frozenset[str] = frozenset(),
    scope_sources: frozenset[str] = frozenset(),
    scope_queries: frozenset[str] = frozenset(),
    tool_telemetry: ToolTelemetry | None = None,
) -> BatchContribution:
    """Run one registered no-Tool Specialist and stage only its terminal outcome."""
    registration = registry.resolve(
        task.specialist_id, scope_descriptors=scope_descriptors
    )
    actor = registry.bind_actor(
        registration,
        scope_tool_ids=scope_tool_ids,
        scope_sources=scope_sources,
        scope_queries=scope_queries,
        tenant_id=tenant_id or "",
        request_id=request_id or "",
        task_id=task.id,
        tool_telemetry=tool_telemetry,
    )
    attempt = await actor.run(
        SpecialistTaskInput(task_id=task.id, objective=task.objective)
    )
    draft = attempt.finding
    draft.require_canonical_size()
    if catalog is None:
        if attempt.evidence:
            raise ValueError("Evidence cache is not configured")
    else:
        if tenant_id is None or request_id is None:
            raise ValueError("Evidence cache identity is not configured")
        catalog.accept_referenced(
            attempt.evidence,
            finding_evidence_ids=draft.evidence_ids,
            tenant_id=tenant_id,
            request_id=request_id,
            task_id=task.id,
        )
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
