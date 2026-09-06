"""First-batch Task contracts, validation, execution, and acceptance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.langgraph_v2.agent_evidence import (
    DataGapView,
    EvidenceEnvelope,
    EvidenceInvocationContext,
    EvidenceProvider,
    ExpectedToolUnavailability,
    RequestEvidenceCatalog,
    SpecialistToolCapture,
    ToolTelemetryStatus,
    ToolUnavailabilityRecord,
    ToolUnavailableReason,
    bind_evidence_tool,
)
from app.langgraph_v2.agent_scope import SpecialistDescriptor
from app.langgraph_v2.agent_skills import (
    SkillInvocation,
    SkillPin,
    SpecialistSkillRegistry,
)

_SPECIALIST_FINDING_MAX_BYTES = 16 * 1024
_DATA_GAP_MAX_BYTES = 256
_IDENTIFIER_MAX_ASCII_CHARACTERS = 64


def _require_utf8_limit(value: str, *, limit: int, label: str) -> str:
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"{label} exceeds {limit} UTF-8 bytes")
    return value


def _require_ascii_identifier(value: str, *, label: str) -> str:
    if len(value) > _IDENTIFIER_MAX_ASCII_CHARACTERS or not value.isascii():
        raise ValueError(
            f"{label} must contain at most {_IDENTIFIER_MAX_ASCII_CHARACTERS} ASCII characters"
        )
    return value


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
    data_gaps: tuple[DataGap, ...] = Field(max_length=8, default=())


class GapProvenance(BaseModel):
    """Internal retained identity for one accepted unavailable Tool outcome."""

    model_config = ConfigDict(frozen=True)

    unavailability_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    source: str | None = None
    observed_at: datetime

    @field_validator("unavailability_id", "tool_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        return _require_ascii_identifier(value, label="Data Gap identifier")

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_utf8_limit(
            value, limit=_DATA_GAP_MAX_BYTES, label="Data Gap source"
        )

    @field_validator("observed_at")
    @classmethod
    def _validate_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("Data Gap observation time must be normalized UTC")
        return value.astimezone(UTC)


class DataGap(BaseModel):
    """Canonical accepted unavailable-data record derived only by code."""

    model_config = ConfigDict(frozen=True)

    requested_coverage: str = Field(min_length=1)
    reason: ToolUnavailableReason
    provenance: GapProvenance

    @field_validator("requested_coverage")
    @classmethod
    def _validate_requested_coverage(cls, value: str) -> str:
        return _require_utf8_limit(
            value, limit=_DATA_GAP_MAX_BYTES, label="Data Gap coverage"
        )

    def view(self) -> DataGapView:
        """Return the sole projection allowed outside accepted state."""
        return DataGapView(
            requested_coverage=self.requested_coverage,
            reason=self.reason,
            observed_at=self.provenance.observed_at,
        )


class SpecialistAttempt(BaseModel):
    """Actor-local terminal output, including app-only Tool metadata."""

    model_config = ConfigDict(frozen=True)

    finding: SpecialistFindingDraft
    evidence: tuple[EvidenceEnvelope, ...] = ()
    unavailability: tuple[ToolUnavailabilityRecord, ...] = ()
    skill_pins: tuple[SkillPin, ...] = ()


def _derive_data_gaps(
    records: tuple[ToolUnavailabilityRecord, ...],
    *,
    context: EvidenceInvocationContext,
    effective_tool_ids: frozenset[str],
) -> tuple[DataGap, ...]:
    """Promote only current trusted binding records into canonical Data Gaps."""
    records_by_id: dict[str, ToolUnavailabilityRecord] = {}
    records_by_tool_call_id: dict[str, ToolUnavailabilityRecord] = {}
    for record in records:
        if (
            record.tenant_id != context.tenant_id
            or record.request_id != context.request_id
            or record.task_id != context.task_id
            or record.tool_id not in effective_tool_ids
            or (
                record.source is not None
                and record.source not in context.allowed_sources
            )
        ):
            raise ValueError("Data Gap provenance is not eligible")
        if record.attempt != context.attempt:
            raise ValueError("Data Gap provenance is stale")
        if not record.tool_call_id:
            raise ValueError("Data Gap provenance is missing")
        existing = records_by_id.get(record.id)
        if existing is not None:
            if existing != record:
                raise ValueError("Data Gap provenance conflicts")
            continue
        existing_call = records_by_tool_call_id.get(record.tool_call_id)
        if existing_call is not None and existing_call != record:
            raise ValueError("Data Gap provenance conflicts")
        records_by_id[record.id] = record
        records_by_tool_call_id[record.tool_call_id] = record
    return tuple(
        DataGap(
            requested_coverage=record.requested_coverage,
            reason=record.reason,
            provenance=GapProvenance(
                unavailability_id=record.id,
                tool_id=record.tool_id,
                source=record.source,
                observed_at=record.observed_at,
            ),
        )
        for record in sorted(
            records_by_id.values(), key=lambda item: (item.tool_id, item.tool_call_id)
        )
    )


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
    skill_pins: tuple[SkillPin, ...] = ()


class TaskSkillPins(BaseModel):
    """Immutable Skill pins associated with one accepted Task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    pins: tuple[SkillPin, ...]


class AcceptedBatch(BaseModel):
    """One immutable batch promoted only by the barrier."""

    model_config = ConfigDict(frozen=True)

    id: str
    outcomes: tuple[TaskSucceeded, ...]
    skill_pins: tuple[TaskSkillPins, ...] = ()


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


SpecialistTool = Callable[..., object]
ToolTelemetry = Callable[[str, ToolTelemetryStatus], None]


class SpecialistActorFactory(Protocol):
    """Build one Specialist actor after its Tool set is frozen."""

    def __call__(
        self,
        tools: tuple[SpecialistTool, ...],
        tool_capture: SpecialistToolCapture,
        skill_invocation: SkillInvocation | None,
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
    expected_unavailability: tuple[ExpectedToolUnavailability, ...] = ()
    audit: ToolTelemetry | None = None


@dataclass(frozen=True)
class SpecialistRegistration:
    """Typed code registration for one Specialist actor."""

    id: str
    actor: SpecialistActor | None = None
    actor_factory: SpecialistActorFactory | None = None
    allowed_tool_ids: frozenset[str] = frozenset()
    allowed_skill_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SpecialistRegistry:
    """Resolve registered and Tenant-eligible Specialists at one deep seam."""

    registrations: Sequence[SpecialistRegistration]
    tenant_eligible_ids: frozenset[str]
    tool_registrations: Sequence[EvidenceToolRegistration] = ()
    tenant_eligible_tool_ids: frozenset[str] = frozenset()
    skill_registry: SpecialistSkillRegistry | None = None

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
        context: EvidenceInvocationContext,
        scope_skill_names: frozenset[str] = frozenset(),
        tool_telemetry: ToolTelemetry | None = None,
    ) -> SpecialistActor:
        """Create one actor with a frozen Scope-narrowed Tool surface."""
        effective_ids = self.effective_tool_ids(
            registration, scope_tool_ids=context.allowed_tool_ids
        )
        skill_invocation = (
            self.skill_registry.begin_invocation(
                specialist_skill_names=registration.allowed_skill_names,
                scope_skill_names=scope_skill_names,
                effective_tool_ids=effective_ids,
            )
            if self.skill_registry is not None
            else None
        )
        has_skill_activation = bool(
            skill_invocation is not None and skill_invocation.summaries
        )
        if (
            not effective_ids
            and not has_skill_activation
            and registration.actor_factory is None
        ):
            if registration.actor is not None:
                return registration.actor
            raise ValueError("Specialist actor factory is not configured")
        actor_factory = registration.actor_factory
        if actor_factory is None:
            raise AssertionError("Specialist actor factory is required")
        registered = {tool.id: tool for tool in self.tool_registrations}
        tool_capture = SpecialistToolCapture()
        tools: list[SpecialistTool] = []
        for tool_id in sorted(effective_ids):
            tool = registered[tool_id]

            def report_tool_status(
                status: ToolTelemetryStatus,
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
                context=context.model_copy(
                    update={
                        "allowed_tool_ids": frozenset({tool_id}),
                        "allowed_sources": tool.allowed_sources
                        & context.allowed_sources,
                        "allowed_queries": tool.allowed_queries
                        & context.allowed_queries,
                    }
                ),
                capture=tool_capture,
                telemetry=report_tool_status,
                tool_id=tool_id,
                expected_unavailability=tool.expected_unavailability,
            )
            binding.__name__ = tool_id
            tools.append(binding)
        if has_skill_activation:
            assert skill_invocation is not None
            tools.append(skill_invocation.activation_tool())
        return actor_factory(
            tuple(tools),
            tool_capture,
            skill_invocation,
        )


@dataclass(frozen=True)
class SpecialistTaskInput:
    """The business input needed by one bounded Specialist invocation."""

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
    context: EvidenceInvocationContext,
    scope_skill_names: frozenset[str] = frozenset(),
    tool_telemetry: ToolTelemetry | None = None,
) -> BatchContribution:
    """Run one registered bounded Specialist and stage its terminal outcome."""
    registration = registry.resolve(
        task.specialist_id, scope_descriptors=scope_descriptors
    )
    actor = registry.bind_actor(
        registration,
        context=context,
        scope_skill_names=scope_skill_names,
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
        catalog.accept_referenced(
            attempt.evidence,
            finding_evidence_ids=draft.evidence_ids,
            context=context,
        )
    data_gaps = _derive_data_gaps(
        attempt.unavailability,
        context=context,
        effective_tool_ids=registry.effective_tool_ids(
            registration,
            scope_tool_ids=context.allowed_tool_ids,
        ),
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
                data_gaps=data_gaps,
            ),
        ),
        skill_pins=attempt.skill_pins,
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
    return AcceptedBatch(
        id=batch.id,
        outcomes=tuple(item.outcome for item in ordered),
        skill_pins=tuple(
            TaskSkillPins(task_id=item.task_id, pins=item.skill_pins)
            for item in ordered
            if item.skill_pins
        ),
    )
