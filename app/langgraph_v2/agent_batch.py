"""First-batch Task contracts, validation, execution, and acceptance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai.exceptions import IncompleteToolCall, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.usage import RunUsage, UsageLimits

from app.langgraph_v2.agent_evidence import (
    DataGapView,
    EvidenceCacheCapacityExceeded,
    EvidenceEnvelope,
    EvidenceInvocationContext,
    EvidenceProvider,
    EvidenceReferenceInvalid,
    ExpectedToolUnavailability,
    RequestEvidenceCatalog,
    SpecialistToolCapture,
    ToolTelemetryStatus,
    ToolUnavailabilityRecord,
    ToolUnavailableReason,
    bind_evidence_tool,
    require_data_gap_identifier,
    require_data_gap_text,
)
from app.langgraph_v2.agent_scope import SpecialistDescriptor
from app.langgraph_v2.agent_skills import (
    SkillInvocation,
    SkillPin,
    SpecialistSkillRegistry,
)
from app.langgraph_v2.specialist_retry import (
    SPECIALIST_MAX_ATTEMPTS,
    RetryDisposition,
    SpecialistExecutionDiagnostics,
    SpecialistInvocationFailure,
    classify_specialist_failure,
    specialist_usage_limits,
)

_SPECIALIST_OUTPUT_MAX_BYTES = 16 * 1024
MAX_DISPATCH_BATCH_TASKS = 8


class StructuredOutputInvalid(ValueError):
    """Reject an over-limit Specialist contract before contribution acceptance."""


def _stable_id(prefix: str, value: Mapping[str, object]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}_{hashlib.sha256(canonical).hexdigest()[:32]}"


def _canonical_json_size(value: BaseModel) -> int:
    return len(
        json.dumps(
            value.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


class TaskProposal(BaseModel):
    """Model-authored bounded request for one registered Specialist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    specialist_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    context_task_ids: tuple[str, ...] = ()


class DispatchBatch(BaseModel):
    """Coordinator decision that requests one independent first-round batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["dispatch"]
    tasks: tuple[TaskProposal, ...] = Field(
        min_length=1, max_length=MAX_DISPATCH_BATCH_TASKS
    )


class SpecialistFindingDraft(BaseModel):
    """Minimal model-authored Specialist terminal output for this ticket."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(max_length=16, default=())

    def canonical_json_size(self) -> int:
        """Return canonical UTF-8 size used by the acceptance boundary."""
        return _canonical_json_size(self)

    def require_canonical_size(self) -> None:
        """Reject an over-limit model output before it becomes a contribution."""
        if self.canonical_json_size() > _SPECIALIST_OUTPUT_MAX_BYTES:
            raise StructuredOutputInvalid("Specialist finding exceeds 16 KiB")


class SpecialistResult(BaseModel):
    """Accepted minimal Result without Tool metadata or diagnostics."""

    model_config = ConfigDict(frozen=True)

    summary: str
    evidence_ids: tuple[str, ...] = Field(max_length=16, default=())
    data_gaps: tuple[DataGap, ...] = Field(max_length=8, default=())

    def canonical_json_size(self) -> int:
        """Return the complete accepted Result's canonical UTF-8 size."""
        return _canonical_json_size(self)

    def require_canonical_size(self) -> None:
        """Reject a Result that exceeds its complete accepted-state bound."""
        if self.canonical_json_size() > _SPECIALIST_OUTPUT_MAX_BYTES:
            raise StructuredOutputInvalid("Specialist result exceeds 16 KiB")


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
        return require_data_gap_identifier(value, label="Data Gap identifier")

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_data_gap_text(value, label="Data Gap source")

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
        return require_data_gap_text(value, label="Data Gap coverage")

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
    messages: tuple[ModelMessage, ...] = ()


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


class TaskFailed(BaseModel):
    """Platform-owned expected terminal Task inability without diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["failed"] = "failed"
    task_id: str


TaskOutcome = TaskSucceeded | TaskFailed


class SpecialistUsage(BaseModel):
    """Graph-owned cumulative Specialist accounting for one Task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_requests: int = Field(default=0, ge=0)
    completed_tool_calls: int = Field(default=0, ge=0)
    tool_attempts: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    cost_is_complete: bool = True

    @classmethod
    def from_run_usage(
        cls,
        usage: RunUsage,
        *,
        tool_attempts: int,
        cost_usd: float,
        cost_is_complete: bool,
    ) -> SpecialistUsage:
        """Project usage without inventing cost that the provider did not return."""
        return cls(
            model_requests=usage.requests,
            completed_tool_calls=usage.tool_calls,
            tool_attempts=tool_attempts,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cost_usd=cost_usd,
            cost_is_complete=cost_is_complete,
        )

    def add(self, other: SpecialistUsage) -> SpecialistUsage:
        """Return a deterministic aggregate without placing usage in an Outcome."""
        return SpecialistUsage(
            model_requests=self.model_requests + other.model_requests,
            completed_tool_calls=self.completed_tool_calls + other.completed_tool_calls,
            tool_attempts=self.tool_attempts + other.tool_attempts,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            cost_is_complete=self.cost_is_complete and other.cost_is_complete,
        )


class BatchContribution(BaseModel):
    """One immutable terminal result staged by a Specialist branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str
    task_id: str
    attempt: int = Field(ge=1, le=SPECIALIST_MAX_ATTEMPTS)
    outcome: TaskOutcome
    usage: SpecialistUsage = Field(default_factory=SpecialistUsage)
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
    outcomes: tuple[TaskOutcome, ...]
    usage: SpecialistUsage = Field(default_factory=SpecialistUsage)
    skill_pins: tuple[TaskSkillPins, ...] = ()


@dataclass(frozen=True)
class AcceptedTask:
    """Graph-owned identity plus trusted Specialist selection."""

    id: str
    objective: str
    specialist_id: str


@dataclass(frozen=True)
class ActiveBatch:
    """Immutable first-round manifest awaiting whole-batch acceptance."""

    id: str
    tasks: tuple[AcceptedTask, ...]

    def __post_init__(self) -> None:
        """Reject malformed manifests before they can be dispatched or promoted."""
        if not 1 <= len(self.tasks) <= MAX_DISPATCH_BATCH_TASKS:
            raise ValueError("Active Batch Task count is invalid")
        if len(set(self.task_ids)) != len(self.tasks):
            raise ValueError("Active Batch Task identities conflict")

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Expose stable manifest order to the barrier."""
        return tuple(task.id for task in self.tasks)


class SpecialistActor(Protocol):
    """Execute one bounded Specialist Task without exposing Tools to graph code."""

    async def run(
        self,
        input: SpecialistTaskInput,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> SpecialistAttempt:
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
    validation_feedback: str | None = None


def accept_initial_dispatch(
    decision: DispatchBatch,
    *,
    request_id: str,
    registry: SpecialistRegistry,
    scope_descriptors: Sequence[SpecialistDescriptor],
) -> ActiveBatch:
    """Validate a first Dispatch before assigning graph-owned Task identities."""
    proposals = tuple(decision.tasks)
    if not 1 <= len(proposals) <= MAX_DISPATCH_BATCH_TASKS:
        raise ValueError(
            f"Dispatch Batch exceeds {MAX_DISPATCH_BATCH_TASKS} Tasks"
        )
    for proposal in proposals:
        if not proposal.objective.strip():
            raise ValueError("Specialist objective must not be blank")
        if proposal.context_task_ids:
            raise ValueError("initial Dispatch cannot select prior Task context")
        registry.resolve(proposal.specialist_id, scope_descriptors=scope_descriptors)
    batch_id = _stable_id(
        "batch", {"request_id": request_id, "round": 1}
    )
    active_batch = ActiveBatch(
        id=batch_id,
        tasks=tuple(
            AcceptedTask(
                id=_stable_id(
                    "task",
                    {
                        "request_id": request_id,
                        "round": 1,
                        "dispatch_order": dispatch_order,
                    },
                ),
                objective=proposal.objective.strip(),
                specialist_id=proposal.specialist_id,
            )
            for dispatch_order, proposal in enumerate(proposals)
        ),
    )
    validate_active_batch_manifest(
        active_batch,
        request_id=request_id,
        registry=registry,
        scope_descriptors=scope_descriptors,
    )
    return active_batch


def validate_active_batch_manifest(
    batch: ActiveBatch,
    *,
    request_id: str,
    registry: SpecialistRegistry,
    scope_descriptors: Sequence[SpecialistDescriptor],
) -> None:
    """Revalidate a checkpointed batch before it can fan out Specialist work."""
    expected_batch_id = _stable_id("batch", {"request_id": request_id, "round": 1})
    if batch.id != expected_batch_id:
        raise ValueError("Active Batch identity is invalid")
    for dispatch_order, task in enumerate(batch.tasks):
        expected_task_id = _stable_id(
            "task",
            {
                "request_id": request_id,
                "round": 1,
                "dispatch_order": dispatch_order,
            },
        )
        if task.id != expected_task_id:
            raise ValueError("Active Batch Task identity is invalid")
        if not task.objective.strip() or task.objective != task.objective.strip():
            raise ValueError("Active Batch Task objective is invalid")
        registry.resolve(task.specialist_id, scope_descriptors=scope_descriptors)


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
    diagnostics: SpecialistExecutionDiagnostics | None = None,
) -> BatchContribution:
    """Run up to three fresh bounded attempts and stage one terminal Outcome."""
    registration = registry.resolve(
        task.specialist_id, scope_descriptors=scope_descriptors
    )
    cumulative_usage = RunUsage()
    cumulative_tool_attempts = 0
    cumulative_cost_usd = 0.0
    cumulative_cost_is_complete = True
    usage_limits = specialist_usage_limits()
    assert usage_limits.request_limit is not None
    assert usage_limits.tool_calls_limit is not None

    def failed_contribution(attempt_number: int) -> BatchContribution:
        return BatchContribution(
            batch_id=batch_id,
            task_id=task.id,
            attempt=attempt_number,
            outcome=TaskFailed(task_id=task.id),
            usage=SpecialistUsage.from_run_usage(
                cumulative_usage,
                tool_attempts=cumulative_tool_attempts,
                cost_usd=cumulative_cost_usd,
                cost_is_complete=cumulative_cost_is_complete,
            ),
        )

    def record_validation_failure(
        attempt_number: int,
        attempt: SpecialistAttempt,
    ) -> None:
        if diagnostics is not None:
            diagnostics.record_failed_attempt(
                attempt=attempt_number,
                messages=tuple(attempt.messages),
            )

    validation_feedback: str | None = None
    for attempt_number in range(1, SPECIALIST_MAX_ATTEMPTS + 1):
        attempt_context = context.model_copy(update={"attempt": attempt_number})
        actor = registry.bind_actor(
            registration,
            context=attempt_context,
            scope_skill_names=scope_skill_names,
            tool_telemetry=tool_telemetry,
        )
        try:
            attempt = await actor.run(
                SpecialistTaskInput(
                    task_id=task.id,
                    objective=task.objective,
                    validation_feedback=validation_feedback,
                ),
                usage=cumulative_usage,
                usage_limits=usage_limits,
            )
        except SpecialistInvocationFailure as failure:
            cumulative_tool_attempts += _tool_attempt_count(failure.messages)
            attempt_cost_usd, attempt_cost_is_complete = _message_cost_usd(
                failure.messages
            )
            cumulative_cost_usd += attempt_cost_usd
            cumulative_cost_is_complete = (
                cumulative_cost_is_complete
                and attempt_cost_is_complete
                and failure.facts.unreturned_model_requests == 0
            )
            if diagnostics is not None:
                diagnostics.record_failed_attempt(
                    attempt=attempt_number,
                    messages=failure.messages,
                )
            disposition = classify_specialist_failure(
                failure.error, facts=failure.facts
            )
            if disposition is RetryDisposition.TASK_FAILED:
                return failed_contribution(attempt_number)
            if disposition is RetryDisposition.RETRY:
                if attempt_number == SPECIALIST_MAX_ATTEMPTS:
                    return failed_contribution(attempt_number)
                if isinstance(
                    failure.error, (IncompleteToolCall, UnexpectedModelBehavior)
                ):
                    validation_feedback = (
                        "Return one valid final_result structured Specialist finding."
                    )
                continue
            raise failure.error

        cumulative_tool_attempts += _tool_attempt_count(attempt.messages)
        attempt_cost_usd, attempt_cost_is_complete = _message_cost_usd(
            attempt.messages
        )
        cumulative_cost_usd += attempt_cost_usd
        cumulative_cost_is_complete = (
            cumulative_cost_is_complete and attempt_cost_is_complete
        )

        if (
            cumulative_usage.requests > usage_limits.request_limit
            or cumulative_usage.tool_calls > usage_limits.tool_calls_limit
        ):
            return failed_contribution(attempt_number)

        try:
            draft = attempt.finding
            draft.require_canonical_size()
            data_gaps = _derive_data_gaps(
                attempt.unavailability,
                context=attempt_context,
                effective_tool_ids=registry.effective_tool_ids(
                    registration,
                    scope_tool_ids=context.allowed_tool_ids,
                ),
            )
            result = SpecialistResult(
                summary=draft.summary,
                evidence_ids=draft.evidence_ids,
                data_gaps=data_gaps,
            )
            result.require_canonical_size()
        except StructuredOutputInvalid:
            record_validation_failure(attempt_number, attempt)
            if attempt_number == SPECIALIST_MAX_ATTEMPTS:
                return failed_contribution(attempt_number)
            validation_feedback = (
                "Return one valid structured Specialist finding within all stated "
                "output limits."
            )
            continue

        if catalog is None:
            if attempt.evidence or draft.evidence_ids:
                raise ValueError("Evidence cache is not configured")
        else:
            try:
                catalog.accept_referenced(
                    attempt.evidence,
                    finding_evidence_ids=draft.evidence_ids,
                    context=attempt_context,
                )
            except EvidenceCacheCapacityExceeded:
                return failed_contribution(attempt_number)
            except EvidenceReferenceInvalid:
                record_validation_failure(attempt_number, attempt)
                if attempt_number == SPECIALIST_MAX_ATTEMPTS:
                    return failed_contribution(attempt_number)
                validation_feedback = (
                    "Reference only Evidence IDs returned by this Specialist run."
                )
                continue
        return BatchContribution(
            batch_id=batch_id,
            task_id=task.id,
            attempt=attempt_number,
            outcome=TaskSucceeded(task_id=task.id, result=result),
            usage=SpecialistUsage.from_run_usage(
                cumulative_usage,
                tool_attempts=cumulative_tool_attempts,
                cost_usd=cumulative_cost_usd,
                cost_is_complete=cumulative_cost_is_complete,
            ),
            skill_pins=attempt.skill_pins,
        )
    raise AssertionError("Specialist attempts did not reach a terminal outcome")


def _tool_attempt_count(messages: Sequence[object]) -> int:
    """Count model-issued business Tool calls, excluding structured output."""
    return sum(
        isinstance(part, ToolCallPart) and part.tool_name != "final_result"
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
    )


def _message_cost_usd(messages: Sequence[object]) -> tuple[float, bool]:
    """Sum provider prices and mark accounting partial when pricing is absent."""
    cost_usd = 0.0
    is_complete = True
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        if message.model_name is None:
            is_complete = False
            continue
        try:
            cost_usd += float(message.cost().total_price)
        except LookupError:
            is_complete = False
    return cost_usd, is_complete


def promote_batch(
    batch: ActiveBatch,
    contributions: Mapping[str, BatchContribution],
) -> AcceptedBatch:
    """Validate exact manifest membership before atomically promoting a batch."""
    if set(contributions) != set(batch.task_ids):
        raise ValueError("Batch contribution manifest is invalid")
    if any(task_id != contribution.task_id for task_id, contribution in contributions.items()):
        raise ValueError("Batch contribution manifest is invalid")
    ordered = tuple(contributions[task_id] for task_id in batch.task_ids)
    if any(
        contribution.batch_id != batch.id
        or contribution.task_id != contribution.outcome.task_id
        or contribution.task_id not in batch.task_ids
        or not 1 <= contribution.attempt <= SPECIALIST_MAX_ATTEMPTS
        for contribution in ordered
    ):
        raise ValueError("Batch contribution manifest is invalid")
    usage = SpecialistUsage()
    for contribution in ordered:
        usage = usage.add(contribution.usage)
    return AcceptedBatch(
        id=batch.id,
        outcomes=tuple(item.outcome for item in ordered),
        usage=usage,
        skill_pins=tuple(
            TaskSkillPins(task_id=item.task_id, pins=item.skill_pins)
            for item in ordered
            if item.skill_pins
        ),
    )
