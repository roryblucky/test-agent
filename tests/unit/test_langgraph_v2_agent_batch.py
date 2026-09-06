"""Public first-batch validation and acceptance coverage."""

from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import cast

import pytest
from pydantic import ValidationError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import IncompleteToolCall, ModelHTTPError
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits

from app.langgraph_v2.agent_batch import (
    AcceptedTask,
    ActiveBatch,
    BatchContribution,
    DataGap,
    DispatchBatch,
    EvidenceToolRegistration,
    GapProvenance,
    SpecialistAttempt,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistResult,
    SpecialistTaskInput,
    SpecialistUsage,
    StructuredOutputInvalid,
    TaskFailed,
    TaskProposal,
    TaskSkillPins,
    TaskSucceeded,
    accept_initial_dispatch,
    execute_specialist,
    promote_batch,
    validate_active_batch_manifest,
)
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    EvidenceInvocationContext,
    RequestEvidenceCatalog,
    SpecialistToolCapture,
    ToolUnavailabilityRecord,
    ToolUnavailableReason,
)
from app.langgraph_v2.agent_scope import SpecialistDescriptor
from app.langgraph_v2.agent_skills import (
    SkillInvocation,
    SkillPin,
    SkillRegistration,
    SpecialistSkillRegistry,
)
from app.langgraph_v2.specialist_retry import (
    SpecialistExecutionDiagnostics,
    SpecialistFailureFacts,
    SpecialistInvocationFailure,
    SpecialistModelBoundary,
    specialist_usage_limits,
)


class _Specialist:
    def __init__(
        self,
        finding: SpecialistFindingDraft | None = None,
        skill_pins: tuple[SkillPin, ...] = (),
        unavailability: tuple[ToolUnavailabilityRecord, ...] = (),
        evidence: tuple[EvidenceEnvelope, ...] = (),
    ) -> None:
        self.finding = finding or SpecialistFindingDraft(summary="No-tool finding")
        self.skill_pins = skill_pins
        self.unavailability = unavailability
        self.evidence = evidence

    async def run(
        self,
        input: object,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> SpecialistAttempt:
        del input
        del usage, usage_limits
        return SpecialistAttempt(
            finding=self.finding,
            skill_pins=self.skill_pins,
            unavailability=self.unavailability,
            evidence=self.evidence,
        )


def _context(*, task_id: str = "task-1") -> EvidenceInvocationContext:
    return EvidenceInvocationContext(
        tenant_id="tenant-a",
        request_id="request-1",
        task_id=task_id,
        allowed_tool_ids=frozenset({"filing-tool", "unregistered"}),
        allowed_sources=frozenset({"filing"}),
        allowed_queries=frozenset({"Apple"}),
    )


def _tool_context(*, tool_name: str) -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        tool_call_id="call-1",
        tool_name=tool_name,
    )


def _registry(
    *,
    tenant_eligible_ids: frozenset[str] = frozenset({"market-data"}),
    finding: SpecialistFindingDraft | None = None,
) -> SpecialistRegistry:
    return SpecialistRegistry(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                actor=_Specialist(finding),
            ),
        ),
        tenant_eligible_ids=tenant_eligible_ids,
    )


def _unavailability_record(
    *, task_id: str, **overrides: object
) -> ToolUnavailabilityRecord:
    values: dict[str, object] = {
        "id": "unavailable-1",
        "tenant_id": "tenant-a",
        "request_id": "request-1",
        "task_id": task_id,
        "attempt": 1,
        "tool_call_id": "call-1",
        "tool_id": "filing-tool",
        "source": "filing",
        "observed_at": datetime(2026, 9, 6, 12, tzinfo=UTC),
        "reason": ToolUnavailableReason.SOURCE_UNREACHABLE,
        "requested_coverage": "Apple revenue",
    }
    return ToolUnavailabilityRecord.model_validate({**values, **overrides})


def _gap_registry(
    records: tuple[ToolUnavailabilityRecord, ...],
    *,
    finding: SpecialistFindingDraft | None = None,
) -> SpecialistRegistry:
    actor_attempt = 0

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise AssertionError("Direct Specialist actor must not call the binding")

    def factory(
        tools: tuple[object, ...],
        tool_capture: object,
        skill_invocation: object,
    ) -> _Specialist:
        nonlocal actor_attempt
        del tools, tool_capture, skill_invocation
        actor_attempt += 1
        attempt_records = (
            tuple(
                record.model_copy(update={"attempt": actor_attempt})
                for record in records
            )
            if all(record.attempt == 1 for record in records)
            else records
        )
        return _Specialist(finding=finding, unavailability=attempt_records)

    return SpecialistRegistry(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                actor_factory=factory,
                allowed_tool_ids=frozenset({"filing-tool"}),
            ),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
        tool_registrations=(
            EvidenceToolRegistration(
                id="filing-tool",
                provider=provider,
                allowed_sources=frozenset({"filing"}),
            ),
        ),
        tenant_eligible_tool_ids=frozenset({"filing-tool"}),
    )


def _dispatch(*, context_task_ids: tuple[str, ...] = ()) -> DispatchBatch:
    return DispatchBatch(
        kind="dispatch",
        tasks=(
            TaskProposal(
                specialist_id="market-data",
                objective="Assess the current market outlook.",
                context_task_ids=context_task_ids,
            ),
        ),
    )


def _retry_failure(*, messages: tuple[object, ...] = ()) -> SpecialistInvocationFailure:
    return SpecialistInvocationFailure(
        ModelHTTPError(429, "specialist"),
        facts=SpecialistFailureFacts(
            boundary=SpecialistModelBoundary.AZURE_OPENAI,
            at_model_request_boundary=True,
            terminal_output_tool_rejected=False,
            count_limit_exhausted=False,
            unreturned_model_requests=0,
            usage_limits=specialist_usage_limits(),
        ),
        messages=messages,
    )


def test_initial_dispatch_intersects_registered_tenant_and_scope_eligibility() -> None:
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )

    assert batch.task_ids == ("task_90fff3e68e9a59d229d7982b65c5fe8b",)
    assert batch.tasks[0].objective == "Assess the current market outlook."


def test_initial_dispatch_accepts_eight_independent_tasks_in_manifest_order() -> None:
    decision = DispatchBatch(
        kind="dispatch",
        tasks=tuple(
            TaskProposal(
                specialist_id="market-data",
                objective=f"Assess market dimension {index}.",
            )
            for index in range(8)
        ),
    )

    batch = accept_initial_dispatch(
        decision,
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )

    assert len(batch.tasks) == 8
    assert [task.objective for task in batch.tasks] == [
        f"Assess market dimension {index}." for index in range(8)
    ]
    assert len(set(batch.task_ids)) == 8


def test_initial_dispatch_rejects_ninth_task_even_if_model_validation_was_bypassed() -> (
    None
):
    decision = DispatchBatch.model_construct(
        kind="dispatch",
        tasks=tuple(
            TaskProposal(
                specialist_id="market-data",
                objective=f"Assess market dimension {index}.",
            )
            for index in range(9)
        ),
    )

    with pytest.raises(ValueError, match="Dispatch Batch exceeds 8 Tasks"):
        accept_initial_dispatch(
            decision,
            request_id="request-1",
            registry=_registry(),
            scope_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
        )

    with pytest.raises(ValidationError):
        DispatchBatch(
            kind="dispatch",
            tasks=decision.tasks,
        )


def test_active_batch_rejects_duplicate_task_identities() -> None:
    task = AcceptedTask(
        id="task-1",
        objective="Assess market outlook.",
        specialist_id="market-data",
    )

    with pytest.raises(ValueError, match="Active Batch Task identities conflict"):
        ActiveBatch(id="batch-1", tasks=(task, task))


def test_active_batch_rejects_recovered_manifest_larger_than_eight_tasks() -> None:
    tasks = tuple(
        AcceptedTask(
            id=f"task-{index}",
            objective=f"Assess market dimension {index}.",
            specialist_id="market-data",
        )
        for index in range(9)
    )

    with pytest.raises(ValueError, match="Active Batch Task count is invalid"):
        ActiveBatch(id="batch-1", tasks=tasks)


def test_recovered_active_batch_is_validated_before_specialist_fanout() -> None:
    valid = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )
    task = valid.tasks[0]
    invalid_batches = (
        (
            ActiveBatch(
                id=valid.id,
                tasks=(
                    AcceptedTask(
                        id="task-invalid",
                        objective=task.objective,
                        specialist_id=task.specialist_id,
                    ),
                ),
            ),
            "Active Batch Task identity is invalid",
        ),
        (
            ActiveBatch(
                id=valid.id,
                tasks=(
                    AcceptedTask(
                        id=task.id,
                        objective=" ",
                        specialist_id=task.specialist_id,
                    ),
                ),
            ),
            "Active Batch Task objective is invalid",
        ),
        (
            ActiveBatch(
                id=valid.id,
                tasks=(
                    AcceptedTask(
                        id=task.id,
                        objective=task.objective,
                        specialist_id="other",
                    ),
                ),
            ),
            "Specialist is not eligible",
        ),
    )

    for batch, error in invalid_batches:
        with pytest.raises(ValueError, match=error):
            validate_active_batch_manifest(
                batch,
                request_id="request-1",
                registry=_registry(),
                scope_descriptors=(
                    SpecialistDescriptor(id="market-data", description="Market data"),
                ),
            )


def test_initial_dispatch_rejects_specialist_outside_tenant_eligibility() -> None:
    with pytest.raises(ValueError, match="Specialist is not eligible"):
        accept_initial_dispatch(
            _dispatch(),
            request_id="request-1",
            registry=_registry(tenant_eligible_ids=frozenset()),
            scope_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
        )


def test_initial_dispatch_rejects_specialist_outside_scope_eligibility() -> None:
    with pytest.raises(ValueError, match="Specialist is not eligible"):
        accept_initial_dispatch(
            _dispatch(),
            request_id="request-1",
            registry=_registry(),
            scope_descriptors=(SpecialistDescriptor(id="other", description="Other"),),
        )


def test_initial_dispatch_rejects_context_and_model_authority_fields() -> None:
    with pytest.raises(
        ValueError, match="initial Dispatch cannot select prior Task context"
    ):
        accept_initial_dispatch(
            _dispatch(context_task_ids=("earlier",)),
            request_id="request-1",
            registry=_registry(),
            scope_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
        )

    for field in ("task_id", "tool", "skill", "limit"):
        with pytest.raises(ValueError):
            TaskProposal.model_validate(
                {
                    "specialist_id": "market-data",
                    "objective": "Assess market outlook.",
                    field: "forged",
                }
            )


def test_task_identity_is_stable_and_changes_with_request_identity() -> None:
    registry = _registry()
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )

    first = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=registry,
        scope_descriptors=scope_descriptors,
    )
    repeated = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=registry,
        scope_descriptors=scope_descriptors,
    )
    changed = accept_initial_dispatch(
        _dispatch(),
        request_id="request-2",
        registry=registry,
        scope_descriptors=scope_descriptors,
    )

    assert first.task_ids == repeated.task_ids
    assert first.task_ids != changed.task_ids


def test_barrier_promotes_exact_immutable_contribution_and_clears_staging() -> None:
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )
    contribution = BatchContribution(
        batch_id=batch.id,
        task_id=batch.task_ids[0],
        attempt=1,
        outcome=TaskSucceeded(
            task_id=batch.task_ids[0],
            result=SpecialistResult(summary="No-tool finding"),
        ),
    )

    accepted = promote_batch(batch, {contribution.task_id: contribution})

    assert accepted.id == batch.id
    assert accepted.outcomes[0].task_id == contribution.task_id
    with pytest.raises(ValidationError):
        contribution.task_id = "forged"
    with pytest.raises(ValueError, match="Batch contribution manifest is invalid"):
        promote_batch(batch, {})
    with pytest.raises(ValueError, match="Batch contribution manifest is invalid"):
        promote_batch(
            batch,
            {
                contribution.task_id: contribution.model_copy(
                    update={"batch_id": "forged"}
                )
            },
        )


def test_barrier_rejects_contributions_staged_under_another_task_identity() -> None:
    batch = accept_initial_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Assess market valuation.",
                ),
                TaskProposal(
                    specialist_id="market-data",
                    objective="Assess market momentum.",
                ),
            ),
        ),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )
    contributions = {
        task.id: BatchContribution(
            batch_id=batch.id,
            task_id=task.id,
            attempt=1,
            outcome=TaskSucceeded(
                task_id=task.id,
                result=SpecialistResult(summary=task.objective),
            ),
        )
        for task in batch.tasks
    }

    with pytest.raises(ValueError, match="Batch contribution manifest is invalid"):
        promote_batch(
            batch,
            {
                batch.task_ids[0]: contributions[batch.task_ids[1]],
                batch.task_ids[1]: contributions[batch.task_ids[0]],
            },
        )


@pytest.mark.parametrize("size", [16 * 1024, 16 * 1024 + 1])
def test_specialist_finding_canonical_size_has_exact_boundary(size: int) -> None:
    summary = "x" * (size - len('{"evidence_ids":[],"summary":""}'))
    finding = SpecialistFindingDraft(summary=summary)

    if size == 16 * 1024:
        assert finding.canonical_json_size() == size
    else:
        with pytest.raises(
            StructuredOutputInvalid, match="Specialist finding exceeds 16 KiB"
        ):
            finding.require_canonical_size()


def test_specialist_finding_rejects_more_than_16_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        SpecialistFindingDraft(
            summary="Too many references",
            evidence_ids=tuple(f"evidence-{index}" for index in range(17)),
        )


def test_specialist_finding_accepts_exactly_16_evidence_ids() -> None:
    finding = SpecialistFindingDraft(
        summary="Maximum references",
        evidence_ids=tuple(f"evidence-{index}" for index in range(16)),
    )

    assert len(finding.evidence_ids) == 16


def test_specialist_result_canonical_size_includes_data_gaps() -> None:
    gap = DataGap(
        requested_coverage="Apple revenue",
        reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
        provenance=GapProvenance(
            unavailability_id="unavailable-1",
            tool_id="filing-tool",
            source="filing",
            observed_at=datetime(2026, 9, 6, 12, tzinfo=UTC),
        ),
    )
    template = SpecialistResult(summary="", data_gaps=(gap,))
    exact = SpecialistResult(
        summary="x" * (16 * 1024 - template.canonical_json_size()),
        data_gaps=(gap,),
    )

    assert exact.canonical_json_size() == 16 * 1024
    exact.require_canonical_size()
    with pytest.raises(
        StructuredOutputInvalid, match="Specialist result exceeds 16 KiB"
    ):
        SpecialistResult(
            summary=f"{exact.summary}x", data_gaps=(gap,)
        ).require_canonical_size()


@pytest.mark.asyncio
async def test_execute_specialist_enforces_the_16_kib_boundary_before_contribution() -> (
    None
):
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=scope_descriptors,
    )
    template = SpecialistResult(summary="")
    exact = SpecialistFindingDraft(
        summary="x" * (16 * 1024 - template.canonical_json_size())
    )
    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=_registry(finding=exact),
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )

    assert isinstance(contribution.outcome, TaskSucceeded)
    assert contribution.outcome.result.summary == exact.summary

    too_large = SpecialistFindingDraft(summary=f"{exact.summary}x")
    failed = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=_registry(finding=too_large),
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )

    assert failed.attempt == 3
    assert failed.outcome == TaskFailed(task_id=batch.tasks[0].id)


@pytest.mark.asyncio
async def test_execute_specialist_retries_fresh_actors_and_keeps_only_last_attempt() -> (
    None
):
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    build_count = 0

    class _RetryingActor:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt

        async def run(
            self,
            input: SpecialistTaskInput,
            *,
            usage: RunUsage | None = None,
            usage_limits: UsageLimits | None = None,
        ) -> SpecialistAttempt:
            del input
            assert usage is not None
            assert usage_limits == specialist_usage_limits()
            usage.incr(RunUsage(requests=4, tool_calls=2))
            if self.attempt == 1:
                raise _retry_failure(messages=("abandoned-message",))
            return SpecialistAttempt(finding=SpecialistFindingDraft(summary="accepted"))

    def factory(
        tools: tuple[object, ...],
        tool_capture: object,
        skill_invocation: object,
    ) -> _RetryingActor:
        nonlocal build_count
        del tools, tool_capture, skill_invocation
        build_count += 1
        return _RetryingActor(build_count)

    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(id="market-data", actor_factory=factory),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=registry,
        scope_descriptors=scope_descriptors,
    )
    diagnostics = SpecialistExecutionDiagnostics()

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=registry,
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
        diagnostics=diagnostics,
    )

    assert build_count == 2
    assert contribution.attempt == 2
    assert contribution.outcome == TaskSucceeded(
        task_id=batch.tasks[0].id,
        result=SpecialistResult(summary="accepted"),
    )
    assert contribution.usage.model_requests == 8
    assert contribution.usage.completed_tool_calls == 4
    assert diagnostics.failed_attempts[0].messages == ("abandoned-message",)


@pytest.mark.asyncio
async def test_execute_specialist_returns_task_failed_after_third_retry() -> None:
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    build_count = 0

    class _AlwaysRetryingActor:
        async def run(
            self,
            input: SpecialistTaskInput,
            *,
            usage: RunUsage | None = None,
            usage_limits: UsageLimits | None = None,
        ) -> SpecialistAttempt:
            del input, usage_limits
            assert usage is not None
            usage.incr(RunUsage(requests=1))
            raise _retry_failure()

    def factory(
        tools: tuple[object, ...],
        tool_capture: object,
        skill_invocation: object,
    ) -> _AlwaysRetryingActor:
        nonlocal build_count
        del tools, tool_capture, skill_invocation
        build_count += 1
        return _AlwaysRetryingActor()

    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(id="market-data", actor_factory=factory),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=registry,
        scope_descriptors=scope_descriptors,
    )

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=registry,
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )

    assert build_count == 3
    assert contribution.attempt == 3
    assert contribution.outcome == TaskFailed(task_id=batch.tasks[0].id)
    assert contribution.usage.model_requests == 3


@pytest.mark.asyncio
async def test_execute_specialist_feeds_validation_failure_to_a_fresh_attempt() -> None:
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    inputs: list[SpecialistTaskInput] = []
    failed_message = ModelRequest(parts=[])

    class _ValidationRetryingActor:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt

        async def run(
            self,
            input: SpecialistTaskInput,
            *,
            usage: RunUsage | None = None,
            usage_limits: UsageLimits | None = None,
        ) -> SpecialistAttempt:
            del usage, usage_limits
            inputs.append(input)
            if self.attempt == 1:
                return SpecialistAttempt(
                    finding=SpecialistFindingDraft(summary="x" * (16 * 1024)),
                    messages=(failed_message,),
                )
            return SpecialistAttempt(
                finding=SpecialistFindingDraft(summary="accepted"),
            )

    actor_count = 0

    def factory(
        tools: tuple[object, ...],
        tool_capture: object,
        skill_invocation: object,
    ) -> _ValidationRetryingActor:
        nonlocal actor_count
        del tools, tool_capture, skill_invocation
        actor_count += 1
        return _ValidationRetryingActor(actor_count)

    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(id="market-data", actor_factory=factory),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=registry,
        scope_descriptors=scope_descriptors,
    )
    diagnostics = SpecialistExecutionDiagnostics()

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=registry,
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
        diagnostics=diagnostics,
    )

    assert contribution.attempt == 2
    assert inputs[0].validation_feedback is None
    assert inputs[1].validation_feedback == (
        "Return one valid structured Specialist finding within all stated "
        "output limits."
    )
    assert diagnostics.failed_attempts[0].messages == (failed_message,)


@pytest.mark.asyncio
async def test_execute_specialist_feeds_sdk_output_rejection_to_a_fresh_attempt() -> (
    None
):
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    inputs: list[SpecialistTaskInput] = []

    class _StructuredOutputRetryingActor:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt

        async def run(
            self,
            input: SpecialistTaskInput,
            *,
            usage: RunUsage | None = None,
            usage_limits: UsageLimits | None = None,
        ) -> SpecialistAttempt:
            del usage, usage_limits
            inputs.append(input)
            if self.attempt == 1:
                raise SpecialistInvocationFailure(
                    IncompleteToolCall("final_result was truncated"),
                    facts=SpecialistFailureFacts(
                        boundary=SpecialistModelBoundary.AZURE_OPENAI,
                        at_model_request_boundary=False,
                        terminal_output_tool_rejected=True,
                        count_limit_exhausted=False,
                        unreturned_model_requests=0,
                        usage_limits=specialist_usage_limits(),
                    ),
                    messages=(),
                )
            return SpecialistAttempt(
                finding=SpecialistFindingDraft(summary="accepted"),
            )

    actor_count = 0

    def factory(
        tools: tuple[object, ...],
        tool_capture: object,
        skill_invocation: object,
    ) -> _StructuredOutputRetryingActor:
        nonlocal actor_count
        del tools, tool_capture, skill_invocation
        actor_count += 1
        return _StructuredOutputRetryingActor(actor_count)

    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(id="market-data", actor_factory=factory),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=registry,
        scope_descriptors=scope_descriptors,
    )

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=registry,
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )

    assert contribution.attempt == 2
    assert inputs[1].validation_feedback == (
        "Return one valid final_result structured Specialist finding."
    )


@pytest.mark.asyncio
async def test_execute_specialist_reports_priced_model_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )

    class _Price:
        total_price = Decimal("1.25")

    def cost(_response: ModelResponse) -> _Price:
        return _Price()

    monkeypatch.setattr(ModelResponse, "cost", cost)
    response = ModelResponse(parts=[], model_name="priced-model")

    class _PricedActor:
        async def run(
            self,
            input: SpecialistTaskInput,
            *,
            usage: RunUsage | None = None,
            usage_limits: UsageLimits | None = None,
        ) -> SpecialistAttempt:
            del input, usage_limits
            assert usage is not None
            usage.incr(RunUsage(requests=1, input_tokens=4, output_tokens=2))
            return SpecialistAttempt(
                finding=SpecialistFindingDraft(summary="accepted"),
                messages=(response,),
            )

    registry = SpecialistRegistry(
        registrations=(SpecialistRegistration(id="market-data", actor=_PricedActor()),),
        tenant_eligible_ids=frozenset({"market-data"}),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=registry,
        scope_descriptors=scope_descriptors,
    )

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=registry,
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )
    accepted = promote_batch(batch, {contribution.task_id: contribution})

    assert contribution.usage.cost_usd == 1.25
    assert contribution.usage.cost_is_complete
    assert accepted.usage.cost_usd == 1.25
    assert accepted.usage.cost_is_complete
    assert SpecialistUsage().add(SpecialistUsage(cost_usd=1.25)).cost_usd == 1.25


@pytest.mark.asyncio
async def test_evidence_cache_overflow_stages_task_failed_without_evidence_ids() -> (
    None
):
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    oversized = EvidenceEnvelope(
        id="evidence-too-large",
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-placeholder",
        source="filing",
        source_url="https://example.test/filing",
        title="Annual filing",
        body="x" * (16 * 1024 + 1),
        excerpt="Too large.",
        as_of_date=date(2026, 9, 6),
    )
    specialist = _Specialist(
        finding=SpecialistFindingDraft(
            summary="Oversized evidence",
            evidence_ids=(oversized.id,),
        )
    )
    registry = SpecialistRegistry(
        registrations=(SpecialistRegistration(id="market-data", actor=specialist),),
        tenant_eligible_ids=frozenset({"market-data"}),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=registry,
        scope_descriptors=scope_descriptors,
    )
    specialist.evidence = (oversized.model_copy(update={"task_id": batch.tasks[0].id}),)

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=registry,
        scope_descriptors=scope_descriptors,
        catalog=RequestEvidenceCatalog(),
        context=_context(task_id=batch.tasks[0].id),
    )

    assert contribution.attempt == 1
    assert contribution.outcome == TaskFailed(task_id=batch.tasks[0].id)


@pytest.mark.asyncio
async def test_execute_specialist_persists_only_activated_skill_pins() -> None:
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=scope_descriptors,
    )
    pin = SkillPin(
        name="filing-analysis",
        version="1",
        content_hash="a" * 64,
    )
    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                actor=_Specialist(skill_pins=(pin,)),
            ),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
    )

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=registry,
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )
    accepted = promote_batch(batch, {contribution.task_id: contribution})

    assert contribution.skill_pins == (pin,)
    assert accepted.skill_pins == (
        TaskSkillPins(task_id=batch.tasks[0].id, pins=(pin,)),
    )


@pytest.mark.asyncio
async def test_execute_specialist_derives_one_data_gap_from_one_accepted_record() -> (
    None
):
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=scope_descriptors,
    )
    record = _unavailability_record(task_id=batch.tasks[0].id)
    registry = _gap_registry((record, record))

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=registry,
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )

    assert isinstance(contribution.outcome, TaskSucceeded)
    gap = contribution.outcome.result.data_gaps[0]
    assert gap.requested_coverage == "Apple revenue"
    assert gap.reason is ToolUnavailableReason.SOURCE_UNREACHABLE
    assert gap.provenance.unavailability_id == "unavailable-1"
    assert gap.provenance.tool_id == "filing-tool"
    assert len(contribution.outcome.result.data_gaps) == 1


@pytest.mark.asyncio
async def test_execute_specialist_checks_result_size_after_deriving_data_gaps() -> None:
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=scope_descriptors,
    )
    record = _unavailability_record(task_id=batch.tasks[0].id)
    gap = DataGap(
        requested_coverage=record.requested_coverage,
        reason=record.reason,
        provenance=GapProvenance(
            unavailability_id=record.id,
            tool_id=record.tool_id,
            source=record.source,
            observed_at=record.observed_at,
        ),
    )
    template = SpecialistResult(summary="", data_gaps=(gap,))
    exact = SpecialistFindingDraft(
        summary="x" * (16 * 1024 - template.canonical_json_size())
    )

    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=_gap_registry((record,), finding=exact),
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )

    assert isinstance(contribution.outcome, TaskSucceeded)
    assert contribution.outcome.result.canonical_json_size() == 16 * 1024
    failed = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=_gap_registry(
            (record,),
            finding=SpecialistFindingDraft(summary=f"{exact.summary}x"),
        ),
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )

    assert failed.attempt == 3
    assert failed.outcome == TaskFailed(task_id=batch.tasks[0].id)


@pytest.mark.asyncio
async def test_execute_specialist_rejects_conflicting_tool_call_provenance() -> None:
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=scope_descriptors,
    )
    record = _unavailability_record(task_id=batch.tasks[0].id)

    with pytest.raises(ValueError, match="Data Gap provenance"):
        await execute_specialist(
            batch.tasks[0],
            batch_id=batch.id,
            registry=_gap_registry(
                (record, record.model_copy(update={"id": "unavailable-2"}))
            ),
            scope_descriptors=scope_descriptors,
            context=_context(task_id=batch.tasks[0].id),
        )


@pytest.mark.asyncio
async def test_execute_specialist_rejects_missing_tool_call_provenance() -> None:
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=scope_descriptors,
    )
    record = _unavailability_record(task_id=batch.tasks[0].id).model_copy(
        update={"tool_call_id": ""}
    )

    with pytest.raises(ValueError, match="Data Gap provenance"):
        await execute_specialist(
            batch.tasks[0],
            batch_id=batch.id,
            registry=_gap_registry((record,)),
            scope_descriptors=scope_descriptors,
            context=_context(task_id=batch.tasks[0].id),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tenant_id": "tenant-b"}, "not eligible"),
        ({"request_id": "request-2"}, "not eligible"),
        ({"task_id": "task-2"}, "not eligible"),
        ({"attempt": 2}, "stale"),
        ({"tool_id": "unregistered"}, "not eligible"),
        ({"source": "private"}, "not eligible"),
    ],
)
@pytest.mark.asyncio
async def test_execute_specialist_rejects_ineligible_data_gap_provenance(
    overrides: dict[str, object], message: str
) -> None:
    scope_descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
    )
    batch = accept_initial_dispatch(
        _dispatch(),
        request_id="request-1",
        registry=_registry(),
        scope_descriptors=scope_descriptors,
    )
    record_values = {"task_id": batch.tasks[0].id, **overrides}
    record_task_id = record_values.pop("task_id")
    assert isinstance(record_task_id, str)
    record = _unavailability_record(task_id=record_task_id, **record_values)

    with pytest.raises(ValueError, match=message):
        await execute_specialist(
            batch.tasks[0],
            batch_id=batch.id,
            registry=_gap_registry((record,)),
            scope_descriptors=scope_descriptors,
            context=_context(task_id=batch.tasks[0].id),
        )


def test_data_gap_enforces_bounds_and_hides_internal_provenance() -> None:
    exact = DataGap(
        requested_coverage="🐍" * 64,
        reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
        provenance=GapProvenance(
            unavailability_id="u" * 64,
            tool_id="t" * 64,
            source="🐍" * 64,
            observed_at=datetime(2026, 9, 6, 12, tzinfo=UTC),
        ),
    )

    assert exact.view().model_dump() == {
        "requested_coverage": "🐍" * 64,
        "reason": ToolUnavailableReason.SOURCE_UNREACHABLE,
        "observed_at": datetime(2026, 9, 6, 12, tzinfo=UTC),
    }
    with pytest.raises(ValidationError, match="Data Gap coverage"):
        DataGap(
            requested_coverage="🐍" * 65,
            reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
            provenance=exact.provenance,
        )
    with pytest.raises(ValidationError, match="Data Gap identifier"):
        GapProvenance(
            unavailability_id="u" * 65,
            tool_id="tool",
            observed_at=datetime(2026, 9, 6, 12, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="Data Gap source"):
        GapProvenance(
            unavailability_id="gap",
            tool_id="tool",
            source="🐍" * 65,
            observed_at=datetime(2026, 9, 6, 12, tzinfo=UTC),
        )
    with pytest.raises(ValidationError):
        SpecialistResult(
            summary="Too many gaps",
            data_gaps=tuple(exact.model_copy() for _ in range(9)),
        )


@pytest.mark.asyncio
async def test_registry_freezes_tool_and_source_intersection_before_provider_access() -> (
    None
):
    calls: list[tuple[str, str]] = []

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        calls.append((source, query))
        return EvidenceEnvelope(
            id="evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            source=source,
            source_url="https://example.test/filing",
            title="Annual filing",
            body="Body",
            excerpt="Excerpt",
            as_of_date=date(2026, 9, 6),
        )

    captured: list[object] = []
    audit: list[tuple[str, str]] = []

    def record_audit(tool_id: str, status: str) -> None:
        audit.append((tool_id, status))
        if status == "completed":
            raise RuntimeError("audit transport failed")

    def factory(
        tools: tuple[object, ...],
        tool_capture: object,
        skill_invocation: object,
    ) -> _Specialist:
        del tool_capture, skill_invocation
        captured.extend(tools)
        return _Specialist()

    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                actor_factory=factory,
                allowed_tool_ids=frozenset({"filing-tool"}),
            ),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
        tool_registrations=(
            EvidenceToolRegistration(
                id="filing-tool",
                provider=provider,
                allowed_sources=frozenset({"filing", "private"}),
                allowed_queries=frozenset({"Apple"}),
                audit=record_audit,
            ),
        ),
        tenant_eligible_tool_ids=frozenset({"filing-tool"}),
    )
    registration = registry.resolve(
        "market-data",
        scope_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )

    actor = registry.bind_actor(
        registration,
        context=_context(),
    )

    assert isinstance(actor, _Specialist)
    assert len(captured) == 1
    tool = cast(Callable[[RunContext[None], str, str], Awaitable[object]], captured[0])
    with pytest.raises(ValueError, match="Evidence source is not eligible"):
        await tool(_tool_context(tool_name="filing-tool"), "private", "Apple")
    assert calls == []
    with pytest.raises(ValueError, match="Evidence query is not eligible"):
        await tool(_tool_context(tool_name="filing-tool"), "filing", "Broad query")
    assert calls == []
    await tool(_tool_context(tool_name="filing-tool"), "filing", "Apple")
    assert calls == [("filing", "Apple")]
    assert audit == [
        ("filing-tool", "rejected"),
        ("filing-tool", "rejected"),
        ("filing-tool", "started"),
        ("filing-tool", "completed"),
    ]


def test_registry_builds_a_no_tool_actor_when_scope_removes_all_tools() -> None:
    captured: list[tuple[object, ...]] = []

    def factory(
        tools: tuple[object, ...],
        tool_capture: object,
        skill_invocation: object,
    ) -> _Specialist:
        del tool_capture, skill_invocation
        captured.append(tools)
        return _Specialist()

    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                actor_factory=factory,
                allowed_tool_ids=frozenset({"filing-tool"}),
            ),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
        tool_registrations=(),
        tenant_eligible_tool_ids=frozenset(),
    )
    actor = registry.bind_actor(
        registry.registrations[0],
        context=_context().model_copy(
            update={
                "allowed_tool_ids": frozenset(),
                "allowed_sources": frozenset(),
                "allowed_queries": frozenset(),
            }
        ),
    )

    assert isinstance(actor, _Specialist)
    assert captured == [()]


def test_registry_keeps_a_direct_no_tool_actor_when_scope_removes_all_skills() -> None:
    direct_actor = _Specialist()
    registry = SpecialistRegistry(
        registrations=(SpecialistRegistration(id="market-data", actor=direct_actor),),
        tenant_eligible_ids=frozenset({"market-data"}),
        skill_registry=SpecialistSkillRegistry(
            registrations=(
                SkillRegistration(
                    name="market-skill",
                    version="1",
                    description="Market summary",
                    instructions="MARKET-FULL-INSTRUCTIONS",
                ),
            ),
            tenant_eligible_names=frozenset({"market-skill"}),
        ),
    )

    actor = registry.bind_actor(
        registry.registrations[0],
        context=_context().model_copy(
            update={
                "allowed_tool_ids": frozenset(),
                "allowed_sources": frozenset(),
                "allowed_queries": frozenset(),
            }
        ),
        scope_skill_names=frozenset(),
    )

    assert actor is direct_actor


def test_registry_binds_skill_activation_without_expanding_frozen_business_tools() -> (
    None
):
    captured_tools: list[Callable[..., object]] = []
    captured_invocation: list[SkillInvocation] = []

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise AssertionError("Tool must not run while binding a Skill")

    def factory(
        tools: tuple[Callable[..., object], ...],
        tool_capture: SpecialistToolCapture,
        skill_invocation: SkillInvocation | None,
    ) -> _Specialist:
        del tool_capture
        assert skill_invocation is not None
        captured_tools.extend(tools)
        captured_invocation.append(skill_invocation)
        return _Specialist()

    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                actor_factory=factory,
                allowed_tool_ids=frozenset({"filing-tool"}),
                allowed_skill_names=frozenset({"market-skill"}),
            ),
        ),
        tenant_eligible_ids=frozenset({"market-data"}),
        tool_registrations=(
            EvidenceToolRegistration(
                id="filing-tool",
                provider=provider,
                allowed_sources=frozenset({"filing"}),
                allowed_queries=frozenset({"Apple"}),
            ),
        ),
        tenant_eligible_tool_ids=frozenset({"filing-tool"}),
        skill_registry=SpecialistSkillRegistry(
            registrations=(
                SkillRegistration(
                    name="shared-skill",
                    version="1",
                    description="Shared summary",
                    instructions="SHARED-FULL-INSTRUCTIONS",
                    required_tool_ids=frozenset({"filing-tool"}),
                ),
                SkillRegistration(
                    name="market-skill",
                    version="1",
                    description="Market summary",
                    instructions="MARKET-FULL-INSTRUCTIONS",
                ),
            ),
            tenant_eligible_names=frozenset({"shared-skill", "market-skill"}),
            shared_skill_names=frozenset({"shared-skill"}),
        ),
    )

    actor = registry.bind_actor(
        registry.registrations[0],
        context=_context(),
        scope_skill_names=frozenset({"shared-skill", "market-skill"}),
    )

    assert isinstance(actor, _Specialist)
    assert [_tool_name(tool) for tool in captured_tools] == [
        "filing-tool",
        "activate_skill",
    ]
    invocation = captured_invocation[0]
    assert [summary.name for summary in invocation.summaries] == [
        "shared-skill",
        "market-skill",
    ]
    assert (
        invocation.activate("shared-skill").instructions == "SHARED-FULL-INSTRUCTIONS"
    )


def _tool_name(tool: Callable[..., object]) -> str:
    name = getattr(tool, "__name__", None)
    assert isinstance(name, str)
    return name
