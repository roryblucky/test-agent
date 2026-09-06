"""Public first-batch validation and acceptance coverage."""

from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from typing import cast

import pytest
from pydantic import ValidationError
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from app.langgraph_v2.agent_batch import (
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
    StructuredOutputInvalid,
    TaskProposal,
    TaskSkillPins,
    TaskSucceeded,
    accept_initial_dispatch,
    execute_specialist,
    promote_batch,
)
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    EvidenceInvocationContext,
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


class _Specialist:
    def __init__(
        self,
        finding: SpecialistFindingDraft | None = None,
        skill_pins: tuple[SkillPin, ...] = (),
        unavailability: tuple[ToolUnavailabilityRecord, ...] = (),
    ) -> None:
        self.finding = finding or SpecialistFindingDraft(summary="No-tool finding")
        self.skill_pins = skill_pins
        self.unavailability = unavailability

    async def run(self, input: object) -> SpecialistAttempt:
        del input
        return SpecialistAttempt(
            finding=self.finding,
            skill_pins=self.skill_pins,
            unavailability=self.unavailability,
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
) -> SpecialistRegistry:
    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise AssertionError("Direct Specialist actor must not call the binding")

    def factory(
        tools: tuple[object, ...],
        tool_capture: object,
        skill_invocation: object,
    ) -> _Specialist:
        del tools, tool_capture, skill_invocation
        return _Specialist(unavailability=records)

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
    exact = SpecialistFindingDraft(
        summary="x" * (16 * 1024 - len('{"evidence_ids":[],"summary":""}'))
    )
    contribution = await execute_specialist(
        batch.tasks[0],
        batch_id=batch.id,
        registry=_registry(finding=exact),
        scope_descriptors=scope_descriptors,
        context=_context(task_id=batch.tasks[0].id),
    )

    assert contribution.outcome.result.summary == exact.summary

    too_large = SpecialistFindingDraft(summary=f"{exact.summary}x")
    with pytest.raises(
        StructuredOutputInvalid, match="Specialist finding exceeds 16 KiB"
    ):
        await execute_specialist(
            batch.tasks[0],
            batch_id=batch.id,
            registry=_registry(finding=too_large),
            scope_descriptors=scope_descriptors,
            context=_context(task_id=batch.tasks[0].id),
        )


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
    assert accepted.skill_pins == (TaskSkillPins(task_id=batch.tasks[0].id, pins=(pin,)),)


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

    gap = contribution.outcome.result.data_gaps[0]
    assert gap.requested_coverage == "Apple revenue"
    assert gap.reason is ToolUnavailableReason.SOURCE_UNREACHABLE
    assert gap.provenance.unavailability_id == "unavailable-1"
    assert gap.provenance.tool_id == "filing-tool"
    assert len(contribution.outcome.result.data_gaps) == 1


@pytest.mark.asyncio
async def test_execute_specialist_rejects_conflicting_tool_call_provenance(
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
    record = _unavailability_record(
        task_id=record_task_id, **record_values
    )

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


def test_registry_keeps_a_direct_no_tool_actor_when_scope_removes_all_skills() -> (
    None
):
    direct_actor = _Specialist()
    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(id="market-data", actor=direct_actor),
        ),
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
    assert invocation.activate("shared-skill").instructions == "SHARED-FULL-INSTRUCTIONS"


def _tool_name(tool: Callable[..., object]) -> str:
    name = getattr(tool, "__name__", None)
    assert isinstance(name, str)
    return name
