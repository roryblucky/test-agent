"""Public first-batch validation and acceptance coverage."""

from collections.abc import Awaitable, Callable
from datetime import date
from typing import cast

import pytest
from pydantic import ValidationError

from app.langgraph_v2.agent_batch import (
    BatchContribution,
    DispatchBatch,
    EvidenceToolRegistration,
    SpecialistAttempt,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistResult,
    StructuredOutputInvalid,
    TaskProposal,
    TaskSucceeded,
    accept_initial_dispatch,
    execute_specialist,
    promote_batch,
)
from app.langgraph_v2.agent_evidence import EvidenceEnvelope, EvidenceInvocationContext
from app.langgraph_v2.agent_scope import SpecialistDescriptor


class _Specialist:
    def __init__(self, finding: SpecialistFindingDraft | None = None) -> None:
        self.finding = finding or SpecialistFindingDraft(summary="No-tool finding")

    async def run(self, input: object) -> SpecialistAttempt:
        del input
        return SpecialistAttempt(finding=self.finding)


def _context(*, task_id: str = "task-1") -> EvidenceInvocationContext:
    return EvidenceInvocationContext(
        tenant_id="tenant-a",
        request_id="request-1",
        task_id=task_id,
        allowed_tool_ids=frozenset({"filing-tool", "unregistered"}),
        allowed_sources=frozenset({"filing"}),
        allowed_queries=frozenset({"Apple"}),
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
async def test_registry_freezes_tool_and_source_intersection_before_provider_access() -> None:
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

    def factory(tools: tuple[object, ...], returned_evidence: object) -> _Specialist:
        del returned_evidence
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
    tool = cast(
        Callable[[str, str], Awaitable[object]], captured[0]
    )
    with pytest.raises(ValueError, match="Evidence source is not eligible"):
        await tool("private", "Apple")
    assert calls == []
    with pytest.raises(ValueError, match="Evidence query is not eligible"):
        await tool("filing", "Broad query")
    assert calls == []
    await tool("filing", "Apple")
    assert calls == [("filing", "Apple")]
    assert audit == [
        ("filing-tool", "rejected"),
        ("filing-tool", "rejected"),
        ("filing-tool", "started"),
        ("filing-tool", "completed"),
    ]


def test_registry_builds_a_no_tool_actor_when_scope_removes_all_tools() -> None:
    captured: list[tuple[object, ...]] = []

    def factory(tools: tuple[object, ...], returned_evidence: object) -> _Specialist:
        del returned_evidence
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
