"""Public first-batch validation and acceptance coverage."""

import pytest
from pydantic import ValidationError

from app.langgraph_v2.agent_batch import (
    BatchContribution,
    DispatchBatch,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistResult,
    StructuredOutputInvalid,
    TaskProposal,
    TaskSucceeded,
    accept_initial_dispatch,
    promote_batch,
)
from app.langgraph_v2.agent_scope import SpecialistDescriptor


class _Specialist:
    async def run(self, input: object) -> SpecialistFindingDraft:
        del input
        return SpecialistFindingDraft(summary="No-tool finding")


def _registry(
    *, tenant_eligible_ids: frozenset[str] = frozenset({"market-data"})
) -> SpecialistRegistry:
    return SpecialistRegistry(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                actor=_Specialist(),
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
        with pytest.raises(StructuredOutputInvalid, match="Specialist finding exceeds 16 KiB"):
            finding.require_canonical_size()
