"""Rolling Coordination Round acceptance and projection coverage."""

from collections.abc import Callable
from typing import Any, cast

import pytest

from app.langgraph_v2.agent_batch import (
    AcceptedBatch,
    DispatchBatch,
    SpecialistCatalog,
    SpecialistRegistration,
    SpecialistResult,
    TaskFailed,
    TaskProposal,
    TaskSucceeded,
    batch_id_for,
    task_id_for,
)
from app.langgraph_v2.agent_coordination import (
    CoordinationCandidateRejected,
    CoordinationInvariantError,
    CoordinationRound,
    CoordinationStopped,
    CoordinatorDecisionExhausted,
    CoordinatorInput,
    accept_coordination_dispatch,
    accept_coordination_finish,
    decide_coordination_round,
    materialize_specialist_context,
    project_coordinator_input,
    validate_coordination_rounds,
    validate_coordinator_decision,
)
from app.langgraph_v2.agent_scope import SpecialistDescriptor


def _catalog() -> SpecialistCatalog:
    return SpecialistCatalog(
        registrations=(
            SpecialistRegistration(
                id="market-data",
                description="market-data",
                actor=cast(Any, object()),
            ),
        ),
    )


def _accepted_batch(batch_id: str, task_id: str) -> AcceptedBatch:
    return AcceptedBatch(
        id=batch_id,
        outcomes=(
            TaskSucceeded(
                task_id=task_id,
                result=SpecialistResult(
                    summary="The first-round market finding.",
                    evidence_ids=("evidence-1",),
                ),
            ),
        ),
    )


def _accepted_rounds(
    *,
    tasks_per_round: int,
    count: int,
    summary_for: Callable[[int, int], str] = lambda round_index, task_index: (
        f"Finding {round_index}-{task_index}."
    ),
) -> tuple[tuple[CoordinationRound, ...], dict[str, AcceptedBatch]]:
    """Build accepted dispatch rounds through the public acceptance boundary."""
    rounds: tuple[CoordinationRound, ...] = ()
    accepted_batches: dict[str, AcceptedBatch] = {}
    for round_index in range(count):
        accepted = accept_coordination_dispatch(
            DispatchBatch(
                kind="dispatch",
                tasks=tuple(
                    TaskProposal(
                        specialist_id="market-data",
                        objective=f"Assess dimension {round_index}-{task_index}.",
                    )
                    for task_index in range(tasks_per_round)
                ),
            ),
            request_id="request-1",
            rounds=rounds,
            accepted_batches=accepted_batches,
            specialist_catalog=_catalog(),
        )
        assert accepted.batch_id is not None
        rounds += (accepted,)
        accepted_batches[accepted.batch_id] = AcceptedBatch(
            id=accepted.batch_id,
            outcomes=tuple(
                TaskSucceeded(
                    task_id=task.id,
                    result=SpecialistResult(
                        summary=summary_for(round_index, task_index)
                    ),
                )
                for task_index, task in enumerate(accepted.tasks)
            ),
        )
    return rounds, accepted_batches


def test_follow_up_dispatch_projects_and_materializes_only_prior_successes() -> None:
    scope = (SpecialistDescriptor(id="market-data", description="Market data"),)
    first = accept_coordination_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Assess market conditions.",
                ),
            ),
        ),
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        specialist_catalog=_catalog(),
    )
    assert first.batch_id is not None
    accepted_batches = {
        first.batch_id: _accepted_batch(first.batch_id, first.tasks[0].id)
    }
    second = accept_coordination_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Assess implications for the next step.",
                    context_task_ids=(first.tasks[0].id,),
                ),
            ),
        ),
        request_id="request-1",
        rounds=(first,),
        accepted_batches=accepted_batches,
        specialist_catalog=_catalog(),
    )

    coordinator_input = project_coordinator_input(
        standalone_query="Market outlook",
        intent="market_outlook",
        specialist_descriptors=scope,
        rounds=(first,),
        accepted_batches=accepted_batches,
    )
    context = materialize_specialist_context(
        second.tasks[0],
        current_round=second.revision,
        rounds=(first,),
        accepted_batches=accepted_batches,
    )

    assert first.revision == 1
    assert second.revision == 2
    assert coordinator_input.prior_results[0].task_id == first.tasks[0].id
    assert (
        coordinator_input.prior_results[0].summary == "The first-round market finding."
    )
    assert coordinator_input.remaining_task_slots == 31
    assert coordinator_input.dispatch_allowed is True
    assert context == coordinator_input.prior_results


def test_failed_prior_task_is_projected_without_diagnostics() -> None:
    scope = (SpecialistDescriptor(id="market-data", description="Market data"),)
    first = accept_coordination_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Assess market conditions.",
                ),
            ),
        ),
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        specialist_catalog=_catalog(),
    )
    assert first.batch_id is not None
    failed = AcceptedBatch(
        id=first.batch_id,
        outcomes=(TaskFailed(task_id=first.tasks[0].id),),
    )

    input = project_coordinator_input(
        standalone_query="Market outlook",
        intent="market_outlook",
        specialist_descriptors=scope,
        rounds=(first,),
        accepted_batches={first.batch_id: failed},
    )

    assert input.prior_results == ()
    assert input.failed_tasks[0].task_id == first.tasks[0].id
    assert input.failed_tasks[0].objective == "Assess market conditions."
    assert input.remaining_task_slots == 31
    assert input.dispatch_allowed is True


def test_prompt_visible_validation_rejects_unknown_context_without_accepting() -> None:
    input = CoordinatorInput(
        standalone_query="Market outlook",
        intent="market_outlook",
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
        remaining_task_slots=32,
        dispatch_allowed=True,
    )
    decision = DispatchBatch(
        kind="dispatch",
        tasks=(
            TaskProposal(
                specialist_id="market-data",
                objective="Invalid dependency.",
                context_task_ids=("missing",),
            ),
        ),
    )

    with pytest.raises(
        CoordinationCandidateRejected,
        match="Task context is not an accepted prior success",
    ):
        validate_coordinator_decision(input, decision)


@pytest.mark.asyncio
async def test_actor_output_exhaustion_stops_without_an_accepted_round() -> None:
    class _ExhaustedCoordinator:
        def __init__(self) -> None:
            self.calls = 0

        async def decide(self, input: CoordinatorInput) -> CoordinatorDecisionExhausted:
            del input
            self.calls += 1
            return CoordinatorDecisionExhausted(
                reason="Task context is not an accepted prior success"
            )

    actor = _ExhaustedCoordinator()
    result = await decide_coordination_round(
        actor,
        CoordinatorInput(
            standalone_query="Market outlook",
            intent="market_outlook",
            specialist_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
            remaining_task_slots=32,
            dispatch_allowed=True,
        ),
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        specialist_catalog=_catalog(),
    )

    assert result == CoordinationStopped("coordination_invalid")
    assert actor.calls == 1


def test_coordination_acceptance_persists_canonical_objectives_and_round_manifest() -> (
    None
):
    accepted = accept_coordination_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Cafe\u0301\r\nanalysis\x00",
                ),
            ),
        ),
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        specialist_catalog=_catalog(),
    )

    assert accepted.tasks[0].objective == "Café analysis"
    assert validate_coordination_rounds((accepted,), request_id="request-1") == (
        accepted,
    )
    tampered = accepted.model_copy(update={"batch_id": "batch_tampered"})
    with pytest.raises(CoordinationInvariantError, match="batch is invalid"):
        validate_coordination_rounds((tampered,), request_id="request-1")


def test_fourth_dispatch_projects_finish_only_policy() -> None:
    rounds, accepted_batches = _accepted_rounds(tasks_per_round=7, count=4)
    finish = accept_coordination_finish(request_id="request-1", rounds=rounds)

    assert finish.revision == 5
    input = project_coordinator_input(
        standalone_query="Market outlook",
        intent="market_outlook",
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
        rounds=rounds,
        accepted_batches=accepted_batches,
    )
    decision = DispatchBatch(
        kind="dispatch",
        tasks=(
            TaskProposal(
                specialist_id="market-data",
                objective="Illegal fifth dispatch.",
            ),
        ),
    )

    assert input.remaining_task_slots == 4
    assert input.dispatch_allowed is False
    with pytest.raises(CoordinationCandidateRejected, match="coordination_limit"):
        validate_coordinator_decision(input, decision)


def test_task_limit_takes_precedence_when_all_32_tasks_are_accepted() -> None:
    rounds, accepted_batches = _accepted_rounds(tasks_per_round=8, count=4)

    with pytest.raises(CoordinationCandidateRejected, match="task_limit"):
        accept_coordination_dispatch(
            DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="One task too many.",
                    ),
                ),
            ),
            request_id="request-1",
            rounds=rounds,
            accepted_batches=accepted_batches,
            specialist_catalog=_catalog(),
        )


def test_checkpoint_round_validation_rejects_a_fifth_dispatch_round() -> None:
    rounds, _ = _accepted_rounds(tasks_per_round=1, count=4)
    fifth_finish = accept_coordination_finish(request_id="request-1", rounds=rounds)
    fifth_task = (
        rounds[-1]
        .tasks[0]
        .model_copy(
            update={
                "id": task_id_for(request_id="request-1", round=5, dispatch_order=0)
            }
        )
    )
    fifth_dispatch = fifth_finish.model_copy(
        update={
            "kind": "dispatch",
            "batch_id": batch_id_for(request_id="request-1", round=5),
            "tasks": (fifth_task,),
        }
    )

    with pytest.raises(CoordinationInvariantError, match="dispatch limit"):
        validate_coordination_rounds(rounds + (fifth_dispatch,), request_id="request-1")


def test_context_reference_count_accepts_eight_and_rejects_nine_before_send() -> None:
    first, accepted_batches = _accepted_rounds(tasks_per_round=8, count=1)
    first_round = first[0]
    first_batch = accepted_batches[first_round.batch_id or ""]
    context_ids = tuple(outcome.task_id for outcome in first_batch.outcomes)
    accepted = accept_coordination_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Use all accepted context references.",
                    context_task_ids=context_ids,
                ),
            ),
        ),
        request_id="request-1",
        rounds=first,
        accepted_batches=accepted_batches,
        specialist_catalog=_catalog(),
    )

    assert accepted.tasks[0].context_task_ids == context_ids
    assert (
        tuple(
            result.task_id
            for result in materialize_specialist_context(
                accepted.tasks[0],
                current_round=accepted.revision,
                rounds=first,
                accepted_batches=accepted_batches,
            )
        )
        == context_ids
    )

    with pytest.raises(CoordinationCandidateRejected, match="Task context is invalid"):
        accept_coordination_dispatch(
            DispatchBatch.model_construct(
                kind="dispatch",
                tasks=(
                    TaskProposal.model_construct(
                        specialist_id="market-data",
                        objective="Too many context references.",
                        context_task_ids=context_ids + ("missing",),
                    ),
                ),
            ),
            request_id="request-1",
            rounds=first,
            accepted_batches=accepted_batches,
            specialist_catalog=_catalog(),
        )
