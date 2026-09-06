"""Rolling Coordination Round acceptance and projection coverage."""

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from app.langgraph_v2.agent_batch import (
    AcceptedBatch,
    ActiveBatch,
    DispatchBatch,
    PriorResultView,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistResult,
    TaskFailed,
    TaskProposal,
    TaskSucceeded,
    batch_id_for,
    task_id_for,
)
from app.langgraph_v2.agent_coordination import (
    MAX_COORDINATOR_CONTEXT_BYTES,
    MAX_COORDINATOR_RESULT_BYTES,
    MAX_SPECIALIST_CONTEXT_BYTES,
    AcceptedCoordinationDispatch,
    CoordinationCandidateRejected,
    CoordinationContextLimitExceeded,
    CoordinationInvariantError,
    CoordinationRound,
    CoordinationStopped,
    CoordinatorInput,
    CoordinatorOutputInvalid,
    accept_coordination_dispatch,
    accept_coordination_finish,
    decide_coordination_round,
    materialize_specialist_context,
    project_coordinator_input,
    validate_active_batch_coordination_round,
    validate_coordination_rounds,
)
from app.langgraph_v2.agent_scope import SpecialistDescriptor


def _registry() -> SpecialistRegistry:
    return SpecialistRegistry(
        registrations=(SpecialistRegistration(id="market-data"),),
        tenant_eligible_ids=frozenset({"market-data"}),
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
    scope = (SpecialistDescriptor(id="market-data", description="Market data"),)
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
            registry=_registry(),
            scope_descriptors=scope,
        )
        rounds += (accepted.round,)
        accepted_batches[accepted.active_batch.id] = AcceptedBatch(
            id=accepted.active_batch.id,
            outcomes=tuple(
                TaskSucceeded(
                    task_id=task.id,
                    result=SpecialistResult(
                        summary=summary_for(round_index, task_index)
                    ),
                )
                for task_index, task in enumerate(accepted.active_batch.tasks)
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
        registry=_registry(),
        scope_descriptors=scope,
    )
    accepted_batches = {
        first.active_batch.id: _accepted_batch(
            first.active_batch.id, first.active_batch.tasks[0].id
        )
    }
    second = accept_coordination_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Assess implications for the next step.",
                    context_task_ids=(first.active_batch.tasks[0].id,),
                ),
            ),
        ),
        request_id="request-1",
        rounds=(first.round,),
        accepted_batches=accepted_batches,
        registry=_registry(),
        scope_descriptors=scope,
    )

    coordinator_input = project_coordinator_input(
        standalone_query="Market outlook",
        intent="market_outlook",
        specialist_descriptors=scope,
        rounds=(first.round,),
        accepted_batches=accepted_batches,
    )
    context = materialize_specialist_context(
        second.active_batch.tasks[0],
        current_round=second.round.revision,
        rounds=(first.round,),
        accepted_batches=accepted_batches,
    )

    assert first.round.revision == 1
    assert second.round.revision == 2
    assert coordinator_input.prior_results[0].task_id == first.active_batch.tasks[0].id
    assert coordinator_input.prior_results[0].summary == "The first-round market finding."
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
        registry=_registry(),
        scope_descriptors=scope,
    )
    failed = AcceptedBatch(
        id=first.active_batch.id,
        outcomes=(TaskFailed(task_id=first.active_batch.tasks[0].id),),
    )

    input = project_coordinator_input(
        standalone_query="Market outlook",
        intent="market_outlook",
        specialist_descriptors=scope,
        rounds=(first.round,),
        accepted_batches={first.active_batch.id: failed},
    )

    assert input.prior_results == ()
    assert input.failed_tasks[0].task_id == first.active_batch.tasks[0].id
    assert input.failed_tasks[0].objective == "Assess market conditions."


@pytest.mark.asyncio
async def test_rejected_candidate_uses_one_frozen_input_repair_without_revision_gap() -> (
    None
):
    class _RepairingCoordinator:
        def __init__(self) -> None:
            self.inputs: list[CoordinatorInput] = []
            self.repair_inputs: list[CoordinatorInput] = []

        async def decide(self, input: CoordinatorInput) -> DispatchBatch:
            self.inputs.append(input)
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Invalid dependency.",
                        context_task_ids=("missing",),
                    ),
                ),
            )

        async def repair(
            self,
            input: CoordinatorInput,
            *,
            rejection: str,
        ) -> DispatchBatch:
            assert rejection == "Task context is not an accepted prior success"
            self.repair_inputs.append(input)
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Valid repaired objective.",
                    ),
                ),
            )

    actor = _RepairingCoordinator()
    input = CoordinatorInput(
        standalone_query="Market outlook",
        intent="market_outlook",
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )
    result = await decide_coordination_round(
        actor,
        input,
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        registry=_registry(),
        scope_descriptors=input.specialist_descriptors,
    )

    assert isinstance(result, AcceptedCoordinationDispatch)
    assert result.round.revision == 1
    assert actor.inputs == [input]
    assert actor.repair_inputs == [input]


@pytest.mark.asyncio
async def test_second_rejected_candidate_stops_without_an_accepted_round() -> None:
    class _InvalidCoordinator:
        async def decide(self, input: CoordinatorInput) -> DispatchBatch:
            del input
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Invalid dependency.",
                        context_task_ids=("missing",),
                    ),
                ),
            )

        async def repair(
            self,
            input: CoordinatorInput,
            *,
            rejection: str,
        ) -> DispatchBatch:
            del input, rejection
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Still invalid.",
                        context_task_ids=("missing",),
                    ),
                ),
            )

    result = await decide_coordination_round(
        _InvalidCoordinator(),
        CoordinatorInput(
            standalone_query="Market outlook",
            intent="market_outlook",
            specialist_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
        ),
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        registry=_registry(),
        scope_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )

    assert result == CoordinationStopped("coordination_invalid")


@pytest.mark.asyncio
async def test_schema_invalid_actor_output_gets_one_repair_then_stops() -> None:
    class _SchemaInvalidCoordinator:
        def __init__(self) -> None:
            self.calls = 0

        async def decide(self, input: CoordinatorInput) -> DispatchBatch:
            del input
            self.calls += 1
            raise CoordinatorOutputInvalid("invalid output")

        async def repair(
            self,
            input: CoordinatorInput,
            *,
            rejection: str,
        ) -> DispatchBatch:
            del input
            assert rejection == "Coordinator decision is invalid"
            self.calls += 1
            raise CoordinatorOutputInvalid("still invalid")

    actor = _SchemaInvalidCoordinator()
    result = await decide_coordination_round(
        actor,
        CoordinatorInput(
            standalone_query="Market outlook",
            intent="market_outlook",
            specialist_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
        ),
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        registry=_registry(),
        scope_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )

    assert result == CoordinationStopped("coordination_invalid")
    assert actor.calls == 2


def test_context_serialization_change_after_acceptance_is_fatal() -> None:
    scope = (SpecialistDescriptor(id="market-data", description="Market data"),)
    first = accept_coordination_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Establish the premise.",
                ),
            ),
        ),
        request_id="request-1",
        rounds=(),
        accepted_batches={},
        registry=_registry(),
        scope_descriptors=scope,
    )
    accepted = _accepted_batch(first.active_batch.id, first.active_batch.tasks[0].id)
    second = accept_coordination_dispatch(
        DispatchBatch(
            kind="dispatch",
            tasks=(
                TaskProposal(
                    specialist_id="market-data",
                    objective="Use the premise.",
                    context_task_ids=(first.active_batch.tasks[0].id,),
                ),
            ),
        ),
        request_id="request-1",
        rounds=(first.round,),
        accepted_batches={first.active_batch.id: accepted},
        registry=_registry(),
        scope_descriptors=scope,
    )
    changed = accepted.model_copy(
        update={
            "outcomes": (
                TaskSucceeded(
                    task_id=first.active_batch.tasks[0].id,
                    result=SpecialistResult(
                        summary="X" * len("The first-round market finding.")
                    ),
                ),
            )
        }
    )

    with pytest.raises(CoordinationInvariantError, match="changed after validation"):
        materialize_specialist_context(
            second.active_batch.tasks[0],
            current_round=second.round.revision,
            rounds=(first.round,),
            accepted_batches={first.active_batch.id: changed},
        )


def test_coordination_acceptance_persists_canonical_objectives_and_round_manifest() -> (
    None
):
    scope = (SpecialistDescriptor(id="market-data", description="Market data"),)
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
        registry=_registry(),
        scope_descriptors=scope,
    )

    assert accepted.active_batch.tasks[0].objective == "Café analysis"
    assert accepted.round.tasks[0].objective == "Café analysis"
    validate_active_batch_coordination_round(
        accepted.active_batch,
        rounds=(accepted.round,),
    )

    tampered = ActiveBatch(
        id=accepted.active_batch.id,
        round=accepted.active_batch.round,
        tasks=(
            replace(accepted.active_batch.tasks[0], objective="Different objective."),
        ),
    )
    with pytest.raises(CoordinationInvariantError, match="Active Batch Coordination"):
        validate_active_batch_coordination_round(
            tampered,
            rounds=(accepted.round,),
        )


@pytest.mark.asyncio
async def test_fourth_dispatch_only_allows_finish_and_fifth_dispatch_stops() -> None:
    rounds, accepted_batches = _accepted_rounds(tasks_per_round=7, count=4)
    finish = accept_coordination_finish(request_id="request-1", rounds=rounds)

    assert finish.revision == 5

    class _FifthDispatchCoordinator:
        def __init__(self) -> None:
            self.repairs = 0

        async def decide(self, input: CoordinatorInput) -> DispatchBatch:
            del input
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Illegal fifth dispatch.",
                    ),
                ),
            )

        async def repair(
            self, input: CoordinatorInput, *, rejection: str
        ) -> DispatchBatch:
            del input
            assert rejection == "coordination_limit"
            self.repairs += 1
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Illegal fifth dispatch.",
                    ),
                ),
            )

    actor = _FifthDispatchCoordinator()
    stopped = await decide_coordination_round(
        actor,
        CoordinatorInput(
            standalone_query="Market outlook",
            intent="market_outlook",
            specialist_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
        ),
        request_id="request-1",
        rounds=rounds,
        accepted_batches=accepted_batches,
        registry=_registry(),
        scope_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
        ),
    )

    assert stopped == CoordinationStopped("coordination_limit")
    assert actor.repairs == 1


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
            registry=_registry(),
            scope_descriptors=(
                SpecialistDescriptor(id="market-data", description="Market data"),
            ),
        )


def test_checkpoint_round_validation_rejects_a_fifth_dispatch_round() -> None:
    rounds, _ = _accepted_rounds(tasks_per_round=1, count=4)
    fifth_finish = accept_coordination_finish(request_id="request-1", rounds=rounds)
    fifth_task = rounds[-1].tasks[0].model_copy(
        update={
            "id": task_id_for(request_id="request-1", round=5, dispatch_order=0)
        }
    )
    fifth_dispatch = fifth_finish.model_copy(
        update={
            "kind": "dispatch",
            "batch_id": batch_id_for(request_id="request-1", round=5),
            "tasks": (fifth_task,),
        }
    )

    with pytest.raises(CoordinationInvariantError, match="dispatch limit"):
        validate_coordination_rounds(
            rounds + (fifth_dispatch,), request_id="request-1"
        )


def test_context_reference_count_and_specialist_context_bytes_fail_before_send() -> (
    None
):
    scope = (SpecialistDescriptor(id="market-data", description="Market data"),)
    first, accepted_batches = _accepted_rounds(tasks_per_round=5, count=1)
    first_round = first[0]
    first_batch = accepted_batches[first_round.batch_id or ""]
    context_ids = tuple(outcome.task_id for outcome in first_batch.outcomes)

    with pytest.raises(CoordinationCandidateRejected, match="Task context is invalid"):
        accept_coordination_dispatch(
            DispatchBatch.model_construct(
                kind="dispatch",
                tasks=(
                    TaskProposal.model_construct(
                        specialist_id="market-data",
                        objective="Too many context references.",
                        context_task_ids=context_ids
                        + ("missing-1", "missing-2", "missing-3", "missing-4"),
                    ),
                ),
            ),
            request_id="request-1",
            rounds=first,
            accepted_batches=accepted_batches,
            registry=_registry(),
            scope_descriptors=scope,
        )

    rounds, accepted_batches = _accepted_rounds(
        tasks_per_round=5,
        count=1,
        summary_for=lambda _round, _task: "x" * 15_900,
    )
    first_round = rounds[0]
    batch = accepted_batches[first_round.batch_id or ""]
    with pytest.raises(
        CoordinationCandidateRejected, match="Specialist context exceeds 64 KiB"
    ):
        accept_coordination_dispatch(
            DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Use all prior results.",
                        context_task_ids=tuple(
                            outcome.task_id for outcome in batch.outcomes
                        ),
                    ),
                ),
            ),
            request_id="request-1",
            rounds=rounds,
            accepted_batches=accepted_batches,
            registry=_registry(),
            scope_descriptors=scope,
        )


def test_coordinator_projection_rejects_single_and_aggregate_context_limits() -> None:
    scope = (SpecialistDescriptor(id="market-data", description="Market data"),)
    rounds, accepted_batches = _accepted_rounds(
        tasks_per_round=1,
        count=1,
        summary_for=lambda _round, _task: "x" * MAX_COORDINATOR_RESULT_BYTES,
    )
    with pytest.raises(CoordinationContextLimitExceeded):
        project_coordinator_input(
            standalone_query="Market outlook",
            intent="market_outlook",
            specialist_descriptors=scope,
            rounds=rounds,
            accepted_batches=accepted_batches,
        )


def test_specialist_context_byte_limit_accepts_exactly_64_kib_and_rejects_one_more() -> (
    None
):
    scope = (SpecialistDescriptor(id="market-data", description="Market data"),)

    def summaries_for_context_size(target: int) -> tuple[str, ...]:
        views = tuple(
            PriorResultView(
                task_id=task_id_for(
                    request_id="request-1", round=1, dispatch_order=index
                ),
                summary="",
            )
            for index in range(5)
        )
        baseline = len(
            json.dumps(
                [view.model_dump(mode="json") for view in views],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        quotient, remainder = divmod(target - baseline, len(views))
        return tuple("x" * (quotient + (index < remainder)) for index in range(5))

    def dispatch_with_context_size(target: int) -> AcceptedCoordinationDispatch:
        summaries = summaries_for_context_size(target)
        rounds, accepted_batches = _accepted_rounds(
            tasks_per_round=5,
            count=1,
            summary_for=lambda _round, task: summaries[task],
        )
        first_round = rounds[0]
        batch = accepted_batches[first_round.batch_id or ""]
        return accept_coordination_dispatch(
            DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="market-data",
                        objective="Use the exact prior context.",
                        context_task_ids=tuple(
                            outcome.task_id for outcome in batch.outcomes
                        ),
                    ),
                ),
            ),
            request_id="request-1",
            rounds=rounds,
            accepted_batches=accepted_batches,
            registry=_registry(),
            scope_descriptors=scope,
        )

    exact = dispatch_with_context_size(MAX_SPECIALIST_CONTEXT_BYTES)
    assert exact.active_batch.tasks[0].context_json_bytes == MAX_SPECIALIST_CONTEXT_BYTES

    with pytest.raises(
        CoordinationCandidateRejected, match="Specialist context exceeds 64 KiB"
    ):
        dispatch_with_context_size(MAX_SPECIALIST_CONTEXT_BYTES + 1)


def test_coordinator_projection_byte_limits_accept_exactly_and_reject_one_more() -> (
    None
):
    scope = (SpecialistDescriptor(id="market-data", description="Market data"),)

    def summary_for_exact_result_size(target: int, task_index: int) -> str:
        task_id = task_id_for(
            request_id="request-1", round=1, dispatch_order=task_index
        )
        baseline = len(
            json.dumps(
                PriorResultView(task_id=task_id, summary="").model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return "x" * (target - baseline)

    rounds, accepted_batches = _accepted_rounds(
        tasks_per_round=1,
        count=1,
        summary_for=lambda _round, task: summary_for_exact_result_size(
            MAX_COORDINATOR_RESULT_BYTES, task
        ),
    )
    assert project_coordinator_input(
        standalone_query="Market outlook",
        intent="market_outlook",
        specialist_descriptors=scope,
        rounds=rounds,
        accepted_batches=accepted_batches,
    ).prior_results

    rounds, accepted_batches = _accepted_rounds(
        tasks_per_round=1,
        count=1,
        summary_for=lambda _round, task: summary_for_exact_result_size(
            MAX_COORDINATOR_RESULT_BYTES + 1, task
        ),
    )
    with pytest.raises(CoordinationContextLimitExceeded):
        project_coordinator_input(
            standalone_query="Market outlook",
            intent="market_outlook",
            specialist_descriptors=scope,
            rounds=rounds,
            accepted_batches=accepted_batches,
        )

    def summaries_for_aggregate_size(target: int) -> tuple[str, ...]:
        views = tuple(
            PriorResultView(
                task_id=task_id_for(
                    request_id="request-1", round=1, dispatch_order=index
                ),
                summary="",
            )
            for index in range(8)
        )
        baseline = len(
            json.dumps(
                {
                    "prior_results": [view.model_dump(mode="json") for view in views],
                    "failed_tasks": [],
                    "data_gaps": [],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        quotient, remainder = divmod(target - baseline, len(views))
        return tuple("x" * (quotient + (index < remainder)) for index in range(8))

    def projection_with_aggregate_size(target: int) -> CoordinatorInput:
        summaries = summaries_for_aggregate_size(target)
        rounds, accepted_batches = _accepted_rounds(
            tasks_per_round=8,
            count=1,
            summary_for=lambda _round, task: summaries[task],
        )
        return project_coordinator_input(
            standalone_query="Market outlook",
            intent="market_outlook",
            specialist_descriptors=scope,
            rounds=rounds,
            accepted_batches=accepted_batches,
        )

    assert (
        len(
            projection_with_aggregate_size(MAX_COORDINATOR_CONTEXT_BYTES).prior_results
        )
        == 8
    )
    with pytest.raises(CoordinationContextLimitExceeded):
        projection_with_aggregate_size(MAX_COORDINATOR_CONTEXT_BYTES + 1)
