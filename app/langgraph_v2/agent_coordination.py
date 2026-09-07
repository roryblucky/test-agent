"""Bounded rolling Coordination Round acceptance and context projections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.langgraph_v2.agent_batch import (
    MAX_DISPATCH_BATCH_TASKS,
    AcceptedBatch,
    AcceptedTask,
    ActiveBatch,
    DispatchBatch,
    PriorResultView,
    SpecialistRegistry,
    TaskSucceeded,
    batch_id_for,
    normalize_task_objective,
    task_id_for,
    validate_active_batch_manifest,
)
from app.langgraph_v2.agent_evidence import DataGapView
from app.langgraph_v2.agent_scope import SpecialistDescriptor
from app.langgraph_v2.agent_termination import (
    COORDINATION_INVALID,
    COORDINATION_LIMIT,
    TASK_LIMIT,
    CoordinationStopReason,
)

MAX_COORDINATION_DECISIONS = 5
MAX_DISPATCH_ROUNDS = 4
MAX_ACCEPTED_TASKS = 32
MAX_TASK_CONTEXT_RESULTS = 8
AGENT_RECURSION_LIMIT = 40

StructuralStopReason = CoordinationStopReason


class CoordinationCandidateRejected(ValueError):
    """A Coordinator candidate that may consume the one same-round repair."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CoordinationInvariantError(ValueError):
    """A post-validation state change invalidated a frozen dispatch decision."""


class CoordinatorOutputInvalid(ValueError):
    """A model output that may consume the one same-round repair."""


class Finish(BaseModel):
    """Coordinator choice to end research without a business payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["finish"]


class PriorFailedTaskView(BaseModel):
    """Safe prior Task outcome that lets the Coordinator avoid repeating failure."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    objective: str


class CoordinatorInput(BaseModel):
    """The complete projection permitted to one Coordinator invocation."""

    model_config = ConfigDict(frozen=True)

    standalone_query: str
    intent: str
    specialist_descriptors: tuple[SpecialistDescriptor, ...]
    prior_results: tuple[PriorResultView, ...] = ()
    failed_tasks: tuple[PriorFailedTaskView, ...] = ()
    data_gaps: tuple[DataGapView, ...] = ()


CoordinatorDecision = Finish | DispatchBatch


class CoordinatorActor(Protocol):
    """Propose one bounded Coordinator decision."""

    async def decide(self, input: CoordinatorInput) -> CoordinatorDecision:
        """Return the next candidate decision from the frozen projection."""
        ...


class CoordinationTask(BaseModel):
    """Checkpointed manifest entry for one accepted dispatch decision."""

    model_config = ConfigDict(frozen=True)

    id: str
    objective: str
    specialist_id: str
    context_task_ids: tuple[str, ...] = Field(
        max_length=MAX_TASK_CONTEXT_RESULTS, default=()
    )


class CoordinationRound(BaseModel):
    """Immutable accepted Coordinator decision and monotonic revision."""

    model_config = ConfigDict(frozen=True)

    id: str
    revision: int = Field(ge=1, le=MAX_COORDINATION_DECISIONS)
    kind: Literal["dispatch", "finish"]
    batch_id: str | None = None
    tasks: tuple[CoordinationTask, ...] = Field(
        max_length=MAX_DISPATCH_BATCH_TASKS, default=()
    )

    @model_validator(mode="after")
    def _validate_decision_shape(self) -> CoordinationRound:
        """Keep persisted Decision shapes unambiguous and immutable."""
        if self.kind == "dispatch" and (self.batch_id is None or not self.tasks):
            raise ValueError("Dispatch Coordination Round is invalid")
        if self.kind == "finish" and (self.batch_id is not None or self.tasks):
            raise ValueError("Finish Coordination Round is invalid")
        return self


@dataclass(frozen=True)
class AcceptedCoordinationDispatch:
    """One accepted dispatch decision ready for the graph's single Send fanout."""

    round: CoordinationRound
    active_batch: ActiveBatch


@dataclass(frozen=True)
class CoordinationStopped:
    """An exhausted same-round repair that must end coordination incomplete."""

    reason: StructuralStopReason


def _round_id(*, request_id: str, revision: int) -> str:
    canonical = json.dumps(
        {"request_id": request_id, "revision": revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"round_{hashlib.sha256(canonical).hexdigest()[:32]}"


def _ordered_rounds(
    rounds: Sequence[CoordinationRound],
) -> tuple[CoordinationRound, ...]:
    """Return a contiguous immutable revision sequence or fail closed."""
    ordered = tuple(sorted(rounds, key=lambda round_: round_.revision))
    if tuple(round_.revision for round_ in ordered) != tuple(
        range(1, len(ordered) + 1)
    ):
        raise CoordinationInvariantError("Coordination Round revisions are invalid")
    finished_revisions = [
        round_.revision for round_ in ordered if round_.kind == "finish"
    ]
    if finished_revisions and finished_revisions != [ordered[-1].revision]:
        raise CoordinationInvariantError("Coordination Round sequence is invalid")
    if sum(round_.kind == "dispatch" for round_ in ordered) > MAX_DISPATCH_ROUNDS:
        raise CoordinationInvariantError("Coordination Round dispatch limit is invalid")
    return ordered


def validate_coordination_rounds(
    rounds: Sequence[CoordinationRound],
    *,
    request_id: str,
) -> tuple[CoordinationRound, ...]:
    """Verify persisted accepted Decisions form one legal request-owned sequence."""
    ordered_rounds = _ordered_rounds(rounds)
    for round_ in ordered_rounds:
        if round_.id != _round_id(request_id=request_id, revision=round_.revision):
            raise CoordinationInvariantError("Coordination Round identity is invalid")
        if round_.kind != "dispatch":
            continue
        if round_.batch_id != batch_id_for(
            request_id=request_id, round=round_.revision
        ):
            raise CoordinationInvariantError("Coordination Round batch is invalid")
        if not 1 <= len(round_.tasks) <= MAX_DISPATCH_BATCH_TASKS:
            raise CoordinationInvariantError("Coordination Round Task count is invalid")
        for dispatch_order, task in enumerate(round_.tasks):
            if task.id != task_id_for(
                request_id=request_id,
                round=round_.revision,
                dispatch_order=dispatch_order,
            ):
                raise CoordinationInvariantError(
                    "Coordination Round Task identity is invalid"
                )
            try:
                canonical_objective = normalize_task_objective(task.objective)
            except ValueError as error:
                raise CoordinationInvariantError(
                    "Coordination Round Task objective is invalid"
                ) from error
            if (
                task.objective != canonical_objective
                or not task.specialist_id
                or len(task.context_task_ids) > MAX_TASK_CONTEXT_RESULTS
                or len(set(task.context_task_ids)) != len(task.context_task_ids)
            ):
                raise CoordinationInvariantError("Coordination Round Task is invalid")
    return ordered_rounds


def _prior_projection(
    rounds: Sequence[CoordinationRound],
    accepted_batches: Mapping[str, AcceptedBatch],
) -> tuple[tuple[PriorResultView, ...], tuple[PriorFailedTaskView, ...]]:
    """Project stable successful Results and safe failed Task outcomes."""
    projected: list[PriorResultView] = []
    failed_tasks: list[PriorFailedTaskView] = []
    for round_ in _ordered_rounds(rounds):
        if round_.kind == "finish":
            continue
        if round_.batch_id is None:
            raise CoordinationInvariantError("Coordination Round batch is invalid")
        accepted = accepted_batches.get(round_.batch_id)
        if accepted is None:
            raise CoordinationInvariantError("Accepted Batch is missing")
        outcomes = {outcome.task_id: outcome for outcome in accepted.outcomes}
        if set(outcomes) != {task.id for task in round_.tasks}:
            raise CoordinationInvariantError("Accepted Batch manifest is invalid")
        for task in round_.tasks:
            outcome = outcomes[task.id]
            if not isinstance(outcome, TaskSucceeded):
                failed_tasks.append(
                    PriorFailedTaskView(task_id=task.id, objective=task.objective)
                )
                continue
            view = PriorResultView(
                task_id=task.id,
                summary=outcome.result.summary,
                evidence_ids=outcome.result.evidence_ids,
                data_gaps=tuple(gap.view() for gap in outcome.result.data_gaps),
            )
            projected.append(view)
    return tuple(projected), tuple(failed_tasks)


def _prior_results(
    rounds: Sequence[CoordinationRound],
    accepted_batches: Mapping[str, AcceptedBatch],
) -> tuple[PriorResultView, ...]:
    """Return successful Results for explicit Specialist context selection."""
    return _prior_projection(rounds, accepted_batches)[0]


def project_coordinator_input(
    *,
    standalone_query: str,
    intent: str,
    specialist_descriptors: tuple[SpecialistDescriptor, ...],
    rounds: Sequence[CoordinationRound],
    accepted_batches: Mapping[str, AcceptedBatch],
) -> CoordinatorInput:
    """Build the sole prompt-safe projection from complete accepted state."""
    prior_results, failed_tasks = _prior_projection(rounds, accepted_batches)
    data_gaps = tuple(gap for result in prior_results for gap in result.data_gaps)
    return CoordinatorInput(
        standalone_query=standalone_query,
        intent=intent,
        specialist_descriptors=specialist_descriptors,
        prior_results=prior_results,
        failed_tasks=failed_tasks,
        data_gaps=data_gaps,
    )


def _select_context(
    context_task_ids: tuple[str, ...],
    *,
    prior_results: Sequence[PriorResultView],
    error_type: type[CoordinationCandidateRejected | CoordinationInvariantError],
) -> tuple[PriorResultView, ...]:
    """Resolve one Task's explicit earlier successful Result references."""
    selected_ids = frozenset(context_task_ids)
    if len(context_task_ids) > MAX_TASK_CONTEXT_RESULTS or len(selected_ids) != len(
        context_task_ids
    ):
        raise error_type("Task context is invalid")
    by_task_id = {result.task_id: result for result in prior_results}
    if not selected_ids <= set(by_task_id):
        raise error_type("Task context is not an accepted prior success")
    selected = tuple(
        result for result in prior_results if result.task_id in selected_ids
    )
    return selected


def accept_coordination_dispatch(
    decision: DispatchBatch,
    *,
    request_id: str,
    rounds: Sequence[CoordinationRound],
    accepted_batches: Mapping[str, AcceptedBatch],
    registry: SpecialistRegistry,
    scope_descriptors: Sequence[SpecialistDescriptor],
) -> AcceptedCoordinationDispatch:
    """Validate one candidate Dispatch before it can create any graph Send."""
    ordered_rounds = validate_coordination_rounds(rounds, request_id=request_id)
    proposals = tuple(decision.tasks)
    if not 1 <= len(proposals) <= MAX_DISPATCH_BATCH_TASKS:
        raise CoordinationCandidateRejected("Task count is invalid")
    accepted_task_count = sum(
        len(round_.tasks) for round_ in ordered_rounds if round_.kind == "dispatch"
    )
    if accepted_task_count + len(proposals) > MAX_ACCEPTED_TASKS:
        raise CoordinationCandidateRejected("task_limit")
    dispatch_rounds = sum(round_.kind == "dispatch" for round_ in ordered_rounds)
    if dispatch_rounds >= MAX_DISPATCH_ROUNDS:
        raise CoordinationCandidateRejected("coordination_limit")
    prior_results = _prior_results(ordered_rounds, accepted_batches)
    revision = len(ordered_rounds) + 1
    active_tasks: list[AcceptedTask] = []
    for dispatch_order, proposal in enumerate(proposals):
        try:
            objective = normalize_task_objective(proposal.objective)
            registry.resolve(
                proposal.specialist_id, scope_descriptors=scope_descriptors
            )
            _select_context(
                proposal.context_task_ids,
                prior_results=prior_results,
                error_type=CoordinationCandidateRejected,
            )
        except ValueError as error:
            if isinstance(error, CoordinationCandidateRejected):
                raise
            raise CoordinationCandidateRejected(str(error)) from error
        active_tasks.append(
            AcceptedTask(
                id=task_id_for(
                    request_id=request_id,
                    round=revision,
                    dispatch_order=dispatch_order,
                ),
                objective=objective,
                specialist_id=proposal.specialist_id,
                context_task_ids=proposal.context_task_ids,
            )
        )
    active_batch = ActiveBatch(
        id=batch_id_for(request_id=request_id, round=revision),
        tasks=tuple(active_tasks),
        round=revision,
    )
    validate_active_batch_manifest(
        active_batch,
        request_id=request_id,
        registry=registry,
        scope_descriptors=scope_descriptors,
    )
    round_ = CoordinationRound(
        id=_round_id(request_id=request_id, revision=revision),
        revision=revision,
        kind="dispatch",
        batch_id=active_batch.id,
        tasks=tuple(
            CoordinationTask(
                id=task.id,
                objective=task.objective,
                specialist_id=task.specialist_id,
                context_task_ids=task.context_task_ids,
            )
            for task in active_batch.tasks
        ),
    )
    return AcceptedCoordinationDispatch(round=round_, active_batch=active_batch)


def accept_coordination_finish(
    *,
    request_id: str,
    rounds: Sequence[CoordinationRound],
) -> CoordinationRound:
    """Accept the terminal Finish decision with the next immutable revision."""
    ordered_rounds = validate_coordination_rounds(rounds, request_id=request_id)
    if len(ordered_rounds) >= MAX_COORDINATION_DECISIONS:
        raise CoordinationCandidateRejected("coordination_limit")
    if any(round_.kind == "finish" for round_ in ordered_rounds):
        raise CoordinationInvariantError("Coordination already finished")
    revision = len(ordered_rounds) + 1
    return CoordinationRound(
        id=_round_id(request_id=request_id, revision=revision),
        revision=revision,
        kind="finish",
    )


async def decide_coordination_round(
    actor: CoordinatorActor,
    input: CoordinatorInput,
    *,
    request_id: str,
    rounds: Sequence[CoordinationRound],
    accepted_batches: Mapping[str, AcceptedBatch],
    registry: SpecialistRegistry,
    scope_descriptors: Sequence[SpecialistDescriptor],
) -> AcceptedCoordinationDispatch | CoordinationRound | CoordinationStopped:
    """Own exactly one same-round repair without changing the frozen input."""
    rejection = CoordinationCandidateRejected("Coordinator decision is invalid")
    for attempt in range(2):
        try:
            if attempt == 0:
                candidate = await actor.decide(input)
            else:
                repair = cast(
                    Callable[..., Awaitable[CoordinatorDecision]] | None,
                    getattr(actor, "repair", None),
                )
                candidate = (
                    await repair(input, rejection=rejection.reason)
                    if repair is not None
                    else await actor.decide(input)
                )
            if isinstance(candidate, Finish):
                return accept_coordination_finish(
                    request_id=request_id,
                    rounds=rounds,
                )
            return accept_coordination_dispatch(
                DispatchBatch.model_validate(candidate),
                request_id=request_id,
                rounds=rounds,
                accepted_batches=accepted_batches,
                registry=registry,
                scope_descriptors=scope_descriptors,
            )
        except (CoordinatorOutputInvalid, ValidationError):
            rejection = CoordinationCandidateRejected("Coordinator decision is invalid")
        except CoordinationCandidateRejected as error:
            rejection = error
        if attempt == 1:
            return CoordinationStopped(reason=_stopped_reason(rejection.reason))
    raise AssertionError("Coordinator repair did not reach a terminal outcome")


def _stopped_reason(rejection: str) -> StructuralStopReason:
    """Map rejected model structure to a safe, deterministic terminal reason."""
    if rejection in {TASK_LIMIT, COORDINATION_LIMIT}:
        return cast(StructuralStopReason, rejection)
    return COORDINATION_INVALID


def validate_active_batch_coordination_round(
    batch: ActiveBatch,
    *,
    rounds: Sequence[CoordinationRound],
) -> None:
    """Ensure a recovered active manifest exactly matches its accepted Round."""
    ordered_rounds = _ordered_rounds(rounds)
    matching = [
        round_
        for round_ in ordered_rounds
        if round_.kind == "dispatch" and round_.batch_id == batch.id
    ]
    if (
        len(matching) != 1
        or matching[0].revision != batch.round
        or matching[0].revision != len(ordered_rounds)
    ):
        raise CoordinationInvariantError("Active Batch Coordination Round is invalid")
    manifest_tasks = tuple(
        CoordinationTask(
            id=task.id,
            objective=task.objective,
            specialist_id=task.specialist_id,
            context_task_ids=task.context_task_ids,
        )
        for task in batch.tasks
    )
    if matching[0].tasks != manifest_tasks:
        raise CoordinationInvariantError("Active Batch Coordination Round is invalid")


def materialize_specialist_context(
    task: AcceptedTask,
    *,
    current_round: int,
    rounds: Sequence[CoordinationRound],
    accepted_batches: Mapping[str, AcceptedBatch],
) -> tuple[PriorResultView, ...]:
    """Rebuild a previously accepted Task context from accepted prior results."""
    earlier_rounds = tuple(
        round_ for round_ in _ordered_rounds(rounds) if round_.revision < current_round
    )
    return _select_context(
        task.context_task_ids,
        prior_results=_prior_results(earlier_rounds, accepted_batches),
        error_type=CoordinationInvariantError,
    )
