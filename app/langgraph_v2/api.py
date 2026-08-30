"""Test-only FastAPI registration for the minimal v2 tracer."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import AnswerActor, build_answer_actor
from app.langgraph_v2.artifacts import ArtifactNotFound, ArtifactRepository
from app.langgraph_v2.authorization import (
    TrustedRequestContext,
    get_trusted_request_context,
)
from app.langgraph_v2.cancellation import CancellationObserver, CancellationRepository
from app.langgraph_v2.checkpointing import (
    FencedAsyncPostgresSaver,
    exact_checkpoint_config,
    initial_checkpoint_config,
)
from app.langgraph_v2.contracts import TracerStreamEvent, V2QueryRequest
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
    ConversationRecord,
    ResumeExpired,
    TurnNotFound,
    TurnRecord,
    turn_id_for_client_request,
)
from app.langgraph_v2.graph import TracerState, build_tracer_graph, tracer_graph
from app.langgraph_v2.groundedness import (
    GroundednessActor,
    UnavailableGroundednessActor,
    build_groundedness_actor,
)
from app.langgraph_v2.history import DEFAULT_HISTORY_TOKEN_BUDGET
from app.langgraph_v2.live_events import LiveEventWakeups
from app.langgraph_v2.output_assessments import (
    LoggingOutputAssessmentAudit,
    OutputAssessmentAudit,
)
from app.langgraph_v2.phase_results import PhaseExecutionContext, PhaseResultRepository
from app.langgraph_v2.pre_moderation import ModerationProvider
from app.langgraph_v2.provider_adapters import (
    MissingModeration,
    MissingRanker,
    MissingRetriever,
    V2ProviderBundle,
    adapt_tenant_providers,
)
from app.langgraph_v2.question_refinement import (
    QuestionRefinementActor,
    build_question_refinement_actor,
)
from app.langgraph_v2.reranking import Ranker
from app.langgraph_v2.retrieval import Retriever
from app.langgraph_v2.run_events import (
    CLAIM_HEARTBEAT_INTERVAL_SECONDS,
    CancellationObserved,
    ClaimFenced,
    EventInput,
    EventInvariantConflict,
    EventRecord,
    RunEventRepository,
    RunNotFound,
    RunRecord,
)
from app.langgraph_v2.stream import (
    GraphStreamCleanupError,
    RequestOwnedGraph,
    stream_graph,
)
from app.services.exceptions import TenantNotFoundError

_INSTANCE_ID = os.environ.get("LANGGRAPH_V2_INSTANCE_ID", socket.gethostname())
_LOGGER = logging.getLogger(__name__)


class CheckpointGraph(Protocol):
    """State lookup seam used by the Resume authorization path."""

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        """Read the current checkpoint state."""
        ...


def _resolve_refinement_actor(
    app: FastAPI,
    tenant_id: str,
    injected: QuestionRefinementActor | None,
) -> QuestionRefinementActor | None:
    """Resolve an injected actor or build one from the tenant model registry."""
    if injected is not None:
        return injected
    configured = getattr(app.state, "langgraph_v2_refinement_actor", None)
    if configured is not None:
        return configured
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        return None
    return build_question_refinement_actor(manager.get_model_registry(tenant_id))


def _resolve_provider_bundle(app: FastAPI, tenant_id: str) -> V2ProviderBundle | None:
    """Adapt the tenant's existing provider instances for v2 graph construction."""
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_providers"):
        return None
    return adapt_tenant_providers(manager.get_providers(tenant_id))


def _resolve_answer_actor(
    app: FastAPI,
    tenant_id: str,
    injected: AnswerActor | None,
) -> AnswerActor | None:
    """Resolve an injected actor or build the tenant's PydanticAI answer actor."""
    if injected is not None:
        return injected
    configured = getattr(app.state, "langgraph_v2_answer_actor", None)
    if configured is not None:
        return configured
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        return None
    return build_answer_actor(manager.get_model_registry(tenant_id))


def _resolve_cancellation_check(
    app: FastAPI,
    tenant_id: str,
    run_id: uuid.UUID,
    observer: CancellationObserver,
) -> Any:
    """Combine the authoritative observer with an optional test seam."""
    checker = getattr(app.state, "langgraph_v2_cancellation_check", None)
    if checker is None:
        return observer.is_requested

    async def check() -> bool:
        return bool(await checker(tenant_id, run_id)) or await observer.is_requested()

    return check


def _resolve_groundedness_actor(
    app: FastAPI,
    tenant_id: str,
    injected: GroundednessActor | None,
) -> GroundednessActor | None:
    """Resolve an injected or tenant-registry groundedness evaluator."""
    if injected is not None:
        return injected
    configured = getattr(app.state, "langgraph_v2_groundedness_actor", None)
    if configured is not None:
        return configured
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        return None
    return build_groundedness_actor(manager.get_model_registry(tenant_id))


def _resolve_groundedness_actor_safely(
    app: FastAPI,
    tenant_id: str,
    injected: GroundednessActor | None,
) -> GroundednessActor | None:
    """Keep groundedness setup failures inside the advisory phase."""
    try:
        return _resolve_groundedness_actor(app, tenant_id, injected)
    except Exception as exc:
        return UnavailableGroundednessActor(exc)


def _resolve_output_assessment_audit(
    app: FastAPI,
    injected: OutputAssessmentAudit | None,
) -> OutputAssessmentAudit:
    """Resolve the optional audit port, defaulting to the logging POC adapter."""
    if injected is not None:
        return injected
    configured = getattr(app.state, "langgraph_v2_output_assessment_audit", None)
    return configured or LoggingOutputAssessmentAudit()


def _ensure_tenant_available(app: FastAPI, tenant_id: str) -> None:
    """Validate a configured tenant before creating a running v2 Run."""
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None:
        return
    try:
        if hasattr(manager, "get_providers"):
            manager.get_providers(tenant_id)
        elif hasattr(manager, "get_model_registry"):
            manager.get_model_registry(tenant_id)
    except TenantNotFoundError as error:
        raise HTTPException(status_code=404, detail="Tenant not found") from error


def _resolve_phase_providers(
    app: FastAPI,
    tenant_id: str,
    *,
    retriever: Retriever | None,
    ranker: Ranker | None,
    moderation_provider: ModerationProvider | None,
) -> tuple[Retriever | None, Ranker | None, ModerationProvider | None]:
    """Resolve injected, app-level, and tenant-scoped v2 providers once."""
    configured_retriever = retriever or getattr(
        app.state, "langgraph_v2_retriever", None
    )
    configured_ranker = ranker or getattr(app.state, "langgraph_v2_ranker", None)
    configured_moderation = moderation_provider or getattr(
        app.state, "langgraph_v2_moderation_provider", None
    )
    provider_bundle = _resolve_provider_bundle(app, tenant_id)
    if provider_bundle is not None:
        configured_retriever = (
            configured_retriever or provider_bundle.retriever or MissingRetriever()
        )
        configured_ranker = (
            configured_ranker or provider_bundle.ranker or MissingRanker()
        )
        configured_moderation = (
            configured_moderation or provider_bundle.moderation or MissingModeration()
        )
    return configured_retriever, configured_ranker, configured_moderation


async def _refresh_claim(
    repository: RunEventRepository,
    tenant_id: str,
    run_id: uuid.UUID,
    owner_instance_id: str,
    execution_epoch: int,
) -> None:
    """Refresh a request's claim until it is cancelled or fenced."""
    while True:
        await asyncio.sleep(CLAIM_HEARTBEAT_INTERVAL_SECONDS)
        try:
            await repository.heartbeat(
                tenant_id=tenant_id,
                run_id=run_id,
                owner_instance_id=owner_instance_id,
                execution_epoch=execution_epoch,
            )
        except ClaimFenced:
            return


async def _persist_event_record(
    repository: RunEventRepository,
    message_repository: ConversationMessageRepository,
    event: TracerStreamEvent,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    conversation_id: str,
    turn_id: uuid.UUID | str | None = None,
    owner_instance_id: str,
    execution_epoch: int,
) -> EventRecord:
    """Persist one graph Event, atomically publishing an assistant answer."""
    event_input = EventInput(
        event_key=event.event_key,
        type=event.type,
        step=event.step,
        data=event.data,
    )
    raw_event_data: object = event.data
    event_data = (
        cast(dict[str, Any], raw_event_data)
        if isinstance(raw_event_data, dict)
        else None
    )
    answer = event_data.get("answer") if event_data is not None else None
    if event.type == "done" and isinstance(answer, str):
        try:
            resolved_turn_id = uuid.UUID(str(turn_id))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "completed Graph state is missing a valid turn_id"
            ) from error
        conflict: EventInvariantConflict | None = None
        persisted: EventRecord | None = None
        async with repository.transaction() as connection:
            try:
                async with connection.transaction():
                    await message_repository.persist_assistant_message_in_terminal_transaction(
                        connection,
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        run_id=run_id,
                        owner_instance_id=owner_instance_id,
                        execution_epoch=execution_epoch,
                        turn_id=resolved_turn_id,
                        content=answer,
                        idempotency_key=f"turn:{resolved_turn_id}:assistant",
                    )
                    persisted = await repository.complete_run_in_transaction(
                        connection,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        event=event_input,
                        owner_instance_id=owner_instance_id,
                        execution_epoch=execution_epoch,
                    )
            except EventInvariantConflict as error:
                await repository.mark_event_conflict_in_transaction(
                    connection,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    event_key=event.event_key,
                )
                conflict = error
        if conflict is not None:
            raise conflict
        if persisted is None:
            raise RuntimeError("terminal event transaction produced no event record")
        await repository.publish_wakeup(tenant_id, run_id)
        return persisted

    if event.type == "done":
        return await repository.complete_run(
            tenant_id=tenant_id,
            run_id=run_id,
            event=event_input,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
        )
    if event.type == "error":
        return await repository.fail_run(
            tenant_id=tenant_id,
            run_id=run_id,
            event=event_input,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
        )
    return await repository.append_event(
        tenant_id=tenant_id,
        run_id=run_id,
        event=event_input,
        owner_instance_id=owner_instance_id,
        execution_epoch=execution_epoch,
    )


async def _terminalize_checkpoint_event(
    repository: RunEventRepository,
    message_repository: ConversationMessageRepository,
    event: TracerStreamEvent,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    conversation_id: str,
    turn_id: uuid.UUID,
    owner_instance_id: str,
    execution_epoch: int,
) -> None:
    """Publish a checkpoint-owned terminal outcome without an Event journal."""
    if event.type == "error":
        await repository.fail_run_without_event(
            tenant_id=tenant_id,
            run_id=run_id,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
            error=event.data,
        )
        return
    if event.type != "done":
        raise ValueError("checkpoint terminalization requires done or error")

    raw_data: object = event.data
    data = cast(dict[str, Any], raw_data) if isinstance(raw_data, dict) else {}
    answer = data.get("answer")
    async with repository.transaction() as connection:
        if isinstance(answer, str):
            await message_repository.persist_assistant_message_in_terminal_transaction(
                connection,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                run_id=run_id,
                owner_instance_id=owner_instance_id,
                execution_epoch=execution_epoch,
                turn_id=turn_id,
                content=answer,
                idempotency_key=f"turn:{turn_id}:assistant",
            )
        await repository.complete_run_without_event_in_transaction(
            connection,
            tenant_id=tenant_id,
            run_id=run_id,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
            outcome=event.data,
        )
    await repository.publish_wakeup(tenant_id, run_id)


async def _cleanup_request_execution(
    graph_stream: AsyncGenerator[str] | None,
    cancellation_observer: CancellationObserver,
    heartbeat_task: asyncio.Task[None] | None,
    repository: RunEventRepository,
    *,
    claim: RunRecord | None,
    terminal: bool,
    primary_error: BaseException | None,
    tenant_id: str,
    run_id: uuid.UUID,
) -> None:
    """Close request-owned work before releasing an unfinished transitional Run."""
    cleanup_error: BaseException | None = None
    cleanup_cancelled = False

    if graph_stream is not None:
        cancelled, error = await _await_cleanup_operation(graph_stream.aclose())
        cleanup_cancelled |= cancelled
        cleanup_error = error

    cancelled, error = await _await_cleanup_operation(cancellation_observer.close())
    cleanup_cancelled |= cancelled
    if cleanup_error is None:
        cleanup_error = error

    if heartbeat_task is not None:
        heartbeat_task.cancel()
        _, heartbeat_error = await _await_cleanup_task(heartbeat_task)
        if cleanup_error is None and not isinstance(
            heartbeat_error, asyncio.CancelledError
        ):
            cleanup_error = heartbeat_error

    if claim is not None and not terminal:
        cancelled, error = await _await_cleanup_operation(
            repository.interrupt_run(
                tenant_id=tenant_id,
                run_id=run_id,
                owner_instance_id=claim.owner_instance_id,
                execution_epoch=claim.execution_epoch,
            )
        )
        cleanup_cancelled |= cancelled
        if isinstance(error, (ClaimFenced, RunNotFound)):
            error = None
        if cleanup_error is None:
            cleanup_error = error

    if cleanup_error is not None and (primary_error is not None or cleanup_cancelled):
        _LOGGER.warning(
            "Request-owned cleanup failed after a primary exception",
            exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
        )
    if primary_error is not None:
        return
    if cleanup_cancelled:
        raise asyncio.CancelledError
    if cleanup_error is not None:
        raise cleanup_error


async def _await_cleanup_task(
    task: asyncio.Task[Any],
) -> tuple[bool, BaseException | None]:
    """Await a cleanup task to completion despite repeated cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    try:
        await task
    except asyncio.CancelledError as error:
        return cancelled, error
    except BaseException as error:
        return cancelled, error
    return cancelled, None


async def _await_cleanup_operation(
    operation: Awaitable[Any],
) -> tuple[bool, BaseException | None]:
    """Own and await one cleanup operation until it reaches a terminal state."""
    return await _await_cleanup_task(asyncio.ensure_future(operation))


async def persist_result_events(  # pyright: ignore[reportUnusedFunction] -- test seam
    repository: RunEventRepository,
    message_repository: ConversationMessageRepository,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    conversation_id: str,
    result: dict[str, Any],
    expected_turn_id: uuid.UUID | None = None,
    owner_instance_id: str,
    execution_epoch: int,
    suppress_sse_for_event_keys: set[str] | None = None,
) -> AsyncIterator[str]:
    """Persist graph Events and serialize newly published SSE frames."""
    prior_keys = suppress_sse_for_event_keys or set()
    for raw_event in result["events"]:
        event = TracerStreamEvent.model_validate(raw_event)
        result_turn_id = result.get("turn_id", expected_turn_id)
        if expected_turn_id is not None and result_turn_id is not None:
            try:
                resolved_result_turn_id = uuid.UUID(str(result_turn_id))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "completed Graph state is missing a valid turn_id"
                ) from error
            if resolved_result_turn_id != expected_turn_id:
                raise ValueError("completed Graph state turn_id does not match Run")
        persisted = await _persist_event_record(
            repository,
            message_repository,
            event,
            tenant_id=tenant_id,
            run_id=run_id,
            conversation_id=conversation_id,
            turn_id=result_turn_id,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
        )
        if event.event_key not in prior_keys:
            yield event.model_copy(update={"sequence": persisted.sequence}).to_sse()


async def _persist_setup_failure(
    repository: RunEventRepository,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    owner_instance_id: str,
    execution_epoch: int,
    message: str,
) -> str:
    """Terminalize a Run when role actor construction fails before graph start."""
    event = await repository.fail_run(
        tenant_id=tenant_id,
        run_id=run_id,
        event=EventInput(
            event_key="lifecycle:groundedness_setup:error:0",
            type="error",
            data=message,
        ),
        owner_instance_id=owner_instance_id,
        execution_epoch=execution_epoch,
    )
    return TracerStreamEvent(
        event_key=event.event_key,
        type="error",
        data=message,
        sequence=event.sequence,
    ).to_sse()


type TracerGraph = RequestOwnedGraph

_RESUMABLE_NODES = frozenset(
    {
        "pre_moderation",
        "question_refinement",
        "retrieval",
        "reranking",
        "answer",
        "groundedness",
        "post_moderation",
        "finalization",
    }
)


class ThreadResumeConflict(RuntimeError):
    """A thread checkpoint cannot be recovered by the pre-Answer Resume route."""


@dataclass(frozen=True)
class ThreadResumeTarget:
    """Authorized checkpoint target for a request-owned thread Resume."""

    conversation: ConversationRecord
    turn: TurnRecord
    config: RunnableConfig


def _live_events(app: FastAPI) -> LiveEventWakeups:
    """Return the app-local wakeup relay used by durable Event followers."""
    wakeups = getattr(app.state, "langgraph_v2_live_events", None)
    if wakeups is None:
        wakeups = LiveEventWakeups(instance_id=_INSTANCE_ID)
        app.state.langgraph_v2_live_events = wakeups
    return cast(LiveEventWakeups, wakeups)


def _message_repository(
    app: FastAPI, pool: AsyncConnectionPool[Any]
) -> ConversationMessageRepository:
    """Build the Message repository with the deployment's fixed Resume TTL."""
    return ConversationMessageRepository(
        pool,
        resume_ttl=timedelta(
            seconds=getattr(app.state, "langgraph_v2_resume_ttl_seconds", 3600)
        ),
    )


def _checkpoint_turn_id(values: dict[str, Any]) -> uuid.UUID:
    """Extract the Turn bound into the latest checkpoint state."""
    try:
        return uuid.UUID(str(values["turn_id"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ThreadResumeConflict("checkpoint is missing a valid turn_id") from error


async def _authorize_thread_resume_target(
    *,
    checkpoint_graph: CheckpointGraph,
    message_repository: ConversationMessageRepository,
    context: TrustedRequestContext,
    thread_id: str,
    expected_turn_id: uuid.UUID,
) -> ThreadResumeTarget:
    """Authorize one thread Resume and reject non-recoverable checkpoints."""
    conversation = await message_repository.get_conversation_by_thread(
        context=context,
        thread_id=thread_id,
    )
    turn = await message_repository.get_turn_for_resume(
        context=context,
        conversation_id=conversation.conversation_id,
        turn_id=expected_turn_id,
    )
    config = initial_checkpoint_config(
        thread_id=conversation.thread_id, checkpoint_ns=""
    )
    try:
        snapshot = await checkpoint_graph.aget_state(config)
    except ValueError as error:
        raise ThreadResumeConflict(str(error)) from error
    if snapshot.metadata is None:
        raise ConversationNotFound(thread_id)
    if not snapshot.next:
        raise ThreadResumeConflict("checkpoint is already complete")
    if any(node not in _RESUMABLE_NODES for node in snapshot.next):
        raise ThreadResumeConflict("checkpoint is not resumable")
    raw_snapshot_values: object = snapshot.values
    if not isinstance(raw_snapshot_values, dict):
        raise ThreadResumeConflict("checkpoint state is not a mapping")
    snapshot_values = cast(dict[str, Any], raw_snapshot_values)
    if snapshot_values.get("conversation_id") != conversation.conversation_id:
        raise ThreadResumeConflict("checkpoint belongs to another Conversation")

    turn_id = _checkpoint_turn_id(snapshot_values)
    if turn_id != expected_turn_id:
        raise ThreadResumeConflict("checkpoint does not match the expected Turn")
    latest_turn = await message_repository.get_latest_turn(
        context=context,
        conversation_id=conversation.conversation_id,
    )
    if latest_turn.turn_id != turn.turn_id:
        raise ThreadResumeConflict("checkpoint Turn has been superseded")
    checkpoint_configurable = snapshot.config.get("configurable", {})
    checkpoint_id = checkpoint_configurable.get("checkpoint_id")
    checkpoint_ns = checkpoint_configurable.get("checkpoint_ns")
    if not isinstance(checkpoint_id, str) or not isinstance(checkpoint_ns, str):
        raise ThreadResumeConflict("checkpoint is missing a valid checkpoint config")
    return ThreadResumeTarget(
        conversation=conversation,
        turn=turn,
        config=exact_checkpoint_config(
            thread_id=conversation.thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
        ),
    )


def create_tracer_router(
    graph: TracerGraph | None = None,
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    moderation_provider: ModerationProvider | None = None,
    answer_actor: AnswerActor | None = None,
    groundedness_actor: GroundednessActor | None = None,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    output_assessment_audit: OutputAssessmentAudit | None = None,
) -> APIRouter:
    """Create the test-only router around an injected request-owned stream seam."""
    if history_token_budget < 0:
        raise ValueError("history_token_budget must not be negative")
    router = APIRouter(tags=["LangGraph v2 tracer"])

    @router.post("/v2/query/stream")
    async def query_stream(  # pyright: ignore[reportUnusedFunction] -- FastAPI route
        payload: V2QueryRequest,
        http_request: Request,
        request_context: Annotated[
            TrustedRequestContext, Depends(get_trusted_request_context)
        ],
        x_user_groups: Annotated[str, Header(alias="X-User-Groups")] = "",
    ) -> StreamingResponse:
        """Run the deterministic tracer and return its events as SSE."""
        del x_user_groups
        _ensure_tenant_available(http_request.app, request_context.tenant_id)
        run_id = uuid.uuid4()
        x_application_id = request_context.tenant_id
        configured_pool = getattr(
            http_request.app.state,
            "langgraph_v2_postgres_pool",
            None,
        )
        if configured_pool is None:
            raise HTTPException(
                status_code=500, detail="LangGraph v2 PostgreSQL is not configured"
            )
        pool = cast(AsyncConnectionPool[Any], configured_pool)
        wakeups = _live_events(http_request.app)
        repository = RunEventRepository(pool, live_events=wakeups)
        cancellation_repository = CancellationRepository(pool, wakeups=wakeups)
        cancellation_observer = CancellationObserver(
            cancellation_repository,
            wakeups,
            tenant_id=x_application_id,
            run_id=run_id,
        )
        message_repository = _message_repository(http_request.app, pool)
        try:
            if payload.conversation_id is None:
                conversation = await message_repository.resolve_conversation(
                    context=request_context,
                )
            else:
                conversation = await message_repository.get_conversation(
                    context=request_context,
                    conversation_id=payload.conversation_id,
                )
        except ConversationNotFound as error:
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            ) from error
        conversation_id = conversation.conversation_id
        if payload.client_request_id is None:
            turn_id = uuid.uuid4()
        else:
            turn_id = turn_id_for_client_request(
                x_application_id, conversation_id, payload.client_request_id
            )
        user_idempotency_key = f"turn:{turn_id}:user"

        async def event_generator() -> AsyncIterator[str]:
            claim = None
            graph_stream: AsyncGenerator[str] | None = None
            heartbeat_task: asyncio.Task[None] | None = None
            terminal = False
            primary_error: BaseException | None = None
            try:
                claim = await repository.create_run(
                    tenant_id=x_application_id,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    owner_instance_id=_INSTANCE_ID,
                )
                await message_repository.create_turn(
                    context=request_context,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    content=payload.query,
                    idempotency_key=user_idempotency_key,
                )
                await cancellation_observer.start()
                heartbeat_task = asyncio.create_task(
                    _refresh_claim(
                        repository,
                        x_application_id,
                        run_id,
                        claim.owner_instance_id,
                        claim.execution_epoch,
                    )
                )
                configured_checkpointer = getattr(
                    http_request.app.state,
                    "langgraph_v2_checkpointer",
                    None,
                )
                configured_refinement_actor = _resolve_refinement_actor(
                    http_request.app,
                    x_application_id,
                    refinement_actor,
                )
                configured_answer_actor = _resolve_answer_actor(
                    http_request.app, x_application_id, answer_actor
                )
                configured_groundedness_actor = _resolve_groundedness_actor_safely(
                    http_request.app, x_application_id, groundedness_actor
                )
                configured_output_assessment_audit = _resolve_output_assessment_audit(
                    http_request.app, output_assessment_audit
                )
                (
                    configured_retriever,
                    configured_ranker,
                    configured_moderation,
                ) = _resolve_phase_providers(
                    http_request.app,
                    x_application_id,
                    retriever=retriever,
                    ranker=ranker,
                    moderation_provider=moderation_provider,
                )
                selected_graph = graph or tracer_graph
                graph_config: RunnableConfig | None = None
                if graph is None:
                    phase_context = PhaseExecutionContext(
                        repository=PhaseResultRepository(pool, live_events=wakeups),
                        artifact_repository=ArtifactRepository(pool),
                        message_repository=message_repository,
                        request_context=request_context,
                        history_token_budget=history_token_budget,
                        current_turn_id=turn_id,
                        tenant_id=x_application_id,
                        run_id=run_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                        cancellation_check=_resolve_cancellation_check(
                            http_request.app,
                            x_application_id,
                            run_id,
                            cancellation_observer,
                        ),
                        output_assessment_audit=configured_output_assessment_audit,
                    )
                    saver: FencedAsyncPostgresSaver | None = None
                    if configured_checkpointer is not None:

                        async def write_checkpoint_pointer(
                            checkpoint_id: str,
                            checkpoint_ns: str,
                        ) -> None:
                            await repository.update_checkpoint_pointer(
                                tenant_id=x_application_id,
                                run_id=run_id,
                                owner_instance_id=claim.owner_instance_id,
                                execution_epoch=claim.execution_epoch,
                                checkpoint_id=checkpoint_id,
                                checkpoint_ns=checkpoint_ns,
                            )

                        checkpoint_ns = ""
                        saver = FencedAsyncPostgresSaver(
                            pool,
                            checkpoint_namespace=checkpoint_ns,
                            pointer_writer=write_checkpoint_pointer,
                        )
                        graph_config = initial_checkpoint_config(
                            thread_id=conversation.thread_id,
                            checkpoint_ns=checkpoint_ns,
                        )
                    selected_graph = build_tracer_graph(
                        saver,
                        phase_context=phase_context,
                        refinement_actor=configured_refinement_actor,
                        retriever=configured_retriever,
                        ranker=configured_ranker,
                        moderation_provider=configured_moderation,
                        answer_actor=configured_answer_actor,
                        groundedness_actor=configured_groundedness_actor,
                    )
                state: TracerState = {
                    "query": payload.query,
                    "conversation_id": conversation_id,
                    "turn_id": str(turn_id),
                    "client_request_id": payload.client_request_id,
                    "events": [],
                }

                async def persist_event(event: TracerStreamEvent) -> None:
                    nonlocal terminal
                    await _persist_event_record(
                        repository,
                        message_repository,
                        event,
                        tenant_id=x_application_id,
                        run_id=run_id,
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                    )
                    if event.type in {"done", "error"}:
                        terminal = True

                async def terminalize_checkpoint_event(
                    event: TracerStreamEvent,
                ) -> None:
                    nonlocal terminal
                    await _terminalize_checkpoint_event(
                        repository,
                        message_repository,
                        event,
                        tenant_id=x_application_id,
                        run_id=run_id,
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                    )
                    terminal = True

                graph_stream = stream_graph(
                    cast(RequestOwnedGraph, selected_graph),
                    state,
                    config=graph_config,
                    event_sink=persist_event,
                    terminal_sink=terminalize_checkpoint_event,
                )
                async for frame in graph_stream:
                    yield frame
            except CancellationObserved:
                if claim is not None:
                    stopped = await cancellation_repository.apply_if_requested(
                        tenant_id=x_application_id,
                        run_id=run_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                    )
                    if stopped is not None:
                        terminal = True
                        yield TracerStreamEvent(
                            event_key=stopped.event_key,
                            type="stopped",
                            sequence=stopped.sequence,
                            data=stopped.data,
                        ).to_sse()
            except asyncio.CancelledError as error:
                primary_error = error
                raise
            except GraphStreamCleanupError as error:
                primary_error = error
                raise
            except Exception as error:
                if claim is None:
                    raise
                if not terminal:
                    with suppress(ClaimFenced, RunNotFound):
                        failure = await _persist_setup_failure(
                            repository,
                            tenant_id=x_application_id,
                            run_id=run_id,
                            owner_instance_id=claim.owner_instance_id,
                            execution_epoch=claim.execution_epoch,
                            message=str(error) or "LangGraph execution failed.",
                        )
                        terminal = True
                        yield failure
            finally:
                await _cleanup_request_execution(
                    graph_stream,
                    cancellation_observer,
                    heartbeat_task,
                    repository,
                    claim=claim,
                    terminal=terminal,
                    primary_error=primary_error,
                    tenant_id=x_application_id,
                    run_id=run_id,
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Run-Id": str(run_id),
                "X-Conversation-Id": conversation_id,
                "X-Turn-Id": str(turn_id),
                "X-Thread-Id": conversation.thread_id,
            },
        )

    @router.post("/v2/threads/{thread_id}/resume/stream")
    async def thread_resume_stream(  # pyright: ignore[reportUnusedFunction] -- FastAPI route
        thread_id: str,
        http_request: Request,
        request_context: Annotated[
            TrustedRequestContext, Depends(get_trusted_request_context)
        ],
        expected_turn_id: Annotated[uuid.UUID, Query(alias="expectedTurnId")],
    ) -> StreamingResponse:
        """Recover an authorized Conversation thread from its latest checkpoint."""
        x_application_id = request_context.tenant_id
        _ensure_tenant_available(http_request.app, x_application_id)
        configured_pool = getattr(
            http_request.app.state,
            "langgraph_v2_postgres_pool",
            None,
        )
        if configured_pool is None:
            raise HTTPException(
                status_code=500, detail="LangGraph v2 PostgreSQL is not configured"
            )
        configured_checkpointer = getattr(
            http_request.app.state,
            "langgraph_v2_checkpointer",
            None,
        )
        if configured_checkpointer is None:
            raise HTTPException(
                status_code=500, detail="LangGraph v2 checkpointer is not configured"
            )
        pool = cast(AsyncConnectionPool[Any], configured_pool)
        wakeups = _live_events(http_request.app)
        repository = RunEventRepository(pool, live_events=wakeups)
        cancellation_repository = CancellationRepository(pool, wakeups=wakeups)
        message_repository = _message_repository(http_request.app, pool)
        configured_refinement_actor = _resolve_refinement_actor(
            http_request.app,
            x_application_id,
            refinement_actor,
        )
        configured_answer_actor = _resolve_answer_actor(
            http_request.app, x_application_id, answer_actor
        )
        configured_groundedness_actor = _resolve_groundedness_actor_safely(
            http_request.app, x_application_id, groundedness_actor
        )
        configured_output_assessment_audit = _resolve_output_assessment_audit(
            http_request.app, output_assessment_audit
        )
        (
            configured_retriever,
            configured_ranker,
            configured_moderation,
        ) = _resolve_phase_providers(
            http_request.app,
            x_application_id,
            retriever=retriever,
            ranker=ranker,
            moderation_provider=moderation_provider,
        )
        checkpoint_graph = build_tracer_graph(
            configured_checkpointer,
            phase_context=PhaseExecutionContext(
                repository=PhaseResultRepository(pool, live_events=wakeups),
                artifact_repository=ArtifactRepository(pool),
                message_repository=message_repository,
                request_context=request_context,
                history_token_budget=history_token_budget,
                tenant_id=x_application_id,
                run_id=uuid.UUID(int=0),
                owner_instance_id="",
                execution_epoch=0,
                output_assessment_audit=configured_output_assessment_audit,
            ),
            refinement_actor=configured_refinement_actor,
            retriever=configured_retriever,
            ranker=configured_ranker,
            moderation_provider=configured_moderation,
            answer_actor=configured_answer_actor,
            groundedness_actor=configured_groundedness_actor,
        )
        try:
            target = await _authorize_thread_resume_target(
                checkpoint_graph=checkpoint_graph,
                message_repository=message_repository,
                context=request_context,
                thread_id=thread_id,
                expected_turn_id=expected_turn_id,
            )
        except ConversationNotFound as error:
            raise HTTPException(status_code=404, detail="Thread not found") from error
        except ResumeExpired as error:
            raise HTTPException(status_code=410, detail="Turn expired") from error
        except TurnNotFound as error:
            raise HTTPException(status_code=404, detail="Turn not found") from error
        except ThreadResumeConflict as error:
            raise HTTPException(
                status_code=409, detail="Thread is not resumable"
            ) from error

        run_id = uuid.uuid4()
        cancellation_observer = CancellationObserver(
            cancellation_repository,
            wakeups,
            tenant_id=x_application_id,
            run_id=run_id,
        )

        async def event_generator() -> AsyncIterator[str]:
            claim = None
            graph_stream: AsyncGenerator[str] | None = None
            heartbeat_task: asyncio.Task[None] | None = None
            terminal = False
            primary_error: BaseException | None = None
            try:
                claim = await repository.create_run(
                    tenant_id=x_application_id,
                    run_id=run_id,
                    conversation_id=target.conversation.conversation_id,
                    owner_instance_id=_INSTANCE_ID,
                )
                await message_repository.associate_run_with_turn(
                    context=request_context,
                    conversation_id=target.conversation.conversation_id,
                    run_id=run_id,
                    owner_instance_id=claim.owner_instance_id,
                    execution_epoch=claim.execution_epoch,
                    turn_id=target.turn.turn_id,
                )
                await cancellation_observer.start()
                heartbeat_task = asyncio.create_task(
                    _refresh_claim(
                        repository,
                        x_application_id,
                        run_id,
                        claim.owner_instance_id,
                        claim.execution_epoch,
                    )
                )

                async def write_checkpoint_pointer(
                    checkpoint_id: str,
                    checkpoint_ns: str,
                ) -> None:
                    await repository.update_checkpoint_pointer(
                        tenant_id=x_application_id,
                        run_id=run_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                        checkpoint_id=checkpoint_id,
                        checkpoint_ns=checkpoint_ns,
                    )

                selected_graph = graph or build_tracer_graph(
                    FencedAsyncPostgresSaver(
                        pool,
                        checkpoint_namespace="",
                        pointer_writer=write_checkpoint_pointer,
                    ),
                    phase_context=PhaseExecutionContext(
                        repository=PhaseResultRepository(pool, live_events=wakeups),
                        artifact_repository=ArtifactRepository(pool),
                        message_repository=message_repository,
                        request_context=request_context,
                        history_token_budget=history_token_budget,
                        current_turn_id=target.turn.turn_id,
                        tenant_id=x_application_id,
                        run_id=run_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                        cancellation_check=_resolve_cancellation_check(
                            http_request.app,
                            x_application_id,
                            run_id,
                            cancellation_observer,
                        ),
                        output_assessment_audit=configured_output_assessment_audit,
                    ),
                    refinement_actor=configured_refinement_actor,
                    retriever=configured_retriever,
                    ranker=configured_ranker,
                    moderation_provider=configured_moderation,
                    answer_actor=configured_answer_actor,
                    groundedness_actor=configured_groundedness_actor,
                )

                async def persist_event(event: TracerStreamEvent) -> None:
                    nonlocal terminal
                    await _persist_event_record(
                        repository,
                        message_repository,
                        event,
                        tenant_id=x_application_id,
                        run_id=run_id,
                        conversation_id=target.conversation.conversation_id,
                        turn_id=target.turn.turn_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                    )
                    if event.type in {"done", "error"}:
                        terminal = True

                async def terminalize_checkpoint_event(
                    event: TracerStreamEvent,
                ) -> None:
                    nonlocal terminal
                    await _terminalize_checkpoint_event(
                        repository,
                        message_repository,
                        event,
                        tenant_id=x_application_id,
                        run_id=run_id,
                        conversation_id=target.conversation.conversation_id,
                        turn_id=target.turn.turn_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                    )
                    terminal = True

                graph_stream = stream_graph(
                    cast(RequestOwnedGraph, selected_graph),
                    None,
                    config=target.config,
                    event_sink=persist_event,
                    terminal_sink=terminalize_checkpoint_event,
                )
                async for frame in graph_stream:
                    yield frame
            except CancellationObserved:
                if claim is not None:
                    stopped = await cancellation_repository.apply_if_requested(
                        tenant_id=x_application_id,
                        run_id=run_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                    )
                    if stopped is not None:
                        terminal = True
                        yield TracerStreamEvent(
                            event_key=stopped.event_key,
                            type="stopped",
                            sequence=stopped.sequence,
                            data=stopped.data,
                        ).to_sse()
            except asyncio.CancelledError as error:
                primary_error = error
                raise
            except GraphStreamCleanupError as error:
                primary_error = error
                raise
            except Exception as error:
                if claim is None:
                    raise
                if not terminal:
                    with suppress(ClaimFenced, RunNotFound):
                        failure = await _persist_setup_failure(
                            repository,
                            tenant_id=x_application_id,
                            run_id=run_id,
                            owner_instance_id=claim.owner_instance_id,
                            execution_epoch=claim.execution_epoch,
                            message=str(error) or "LangGraph execution failed.",
                        )
                        terminal = True
                        yield failure
            finally:
                await _cleanup_request_execution(
                    graph_stream,
                    cancellation_observer,
                    heartbeat_task,
                    repository,
                    claim=claim,
                    terminal=terminal,
                    primary_error=primary_error,
                    tenant_id=x_application_id,
                    run_id=run_id,
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Conversation-Id": target.conversation.conversation_id,
                "X-Turn-Id": str(target.turn.turn_id),
                "X-Thread-Id": target.conversation.thread_id,
            },
        )

    @router.get("/v2/artifacts/{artifact_id}")
    async def get_artifact(  # pyright: ignore[reportUnusedFunction] -- FastAPI route
        artifact_id: uuid.UUID,
        http_request: Request,
        x_application_id: Annotated[str, Header(alias="X-Application-Id")],
    ) -> dict[str, Any]:
        """Read one Artifact through the caller's Tenant boundary."""
        configured_pool = getattr(
            http_request.app.state, "langgraph_v2_postgres_pool", None
        )
        if configured_pool is None:
            raise HTTPException(
                status_code=500, detail="LangGraph v2 PostgreSQL is not configured"
            )
        try:
            artifact = await ArtifactRepository(configured_pool).get(
                tenant_id=x_application_id, artifact_id=artifact_id
            )
        except ArtifactNotFound as error:
            raise HTTPException(status_code=404, detail="Artifact not found") from error
        return artifact.model_dump(mode="json")

    return router


def register_v2_routes(
    app: FastAPI,
    *,
    enabled: bool,
    graph: TracerGraph | None = None,
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    moderation_provider: ModerationProvider | None = None,
    answer_actor: AnswerActor | None = None,
    groundedness_actor: GroundednessActor | None = None,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    thread_resume_enabled: bool = False,
    artifact_lookup_enabled: bool = True,
    output_assessment_audit: OutputAssessmentAudit | None = None,
) -> None:
    """Register the default-off v2 routes when explicitly enabled."""
    if enabled:
        router = create_tracer_router(
            graph,
            refinement_actor,
            retriever,
            ranker,
            moderation_provider,
            answer_actor,
            groundedness_actor,
            history_token_budget,
            output_assessment_audit,
        )
        disabled_control_paths: set[str] = set()
        if not thread_resume_enabled:
            disabled_control_paths.add("/v2/threads/{thread_id}/resume/stream")
        if not artifact_lookup_enabled:
            disabled_control_paths.add("/v2/artifacts/{artifact_id}")
        if disabled_control_paths:
            router.routes = [
                route
                for route in router.routes
                if getattr(route, "path", None) not in disabled_control_paths
            ]
        app.include_router(router)
