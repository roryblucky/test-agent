"""Test-only FastAPI registration for the minimal v2 tracer."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from collections.abc import AsyncIterator, Awaitable, Coroutine
from contextlib import suppress
from datetime import timedelta
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.runnables import RunnableConfig
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
    checkpoint_namespace_for,
    exact_checkpoint_config,
    initial_checkpoint_config,
)
from app.langgraph_v2.contracts import (
    CancellationResponse,
    TracerStreamEvent,
    V2QueryRequest,
)
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
    ConversationRecord,
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
from app.langgraph_v2.replay import PersistedEventFollower
from app.langgraph_v2.reranking import Ranker
from app.langgraph_v2.retrieval import Retriever
from app.langgraph_v2.run_events import (
    CLAIM_HEARTBEAT_INTERVAL_SECONDS,
    CancellationObserved,
    ClaimFenced,
    EventInput,
    EventInvariantConflict,
    EventRecord,
    ResumeConflict,
    RunEventRepository,
    RunNotFound,
    RunRecord,
)
from app.langgraph_v2.runtime import LocalRunRuntime, RuntimeStopping
from app.langgraph_v2.stream import (
    GraphStreamCleanupError,
    RequestOwnedGraph,
    stream_graph,
)
from app.services.exceptions import TenantNotFoundError

_INSTANCE_ID = os.environ.get("LANGGRAPH_V2_INSTANCE_ID", socket.gethostname())
_LOGGER = logging.getLogger(__name__)


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
    if event.type == "done" and isinstance(event.data, dict) and isinstance(
        event.data.get("answer"), str
    ):
        try:
            resolved_turn_id = uuid.UUID(str(turn_id))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "completed Graph state is missing a valid turn_id"
            ) from error
        conflict: EventInvariantConflict | None = None
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
                        content=event.data["answer"],
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


async def _cleanup_request_execution(
    graph_stream: AsyncIterator[str] | None,
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


async def _persist_result_events(
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


class TracerInvoker(Protocol):
    """Legacy invocation seam retained by detached transitional execution."""

    async def ainvoke(
        self,
        state: TracerState | None,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Run one graph invocation and return its final state."""
        ...


type TracerGraph = RequestOwnedGraph | TracerInvoker


async def _stream_unseen_events(
    repository: RunEventRepository,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    sent_keys: set[str],
    suppress_replayed_phase_events: bool = False,
) -> AsyncIterator[str]:
    """Serialize newly journaled Events."""
    for event in await repository.list_events(tenant_id, run_id):
        if event.event_key in sent_keys:
            continue
        sent_keys.add(event.event_key)
        if (
            suppress_replayed_phase_events
            and event.event_key.startswith("phase:")
            and event.type != "token"
            and not event.event_key.startswith("phase:finalization:")
        ):
            continue
        yield TracerStreamEvent(
            event_key=event.event_key,
            type=cast(Any, event.type),
            step=event.step,
            data=event.data,
            sequence=event.sequence,
        ).to_sse()


async def _persist_graph_result(
    selected_graph: TracerInvoker,
    state: TracerState | None,
    graph_config: RunnableConfig | None,
    repository: RunEventRepository,
    cancellation_repository: CancellationRepository,
    message_repository: ConversationMessageRepository,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    conversation_id: str,
    expected_turn_id: uuid.UUID | None = None,
    owner_instance_id: str,
    execution_epoch: int,
    initial_sent_keys: set[str] | None = None,
) -> None:
    """Run a graph and persist its durable result independently of subscribers."""
    if graph_config is None:
        graph_task = asyncio.create_task(selected_graph.ainvoke(state))
    else:
        graph_task = asyncio.create_task(
            selected_graph.ainvoke(state, config=graph_config)
        )
    try:
        while not graph_task.done():
            await asyncio.sleep(0.01)
        result = await graph_task
        sent_keys = set(initial_sent_keys or ())
        sent_keys.update(
            event["event_key"]
            for event in result["events"]
            if event["event_key"].startswith("phase:")
            if event["type"] != "token"
            and not event["event_key"].startswith("phase:finalization:")
            and event["event_key"] != "lifecycle:completed:0"
        )
        async for _ in _persist_result_events(
            repository,
            message_repository,
            tenant_id=tenant_id,
            run_id=run_id,
            conversation_id=conversation_id,
            result=result,
            expected_turn_id=expected_turn_id,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
            suppress_sse_for_event_keys=sent_keys,
        ):
            pass
    except CancellationObserved:
        await cancellation_repository.apply_if_requested(
            tenant_id=tenant_id,
            run_id=run_id,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
        )
    finally:
        if not graph_task.done():
            graph_task.cancel()
        with suppress(asyncio.CancelledError, CancellationObserved):
            await graph_task


async def _execute_graph_run(
    selected_graph: TracerInvoker,
    state: TracerState | None,
    graph_config: RunnableConfig | None,
    repository: RunEventRepository,
    cancellation_repository: CancellationRepository,
    cancellation_observer: CancellationObserver,
    message_repository: ConversationMessageRepository,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    conversation_id: str,
    expected_turn_id: uuid.UUID | None = None,
    owner_instance_id: str,
    execution_epoch: int,
    initial_sent_keys: set[str] | None = None,
) -> None:
    """Execute a Run independently of any one SSE subscriber."""
    heartbeat_task = asyncio.create_task(
        _refresh_claim(
            repository,
            tenant_id,
            run_id,
            owner_instance_id,
            execution_epoch,
        )
    )
    await cancellation_observer.start()
    try:
        await _persist_graph_result(
            selected_graph,
            state,
            graph_config,
            repository,
            cancellation_repository,
            message_repository,
            tenant_id=tenant_id,
            run_id=run_id,
            conversation_id=conversation_id,
            expected_turn_id=expected_turn_id,
            owner_instance_id=owner_instance_id,
            execution_epoch=execution_epoch,
            initial_sent_keys=initial_sent_keys,
        )
    except (CancellationObserved, ClaimFenced):
        return
    except Exception as exc:
        try:
            await _persist_setup_failure(
                repository,
                tenant_id=tenant_id,
                run_id=run_id,
                owner_instance_id=owner_instance_id,
                execution_epoch=execution_epoch,
                message=str(exc) or "LangGraph execution failed.",
            )
        except ClaimFenced:
            return
    finally:
        await cancellation_observer.close()
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task


async def _subscribe_to_run(
    repository: RunEventRepository,
    execution_task: asyncio.Task[None],
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    initial_sent_keys: set[str] | None = None,
    suppress_replayed_phase_events: bool = False,
) -> AsyncIterator[str]:
    """Stream durable Events without owning or cancelling graph execution."""
    sent_keys = set(initial_sent_keys or ())
    while True:
        async for frame in _stream_unseen_events(
            repository,
            tenant_id=tenant_id,
            run_id=run_id,
            sent_keys=sent_keys,
            suppress_replayed_phase_events=suppress_replayed_phase_events,
        ):
            yield frame
        if execution_task.done():
            execution_task.result()
            async for frame in _stream_unseen_events(
                repository,
                tenant_id=tenant_id,
                run_id=run_id,
                sent_keys=sent_keys,
                suppress_replayed_phase_events=suppress_replayed_phase_events,
            ):
                yield frame
            return
        if (await repository.get_run(tenant_id, run_id)).status != "running":
            async for frame in _stream_unseen_events(
                repository,
                tenant_id=tenant_id,
                run_id=run_id,
                sent_keys=sent_keys,
                suppress_replayed_phase_events=suppress_replayed_phase_events,
            ):
                yield frame
            return
        await asyncio.sleep(0.01)


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


async def _follow_persisted_events(
    repository: RunEventRepository,
    wakeups: LiveEventWakeups,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    after_sequence: int,
    expected_execution_epoch: int | None = None,
) -> AsyncIterator[str]:
    """Serialize replay/live Event records from their durable sequence space."""
    follower = PersistedEventFollower(repository, wakeups)
    async for event in follower.follow(
        tenant_id=tenant_id,
        run_id=run_id,
        after_sequence=after_sequence,
        expected_execution_epoch=expected_execution_epoch,
    ):
        yield TracerStreamEvent(
            event_key=event.event_key,
            type=cast(Any, event.type),
            step=event.step,
            data=event.data,
            sequence=event.sequence,
        ).to_sse()


def _local_runtime(app: FastAPI) -> LocalRunRuntime:
    """Return the instance-local runtime used to retain detached executions."""
    runtime = getattr(app.state, "langgraph_v2_runtime", None)
    if runtime is None:
        runtime = LocalRunRuntime()
        app.state.langgraph_v2_runtime = runtime
    return cast(LocalRunRuntime, runtime)


async def _start_execution_or_interrupt(
    runtime: LocalRunRuntime,
    execution: Coroutine[Any, Any, None],
    repository: RunEventRepository,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    owner_instance_id: str,
    execution_epoch: int,
) -> asyncio.Task[None] | None:
    """Register execution or release the just-claimed Run during shutdown."""
    try:
        return runtime.start(execution)
    except RuntimeStopping:
        try:
            await repository.interrupt_run(
                tenant_id=tenant_id,
                run_id=run_id,
                owner_instance_id=owner_instance_id,
                execution_epoch=execution_epoch,
            )
        except ClaimFenced:
            pass
        return None


async def _authorized_run_conversation(
    repository: RunEventRepository,
    message_repository: ConversationMessageRepository,
    *,
    context: TrustedRequestContext,
    run_id: uuid.UUID,
) -> ConversationRecord:
    """Load a Run and authorize its Conversation as one API boundary."""
    run = await repository.get_run(context.tenant_id, run_id)
    conversation = await message_repository.get_conversation(
        context=context,
        conversation_id=run.conversation_id,
    )
    return conversation


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
    """Create the test-only router around an injected graph invocation seam."""
    if history_token_budget < 0:
        raise ValueError("history_token_budget must not be negative")
    router = APIRouter(tags=["LangGraph v2 tracer"])

    @router.post("/v2/query/stream")
    async def query_stream(
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
        runtime = _local_runtime(http_request.app)
        if not runtime.accepting:
            raise HTTPException(status_code=503, detail="LangGraph v2 is shutting down")
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
            graph_stream: AsyncIterator[str] | None = None
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

                        checkpoint_ns = checkpoint_namespace_for(
                            x_application_id,
                            str(run_id),
                            claim.execution_epoch,
                        )
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

                graph_stream = stream_graph(
                    cast(RequestOwnedGraph, selected_graph),
                    state,
                    config=graph_config,
                    event_sink=persist_event,
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

    @router.post("/v2/runs/{run_id}/resume/stream")
    async def resume_stream(
        run_id: uuid.UUID,
        http_request: Request,
        request_context: Annotated[
            TrustedRequestContext, Depends(get_trusted_request_context)
        ],
        after_sequence: Annotated[int, Query(alias="afterSequence", ge=0)] = 0,
    ) -> StreamingResponse:
        """Resume one stale or interrupted Run from its exact checkpoint."""
        x_application_id = request_context.tenant_id
        _ensure_tenant_available(http_request.app, x_application_id)
        runtime = _local_runtime(http_request.app)
        if not runtime.accepting:
            raise HTTPException(status_code=503, detail="LangGraph v2 is shutting down")
        configured_pool = getattr(
            http_request.app.state, "langgraph_v2_postgres_pool", None
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
            conversation = await _authorized_run_conversation(
                repository,
                message_repository,
                context=request_context,
                run_id=run_id,
            )
            claim = await repository.resume_run(
                tenant_id=x_application_id,
                run_id=run_id,
                owner_instance_id=_INSTANCE_ID,
            )
            resume_turn_id = await repository.get_run_turn_id(x_application_id, run_id)
        except (RunNotFound, ConversationNotFound) as error:
            raise HTTPException(status_code=404, detail="Run not found") from error
        except ResumeConflict as error:
            raise HTTPException(
                status_code=409, detail="Run is not resumable"
            ) from error
        previous_checkpoint_id = cast(str, claim.checkpoint_id)
        previous_checkpoint_ns = cast(str, claim.checkpoint_ns)

        configured_checkpointer = getattr(
            http_request.app.state, "langgraph_v2_checkpointer", None
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
        if graph is None and configured_checkpointer is not None:

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

            checkpoint_ns = checkpoint_namespace_for(
                x_application_id, str(run_id), claim.execution_epoch
            )
            selected_graph = build_tracer_graph(
                FencedAsyncPostgresSaver(
                    pool,
                    checkpoint_namespace=checkpoint_ns,
                    read_namespace=previous_checkpoint_ns,
                    pointer_writer=write_checkpoint_pointer,
                ),
                phase_context=PhaseExecutionContext(
                    repository=PhaseResultRepository(pool, live_events=wakeups),
                    artifact_repository=ArtifactRepository(pool),
                    message_repository=message_repository,
                    request_context=request_context,
                    history_token_budget=history_token_budget,
                    current_turn_id=resume_turn_id,
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
            graph_config = exact_checkpoint_config(
                thread_id=conversation.thread_id,
                checkpoint_ns="",
                checkpoint_id=previous_checkpoint_id,
            )
        initial_sent_keys = {
            event.event_key
            for event in await repository.list_events(x_application_id, run_id)
            if not (
                event.type == "token"
                and event.event_key.startswith("phase:answer:token:")
            )
        }
        await _start_execution_or_interrupt(
            runtime,
            _execute_graph_run(
                cast(TracerInvoker, selected_graph),
                None,
                graph_config,
                repository,
                cancellation_repository,
                cancellation_observer,
                message_repository,
                tenant_id=x_application_id,
                run_id=run_id,
                conversation_id=claim.conversation_id,
                expected_turn_id=resume_turn_id,
                owner_instance_id=claim.owner_instance_id,
                execution_epoch=claim.execution_epoch,
                initial_sent_keys=initial_sent_keys,
            ),
            repository,
            tenant_id=x_application_id,
            run_id=run_id,
            owner_instance_id=claim.owner_instance_id,
            execution_epoch=claim.execution_epoch,
        )

        async def event_generator() -> AsyncIterator[str]:
            async for frame in _follow_persisted_events(
                repository,
                wakeups,
                tenant_id=x_application_id,
                run_id=run_id,
                after_sequence=after_sequence,
                expected_execution_epoch=claim.execution_epoch,
            ):
                yield frame

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Run-Id": str(run_id)},
        )

    @router.get("/v2/runs/{run_id}/stream")
    async def replay_stream(
        run_id: uuid.UUID,
        http_request: Request,
        request_context: Annotated[
            TrustedRequestContext, Depends(get_trusted_request_context)
        ],
        after_sequence: Annotated[int, Query(alias="afterSequence", ge=0)] = 0,
    ) -> StreamingResponse:
        """Replay and then follow one Run from the requested sequence cursor."""
        configured_pool = getattr(
            http_request.app.state, "langgraph_v2_postgres_pool", None
        )
        if configured_pool is None:
            raise HTTPException(
                status_code=500, detail="LangGraph v2 PostgreSQL is not configured"
            )
        x_application_id = request_context.tenant_id
        wakeups = _live_events(http_request.app)
        repository = RunEventRepository(configured_pool, live_events=wakeups)
        messages = _message_repository(http_request.app, configured_pool)
        try:
            await _authorized_run_conversation(
                repository,
                messages,
                context=request_context,
                run_id=run_id,
            )
        except (RunNotFound, ConversationNotFound) as error:
            raise HTTPException(status_code=404, detail="Run not found") from error

        async def event_generator() -> AsyncIterator[str]:
            async for frame in _follow_persisted_events(
                repository,
                wakeups,
                tenant_id=x_application_id,
                run_id=run_id,
                after_sequence=after_sequence,
            ):
                yield frame

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Run-Id": str(run_id)},
        )

    @router.post(
        "/v2/runs/{run_id}/cancel",
        response_model=CancellationResponse,
        status_code=202,
        responses={200: {"model": CancellationResponse}},
    )
    async def cancel_run(
        run_id: uuid.UUID,
        http_request: Request,
        request_context: Annotated[
            TrustedRequestContext, Depends(get_trusted_request_context)
        ],
    ) -> JSONResponse:
        """Accept a durable cancellation intent without claiming completion."""
        configured_pool = getattr(
            http_request.app.state, "langgraph_v2_postgres_pool", None
        )
        if configured_pool is None:
            raise HTTPException(
                status_code=500, detail="LangGraph v2 PostgreSQL is not configured"
            )
        x_application_id = request_context.tenant_id
        try:
            repository = RunEventRepository(configured_pool)
            messages = _message_repository(http_request.app, configured_pool)
            await _authorized_run_conversation(
                repository,
                messages,
                context=request_context,
                run_id=run_id,
            )
            result = await CancellationRepository(
                configured_pool,
                wakeups=_live_events(http_request.app),
            ).request(tenant_id=x_application_id, run_id=run_id)
        except (RunNotFound, ConversationNotFound) as error:
            raise HTTPException(status_code=404, detail="Run not found") from error
        response = CancellationResponse(
            status="accepted" if result.accepted else "already_terminal",
            run_id=run_id,
            run_status=result.run_status,
        )
        return JSONResponse(
            status_code=202 if result.accepted else 200,
            content=response.model_dump(mode="json", by_alias=True),
        )

    @router.get("/v2/artifacts/{artifact_id}")
    async def get_artifact(
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
    resume_enabled: bool = False,
    replay_enabled: bool = False,
    cancellation_enabled: bool = False,
    artifact_lookup_enabled: bool = True,
    output_assessment_audit: OutputAssessmentAudit | None = None,
) -> None:
    """Register the default-off v2 routes when explicitly enabled."""
    if enabled:
        _local_runtime(app)
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
        if not resume_enabled:
            disabled_control_paths.add("/v2/runs/{run_id}/resume/stream")
        if not replay_enabled:
            disabled_control_paths.add("/v2/runs/{run_id}/stream")
        if not cancellation_enabled:
            disabled_control_paths.add("/v2/runs/{run_id}/cancel")
        if not artifact_lookup_enabled:
            disabled_control_paths.add("/v2/artifacts/{artifact_id}")
        if disabled_control_paths:
            router.routes = [
                route
                for route in router.routes
                if getattr(route, "path", None) not in disabled_control_paths
            ]
        app.include_router(router)
