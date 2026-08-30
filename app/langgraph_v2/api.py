"""FastAPI registration for the production LangGraph Linear Core."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import StateSnapshot
from psycopg_pool import AsyncConnectionPool
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

from app.langgraph_v2.answer import AnswerActor, build_answer_actor
from app.langgraph_v2.authorization import (
    TrustedRequestContext,
    get_trusted_request_context,
)
from app.langgraph_v2.checkpointing import (
    exact_checkpoint_config,
    initial_checkpoint_config,
)
from app.langgraph_v2.contracts import LiveStreamEvent, V2QueryRequest
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
    ConversationRecord,
    ResumeExpired,
    TurnNotFound,
    TurnRecord,
    turn_id_for_client_request,
)
from app.langgraph_v2.graph import (
    LinearGraph,
    LinearGraphState,
    build_linear_graph,
    linear_graph,
)
from app.langgraph_v2.groundedness import (
    GroundednessActor,
    UnavailableGroundednessActor,
    build_groundedness_actor,
)
from app.langgraph_v2.history import DEFAULT_HISTORY_TOKEN_BUDGET
from app.langgraph_v2.output_assessments import (
    LoggingOutputAssessmentAudit,
    OutputAssessmentAudit,
)
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
from app.langgraph_v2.stream import (
    GraphStreamCleanupError,
    RequestOwnedGraph,
    await_task_completion,
    stream_graph,
)
from app.services.exceptions import TenantNotFoundError

_LOGGER = logging.getLogger(__name__)


class _RequestOwnedStreamingResponse(StreamingResponse):
    """Cancel and await stream work as soon as the HTTP client disconnects."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return
        spec_version = tuple(
            map(
                int,
                scope.get("asgi", {}).get("spec_version", "2.0").split("."),
            )
        )
        stream_task = asyncio.create_task(self.stream_response(send))
        disconnect_task = asyncio.create_task(self.listen_for_disconnect(receive))
        try:
            done, _ = await asyncio.wait(
                {stream_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                await disconnect_task
                if not stream_task.done():
                    stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass
                except OSError as error:
                    if spec_version >= (2, 4):
                        raise ClientDisconnect from error
                    raise
            else:
                try:
                    await stream_task
                except OSError as error:
                    if spec_version >= (2, 4):
                        raise ClientDisconnect from error
                    raise
        finally:
            for task in (stream_task, disconnect_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                stream_task,
                disconnect_task,
                return_exceptions=True,
            )
        if self.background is not None:
            await self.background()


class CheckpointGraph(Protocol):
    """State lookup seam used by the Resume authorization path."""

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        """Read the current checkpoint state."""
        ...

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any],
        as_node: str,
    ) -> RunnableConfig:
        """Create a checkpoint whose next node can rebuild transient evidence."""
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
    ranker_top_n: int | None = None
    if hasattr(manager, "get_tenant_config"):
        tenant_config = manager.get_tenant_config(tenant_id)
        if tenant_config.ranking_config is not None:
            ranker_top_n = tenant_config.ranking_config.top_n
    return adapt_tenant_providers(
        manager.get_providers(tenant_id),
        ranker_top_n=ranker_top_n,
    )


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


def _resolve_groundedness_actor(
    app: FastAPI,
    tenant_id: str,
    injected: GroundednessActor | None,
    provider_bundle: V2ProviderBundle | None,
) -> GroundednessActor | None:
    """Resolve an injected or tenant-registry groundedness evaluator."""
    if injected is not None:
        return injected
    configured = getattr(app.state, "langgraph_v2_groundedness_actor", None)
    if configured is not None:
        return configured
    if provider_bundle is not None and provider_bundle.groundedness is not None:
        return provider_bundle.groundedness
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_model_registry"):
        return None
    return build_groundedness_actor(manager.get_model_registry(tenant_id))


def _resolve_groundedness_actor_safely(
    app: FastAPI,
    tenant_id: str,
    injected: GroundednessActor | None,
    provider_bundle: V2ProviderBundle | None,
) -> GroundednessActor | None:
    """Keep groundedness setup failures inside the advisory phase."""
    try:
        return _resolve_groundedness_actor(app, tenant_id, injected, provider_bundle)
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
    """Validate a configured tenant before starting v2 execution."""
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
    *,
    retriever: Retriever | None,
    ranker: Ranker | None,
    moderation_provider: ModerationProvider | None,
    provider_bundle: V2ProviderBundle | None,
) -> tuple[Retriever | None, Ranker | None, ModerationProvider | None]:
    """Resolve injected, app-level, and tenant-scoped v2 providers once."""
    configured_retriever = retriever or getattr(
        app.state, "langgraph_v2_retriever", None
    )
    configured_ranker = ranker or getattr(app.state, "langgraph_v2_ranker", None)
    configured_moderation = moderation_provider or getattr(
        app.state, "langgraph_v2_moderation_provider", None
    )
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


@dataclass(frozen=True)
class LinearGraphDependencies:
    """Resolved tenant and request dependencies for one Linear Graph."""

    checkpointer: BaseCheckpointSaver[Any] | None
    tenant_id: str
    message_repository: ConversationMessageRepository
    request_context: TrustedRequestContext
    history_token_budget: int
    output_assessment_audit: OutputAssessmentAudit
    refinement_actor: QuestionRefinementActor | None
    retriever: Retriever | None
    ranker: Ranker | None
    moderation_provider: ModerationProvider | None
    answer_actor: AnswerActor | None
    groundedness_actor: GroundednessActor | None


@dataclass(frozen=True)
class LinearGraphOverrides:
    """Optional router-level implementations used instead of tenant defaults."""

    refinement_actor: QuestionRefinementActor | None
    retriever: Retriever | None
    ranker: Ranker | None
    moderation_provider: ModerationProvider | None
    answer_actor: AnswerActor | None
    groundedness_actor: GroundednessActor | None
    history_token_budget: int
    output_assessment_audit: OutputAssessmentAudit | None


def _resolve_linear_graph_dependencies(
    app: FastAPI,
    *,
    pool: AsyncConnectionPool[Any],
    tenant_id: str,
    request_context: TrustedRequestContext,
    message_repository: ConversationMessageRepository,
    overrides: LinearGraphOverrides,
) -> LinearGraphDependencies:
    """Resolve request-scoped graph dependencies in one place."""
    provider_bundle = _resolve_provider_bundle(app, tenant_id)
    configured_retriever, configured_ranker, configured_moderation = (
        _resolve_phase_providers(
            app,
            retriever=overrides.retriever,
            ranker=overrides.ranker,
            moderation_provider=overrides.moderation_provider,
            provider_bundle=provider_bundle,
        )
    )
    return LinearGraphDependencies(
        checkpointer=cast(
            BaseCheckpointSaver[Any] | None,
            getattr(app.state, "langgraph_v2_checkpointer", None),
        ),
        tenant_id=tenant_id,
        message_repository=message_repository,
        request_context=request_context,
        history_token_budget=overrides.history_token_budget,
        output_assessment_audit=_resolve_output_assessment_audit(
            app, overrides.output_assessment_audit
        ),
        refinement_actor=_resolve_refinement_actor(
            app, tenant_id, overrides.refinement_actor
        ),
        retriever=configured_retriever,
        ranker=configured_ranker,
        moderation_provider=configured_moderation,
        answer_actor=_resolve_answer_actor(app, tenant_id, overrides.answer_actor),
        groundedness_actor=_resolve_groundedness_actor_safely(
            app,
            tenant_id,
            overrides.groundedness_actor,
            provider_bundle,
        ),
    )


def _build_request_graph(
    dependencies: LinearGraphDependencies,
    *,
    turn_id: uuid.UUID | None = None,
) -> LinearGraph:
    """Build the concrete Linear Graph from resolved request dependencies."""
    return build_linear_graph(
        dependencies.checkpointer,
        tenant_id=dependencies.tenant_id,
        current_turn_id=turn_id,
        message_repository=dependencies.message_repository,
        request_context=dependencies.request_context,
        history_token_budget=dependencies.history_token_budget,
        output_assessment_audit=dependencies.output_assessment_audit,
        refinement_actor=dependencies.refinement_actor,
        retriever=dependencies.retriever,
        ranker=dependencies.ranker,
        moderation_provider=dependencies.moderation_provider,
        answer_actor=dependencies.answer_actor,
        groundedness_actor=dependencies.groundedness_actor,
    )


async def _terminalize_checkpoint_event(
    message_repository: ConversationMessageRepository,
    event: LiveStreamEvent,
    *,
    context: TrustedRequestContext,
    conversation_id: str,
    turn_id: uuid.UUID,
) -> None:
    """Persist a successful checkpoint outcome exactly once by Turn."""
    if event.type == "error":
        return
    if event.type != "done":
        raise ValueError("checkpoint terminalization requires done or error")

    raw_data: object = event.data
    data = cast(dict[str, Any], raw_data) if isinstance(raw_data, dict) else {}
    answer = data.get("answer")
    if isinstance(answer, str):
        await message_repository.persist_assistant_message(
            context=context,
            conversation_id=conversation_id,
            turn_id=turn_id,
            content=answer,
            idempotency_key=f"turn:{turn_id}:assistant",
        )


async def _cleanup_request_execution(
    graph_stream: AsyncGenerator[str] | None,
    *,
    primary_error: BaseException | None,
) -> None:
    """Close request-owned graph work without mutating durable lifecycle state."""
    cleanup_error: BaseException | None = None
    cleanup_cancelled = False

    if graph_stream is not None:
        cancelled, error = await _await_cleanup_operation(graph_stream.aclose())
        cleanup_cancelled |= cancelled
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
    cancelled = await await_task_completion(task)
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


def _setup_failure_frame(message: str) -> str:
    """Render a compatible live error before graph streaming starts."""
    return LiveStreamEvent(
        type="error",
        data=message,
    ).to_sse()


async def _stream_request_execution(
    create_graph_stream: Callable[[], Awaitable[AsyncGenerator[str]]],
) -> AsyncIterator[str]:
    """Stream one request-owned Graph with shared error and cleanup semantics."""
    graph_stream: AsyncGenerator[str] | None = None
    primary_error: BaseException | None = None
    try:
        graph_stream = await create_graph_stream()
        async for frame in graph_stream:
            yield frame
    except asyncio.CancelledError as error:
        primary_error = error
        raise
    except GraphStreamCleanupError as error:
        primary_error = error
        raise
    except Exception as error:
        yield _setup_failure_frame(str(error) or "LangGraph execution failed.")
    finally:
        await _cleanup_request_execution(
            graph_stream,
            primary_error=primary_error,
        )


type LinearGraphStream = RequestOwnedGraph

_REMOVED_REPLAY_QUERY_PARAMETERS = frozenset(
    {"afterSequence", "after_sequence", "cursor"}
)
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
_NODES_REQUIRING_EVIDENCE = frozenset(
    {"reranking", "answer", "groundedness", "post_moderation", "finalization"}
)


def _reject_removed_replay_query_parameters(request: Request) -> None:
    """Reject replay cursors removed with the old Run control API."""
    removed = _REMOVED_REPLAY_QUERY_PARAMETERS.intersection(request.query_params)
    if removed:
        parameter = sorted(removed)[0]
        raise HTTPException(
            status_code=422,
            detail=f"Replay query parameter is no longer supported: {parameter}",
        )


class ThreadResumeConflict(RuntimeError):
    """A thread checkpoint cannot be recovered by the thread Resume route."""


@dataclass(frozen=True)
class ThreadResumeTarget:
    """Authorized checkpoint target for a request-owned thread Resume."""

    conversation: ConversationRecord
    turn: TurnRecord
    config: RunnableConfig


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


async def _prepare_thread_resume_target(
    *,
    checkpoint_graph: CheckpointGraph,
    message_repository: ConversationMessageRepository,
    context: TrustedRequestContext,
    thread_id: str,
    expected_turn_id: uuid.UUID,
) -> ThreadResumeTarget:
    """Authorize Resume and rewind when request-local evidence must be rebuilt."""
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
    exact_config = exact_checkpoint_config(
        thread_id=conversation.thread_id,
        checkpoint_ns=checkpoint_ns,
        checkpoint_id=checkpoint_id,
    )
    if _NODES_REQUIRING_EVIDENCE.intersection(snapshot.next):
        exact_config = await checkpoint_graph.aupdate_state(
            exact_config,
            {},
            as_node="question_refinement",
        )
    return ThreadResumeTarget(
        conversation=conversation,
        turn=turn,
        config=exact_config,
    )


def create_linear_router(
    graph: LinearGraphStream | None,
    overrides: LinearGraphOverrides,
) -> APIRouter:
    """Create the production router around a request-owned Linear Graph."""
    if overrides.history_token_budget < 0:
        raise ValueError("history_token_budget must not be negative")
    router = APIRouter(tags=["LangGraph v2"])

    @router.post("/v2/query/stream")
    async def query_stream(  # pyright: ignore[reportUnusedFunction] -- FastAPI route
        payload: V2QueryRequest,
        http_request: Request,
        request_context: Annotated[
            TrustedRequestContext, Depends(get_trusted_request_context)
        ],
        x_user_groups: Annotated[str, Header(alias="X-User-Groups")] = "",
    ) -> StreamingResponse:
        """Run the deterministic Linear Graph and return its events as SSE."""
        del x_user_groups
        _reject_removed_replay_query_parameters(http_request)
        _ensure_tenant_available(http_request.app, request_context.tenant_id)
        tenant_id = request_context.tenant_id
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
                tenant_id, conversation_id, payload.client_request_id
            )
        user_idempotency_key = f"turn:{turn_id}:user"

        async def create_graph_stream() -> AsyncGenerator[str]:
            await message_repository.create_turn(
                context=request_context,
                conversation_id=conversation_id,
                turn_id=turn_id,
                content=payload.query,
                idempotency_key=user_idempotency_key,
            )
            dependencies = _resolve_linear_graph_dependencies(
                http_request.app,
                pool=pool,
                tenant_id=tenant_id,
                request_context=request_context,
                message_repository=message_repository,
                overrides=overrides,
            )
            selected_graph = graph or linear_graph
            graph_config: RunnableConfig | None = None
            if graph is None:
                if dependencies.checkpointer is not None:
                    graph_config = initial_checkpoint_config(
                        thread_id=conversation.thread_id,
                        checkpoint_ns="",
                    )
                selected_graph = _build_request_graph(
                    dependencies,
                    turn_id=turn_id,
                )
            state: LinearGraphState = {
                "query": payload.query,
                "conversation_id": conversation_id,
                "turn_id": str(turn_id),
                "client_request_id": payload.client_request_id,
            }

            async def terminalize_checkpoint_event(event: LiveStreamEvent) -> None:
                await _terminalize_checkpoint_event(
                    message_repository,
                    event,
                    context=request_context,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                )

            return stream_graph(
                selected_graph,
                state,
                config=graph_config,
                terminal_sink=(
                    terminalize_checkpoint_event if graph is not None else None
                ),
            )

        return _RequestOwnedStreamingResponse(
            _stream_request_execution(create_graph_stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
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
        _reject_removed_replay_query_parameters(http_request)
        tenant_id = request_context.tenant_id
        _ensure_tenant_available(http_request.app, tenant_id)
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
        message_repository = _message_repository(http_request.app, pool)
        dependencies = _resolve_linear_graph_dependencies(
            http_request.app,
            pool=pool,
            tenant_id=tenant_id,
            request_context=request_context,
            message_repository=message_repository,
            overrides=overrides,
        )
        if dependencies.checkpointer is None:
            raise HTTPException(
                status_code=500, detail="LangGraph v2 checkpointer is not configured"
            )
        checkpoint_graph = _build_request_graph(dependencies)
        try:
            target = await _prepare_thread_resume_target(
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

        async def create_graph_stream() -> AsyncGenerator[str]:
            latest_turn = await message_repository.get_latest_turn(
                context=request_context,
                conversation_id=target.conversation.conversation_id,
            )
            if latest_turn.turn_id != target.turn.turn_id:
                raise ThreadResumeConflict("checkpoint Turn has been superseded")
            selected_graph = graph or _build_request_graph(
                dependencies,
                turn_id=target.turn.turn_id,
            )

            async def terminalize_checkpoint_event(event: LiveStreamEvent) -> None:
                await _terminalize_checkpoint_event(
                    message_repository,
                    event,
                    context=request_context,
                    conversation_id=target.conversation.conversation_id,
                    turn_id=target.turn.turn_id,
                )

            return stream_graph(
                selected_graph,
                None,
                config=target.config,
                terminal_sink=(
                    terminalize_checkpoint_event if graph is not None else None
                ),
            )

        return _RequestOwnedStreamingResponse(
            _stream_request_execution(create_graph_stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Conversation-Id": target.conversation.conversation_id,
                "X-Turn-Id": str(target.turn.turn_id),
                "X-Thread-Id": target.conversation.thread_id,
            },
        )

    return router


def register_v2_routes(
    app: FastAPI,
    *,
    enabled: bool,
    graph: LinearGraphStream | None = None,
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    moderation_provider: ModerationProvider | None = None,
    answer_actor: AnswerActor | None = None,
    groundedness_actor: GroundednessActor | None = None,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    thread_resume_enabled: bool = False,
    output_assessment_audit: OutputAssessmentAudit | None = None,
) -> None:
    """Register the default-off v2 routes when explicitly enabled."""
    if enabled:
        router = create_linear_router(
            graph,
            LinearGraphOverrides(
                refinement_actor=refinement_actor,
                retriever=retriever,
                ranker=ranker,
                moderation_provider=moderation_provider,
                answer_actor=answer_actor,
                groundedness_actor=groundedness_actor,
                history_token_budget=history_token_budget,
                output_assessment_audit=output_assessment_audit,
            ),
        )
        disabled_control_paths: set[str] = set()
        if not thread_resume_enabled:
            disabled_control_paths.add("/v2/threads/{thread_id}/resume/stream")
        if disabled_control_paths:
            router.routes = [
                route
                for route in router.routes
                if getattr(route, "path", None) not in disabled_control_paths
            ]
        app.include_router(router)
