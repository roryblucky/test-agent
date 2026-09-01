"""Shared FastAPI Query lifecycle for Tenant-selected LangGraph runtimes."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
)
from typing import Annotated, Any, Protocol, TypedDict, cast

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg_pool import AsyncConnectionPool
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

from app.config.models import LangGraphRuntimeMode
from app.langgraph_v2.answer import AnswerActor
from app.langgraph_v2.authorization import (
    TrustedRequestContext,
    get_trusted_request_context,
)
from app.langgraph_v2.checkpointing import thread_checkpoint_config
from app.langgraph_v2.contracts import (
    LiveStreamEvent,
    V2QueryRequest,
)
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
    turn_id_for_client_request,
)
from app.langgraph_v2.groundedness import GroundednessActor
from app.langgraph_v2.history import DEFAULT_HISTORY_TOKEN_BUDGET
from app.langgraph_v2.linear_runtime import LinearGraphOverrides, build_linear_runtime
from app.langgraph_v2.output_assessments import OutputAssessmentAudit
from app.langgraph_v2.pre_moderation import ModerationProvider
from app.langgraph_v2.question_refinement import QuestionRefinementActor
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


type GraphStream = RequestOwnedGraph

_REMOVED_REPLAY_QUERY_PARAMETERS = frozenset(
    {"afterSequence", "after_sequence", "cursor"}
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


def _message_repository(
    pool: AsyncConnectionPool[Any],
) -> ConversationMessageRepository:
    """Build the Tenant-scoped Message repository."""
    return ConversationMessageRepository(pool)


class GraphRuntimeAdapter(Protocol):
    """One Tenant runtime behind the shared v2 request lifecycle."""

    @property
    def runtime_mode(self) -> LangGraphRuntimeMode:
        """Return the trusted Tenant mode implemented by this adapter."""
        ...

    def build_graph(self, *, turn_id: uuid.UUID) -> RequestOwnedGraph:
        """Build the Graph that executes a Turn."""
        ...

    def initial_state_fields(
        self,
        *,
        payload: V2QueryRequest,
    ) -> Mapping[str, Any]:
        """Return runtime-specific fields for the initial graph state."""
        ...

class GraphRuntimeFactory(Protocol):
    """Construct a request-scoped Agent runtime adapter."""

    def __call__(
        self,
        *,
        app: FastAPI,
        pool: AsyncConnectionPool[Any],
        request_context: TrustedRequestContext,
        message_repository: ConversationMessageRepository,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> GraphRuntimeAdapter:
        """Return the runtime for the authenticated Tenant request."""
        ...


class _QueryGraphState(TypedDict):
    """Shared Query identity that runtime adapters cannot redefine."""

    query: str
    conversation_id: str
    turn_id: str
    client_request_id: str | None


def _initial_graph_state(
    runtime: GraphRuntimeAdapter,
    *,
    payload: V2QueryRequest,
    conversation_id: str,
    turn_id: uuid.UUID,
) -> dict[str, Any]:
    common_state: _QueryGraphState = {
        "query": payload.query,
        "conversation_id": conversation_id,
        "turn_id": str(turn_id),
        "client_request_id": payload.client_request_id,
    }
    runtime_fields = runtime.initial_state_fields(payload=payload)
    overlapping_fields = common_state.keys() & runtime_fields.keys()
    if overlapping_fields:
        fields = ", ".join(sorted(overlapping_fields))
        raise RuntimeError(f"Runtime redefined shared Query state: {fields}")
    return {**common_state, **runtime_fields}


def _tenant_runtime_mode(app: FastAPI, tenant_id: str) -> LangGraphRuntimeMode:
    """Read the fixed runtime mode from trusted Tenant configuration."""
    manager = getattr(app.state, "tenant_manager", None)
    if manager is None or not hasattr(manager, "get_tenant_config"):
        raise HTTPException(
            status_code=500,
            detail="Tenant runtime configuration is not available",
        )
    tenant_config = manager.get_tenant_config(tenant_id)
    return tenant_config.runtime_mode


def _resolve_graph_runtime(
    app: FastAPI,
    *,
    pool: AsyncConnectionPool[Any],
    request_context: TrustedRequestContext,
    message_repository: ConversationMessageRepository,
    checkpointer: BaseCheckpointSaver[Any],
    runtime_mode: LangGraphRuntimeMode,
    linear_graph_override: RequestOwnedGraph | None,
    linear_overrides: LinearGraphOverrides,
    agent_runtime_factory: GraphRuntimeFactory | None,
) -> GraphRuntimeAdapter:
    """Resolve exactly one trusted Tenant runtime for this request."""
    if runtime_mode is LangGraphRuntimeMode.LINEAR:
        return build_linear_runtime(
            app,
            tenant_id=request_context.tenant_id,
            request_context=request_context,
            message_repository=message_repository,
            checkpointer=checkpointer,
            overrides=linear_overrides,
            graph_override=linear_graph_override,
        )
    if agent_runtime_factory is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not configured")
    runtime = agent_runtime_factory(
        app=app,
        pool=pool,
        request_context=request_context,
        message_repository=message_repository,
        checkpointer=checkpointer,
    )
    if runtime.runtime_mode is not LangGraphRuntimeMode.AGENT:
        raise RuntimeError("Agent runtime factory returned the wrong runtime mode")
    return runtime


def create_v2_router(
    linear_graph_override: GraphStream | None,
    overrides: LinearGraphOverrides,
    agent_runtime_factory: GraphRuntimeFactory | None,
) -> APIRouter:
    """Create the shared router around Tenant-selected request-owned Graphs."""
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
        """Run the Tenant-configured Graph and return its events as SSE."""
        del x_user_groups
        _reject_removed_replay_query_parameters(http_request)
        _ensure_tenant_available(http_request.app, request_context.tenant_id)
        tenant_id = request_context.tenant_id
        runtime_mode = _tenant_runtime_mode(http_request.app, tenant_id)
        configured_checkpointer = getattr(
            http_request.app.state,
            "langgraph_v2_checkpointer",
            None,
        )
        if configured_checkpointer is None:
            raise HTTPException(
                status_code=500,
                detail="LangGraph v2 checkpointer is not configured",
            )
        checkpointer = cast(BaseCheckpointSaver[Any], configured_checkpointer)
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
        message_repository = _message_repository(pool)
        runtime = _resolve_graph_runtime(
            http_request.app,
            pool=pool,
            request_context=request_context,
            message_repository=message_repository,
            checkpointer=checkpointer,
            runtime_mode=runtime_mode,
            linear_graph_override=linear_graph_override,
            linear_overrides=overrides,
            agent_runtime_factory=agent_runtime_factory,
        )
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
            selected_graph = runtime.build_graph(turn_id=turn_id)
            graph_config = thread_checkpoint_config(
                thread_id=conversation.thread_id,
                checkpoint_ns="",
            )
            state = _initial_graph_state(
                runtime,
                payload=payload,
                conversation_id=conversation_id,
                turn_id=turn_id,
            )
            await message_repository.create_turn(
                context=request_context,
                conversation_id=conversation_id,
                turn_id=turn_id,
                content=payload.query,
                idempotency_key=user_idempotency_key,
            )

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
                terminal_sink=terminalize_checkpoint_event,
            )

        return _RequestOwnedStreamingResponse(
            _stream_request_execution(create_graph_stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Conversation-Id": conversation_id,
                "X-Turn-Id": str(turn_id),
            },
        )

    return router


def register_v2_routes(
    app: FastAPI,
    *,
    enabled: bool,
    linear_graph_override: GraphStream | None = None,
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    moderation_provider: ModerationProvider | None = None,
    answer_actor: AnswerActor | None = None,
    groundedness_actor: GroundednessActor | None = None,
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    output_assessment_audit: OutputAssessmentAudit | None = None,
    agent_runtime_factory: GraphRuntimeFactory | None = None,
) -> None:
    """Register the default-off v2 routes when explicitly enabled."""
    if enabled:
        router = create_v2_router(
            linear_graph_override,
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
            agent_runtime_factory,
        )
        app.include_router(router)
