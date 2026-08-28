"""Test-only FastAPI registration for the minimal v2 tracer."""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.answer import (
    ANSWER_CHUNK_INTERVAL_MS,
    AnswerActor,
    AnswerCancelled,
    build_answer_actor,
)
from app.langgraph_v2.artifacts import ArtifactNotFound, ArtifactRepository
from app.langgraph_v2.checkpointing import (
    FencedAsyncPostgresSaver,
    checkpoint_namespace_for,
    exact_checkpoint_config,
    initial_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.contracts import TracerStreamEvent, V2QueryRequest
from app.langgraph_v2.conversation_messages import ConversationMessageRepository
from app.langgraph_v2.graph import TracerState, build_tracer_graph, tracer_graph
from app.langgraph_v2.groundedness import GroundednessActor, build_groundedness_actor
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
    ClaimFenced,
    EventInput,
    ResumeConflict,
    RunEventRepository,
    RunNotFound,
)
from app.services.exceptions import TenantNotFoundError

_INSTANCE_ID = os.environ.get("LANGGRAPH_V2_INSTANCE_ID", socket.gethostname())


async def _pace_answer_chunk(
    event: TracerStreamEvent | Any,
    answer_chunk_count: list[int],
    answer_chunk_interval_ms: int,
) -> None:
    """Apply the bounded answer-chunk interval to one newly delivered token."""
    if event.type == "token" and event.event_key.startswith("phase:answer:token:"):
        if answer_chunk_count[0]:
            await asyncio.sleep(answer_chunk_interval_ms / 1000)
        answer_chunk_count[0] += 1


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
) -> Any:
    """Resolve an optional persisted, tenant-scoped cancellation intent check."""
    checker = getattr(app.state, "langgraph_v2_cancellation_check", None)
    if checker is None:
        return None

    async def check() -> bool:
        return bool(await checker(tenant_id, run_id))

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


async def _persist_result_events(
    repository: RunEventRepository,
    message_repository: ConversationMessageRepository,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    conversation_id: str,
    result: dict[str, Any],
    owner_instance_id: str,
    execution_epoch: int,
    suppress_sse_for_event_keys: set[str] | None = None,
    answer_chunk_interval_ms: int = ANSWER_CHUNK_INTERVAL_MS,
) -> AsyncIterator[str]:
    """Persist graph Events and serialize newly published SSE frames."""
    prior_keys = suppress_sse_for_event_keys or set()
    answer_chunk_count = [0]
    for raw_event in result["events"]:
        event = TracerStreamEvent.model_validate(raw_event)
        event_input = EventInput(
            event_key=event.event_key,
            type=event.type,
            step=event.step,
            data=event.data,
        )
        if event.type == "done":
            persisted = await repository.complete_run(
                tenant_id=tenant_id,
                run_id=run_id,
                event=event_input,
                owner_instance_id=owner_instance_id,
                execution_epoch=execution_epoch,
            )
            if isinstance(event.data, dict) and isinstance(
                event.data.get("answer"), str
            ):
                await message_repository.persist_assistant_message_after_completion(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    content=event.data["answer"],
                    idempotency_key=f"run:{run_id}:assistant",
                )
        elif event.type == "error":
            persisted = await repository.fail_run(
                tenant_id=tenant_id,
                run_id=run_id,
                event=event_input,
                owner_instance_id=owner_instance_id,
                execution_epoch=execution_epoch,
            )
        else:
            persisted = await repository.append_event(
                tenant_id=tenant_id,
                run_id=run_id,
                event=event_input,
                owner_instance_id=owner_instance_id,
                execution_epoch=execution_epoch,
            )
        if event.event_key not in prior_keys:
            await _pace_answer_chunk(
                event, answer_chunk_count, answer_chunk_interval_ms
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


class TracerGraph(Protocol):
    """Minimal graph invocation seam used by the test-only HTTP adapter."""

    async def ainvoke(
        self,
        state: TracerState | None,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Run one graph invocation and return its final state."""
        ...


async def _stream_unseen_events(
    repository: RunEventRepository,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    sent_keys: set[str],
    answer_chunk_count: list[int],
    answer_chunk_interval_ms: int,
) -> AsyncIterator[str]:
    """Serialize newly journaled Events and pace answer chunks consistently."""
    for event in await repository.list_events(tenant_id, run_id):
        if event.event_key in sent_keys:
            continue
        sent_keys.add(event.event_key)
        await _pace_answer_chunk(event, answer_chunk_count, answer_chunk_interval_ms)
        yield TracerStreamEvent(
            event_key=event.event_key,
            type=cast(Any, event.type),
            step=event.step,
            data=event.data,
            sequence=event.sequence,
        ).to_sse()


async def _stream_graph_result(
    selected_graph: TracerGraph,
    state: TracerState | None,
    graph_config: RunnableConfig | None,
    repository: RunEventRepository,
    message_repository: ConversationMessageRepository,
    *,
    tenant_id: str,
    run_id: uuid.UUID,
    conversation_id: str,
    owner_instance_id: str,
    execution_epoch: int,
    answer_chunk_interval_ms: int,
    initial_sent_keys: set[str] | None = None,
    forward_live_events: bool = True,
) -> AsyncIterator[str]:
    """Run a graph while forwarding already-journaled Events as they commit."""
    if graph_config is None:
        graph_task = asyncio.create_task(selected_graph.ainvoke(state))
    else:
        graph_task = asyncio.create_task(
            selected_graph.ainvoke(state, config=graph_config)
        )
    sent_keys = set(initial_sent_keys or ())
    answer_chunk_count = [0]
    while not graph_task.done():
        if forward_live_events:
            async for frame in _stream_unseen_events(
                repository,
                tenant_id=tenant_id,
                run_id=run_id,
                sent_keys=sent_keys,
                answer_chunk_count=answer_chunk_count,
                answer_chunk_interval_ms=answer_chunk_interval_ms,
            ):
                yield frame
        await asyncio.sleep(0.01)
    try:
        result = await graph_task
    except AnswerCancelled:
        async for frame in _stream_unseen_events(
            repository,
            tenant_id=tenant_id,
            run_id=run_id,
            sent_keys=sent_keys,
            answer_chunk_count=answer_chunk_count,
            answer_chunk_interval_ms=answer_chunk_interval_ms,
        ):
            yield frame
        return
    if not forward_live_events:
        sent_keys.update(
            event["event_key"]
            for event in result["events"]
            if event["event_key"].startswith("phase:")
            if event["type"] != "token"
            and not event["event_key"].startswith("phase:finalization:")
            and event["event_key"] != "lifecycle:completed:0"
        )
    async for frame in _persist_result_events(
        repository,
        message_repository,
        tenant_id=tenant_id,
        run_id=run_id,
        conversation_id=conversation_id,
        result=result,
        owner_instance_id=owner_instance_id,
        execution_epoch=execution_epoch,
        suppress_sse_for_event_keys=sent_keys,
        answer_chunk_interval_ms=answer_chunk_interval_ms,
    ):
        yield frame


def create_tracer_router(
    graph: TracerGraph | None = None,
    refinement_actor: QuestionRefinementActor | None = None,
    retriever: Retriever | None = None,
    ranker: Ranker | None = None,
    moderation_provider: ModerationProvider | None = None,
    answer_actor: AnswerActor | None = None,
    groundedness_actor: GroundednessActor | None = None,
    answer_chunk_interval_ms: int = ANSWER_CHUNK_INTERVAL_MS,
) -> APIRouter:
    """Create the test-only router around an injected graph invocation seam."""
    if not 200 <= answer_chunk_interval_ms <= 500:
        raise ValueError("answer_chunk_interval_ms must be between 200 and 500")
    router = APIRouter(tags=["LangGraph v2 tracer"])

    @router.post("/v2/query/stream")
    async def query_stream(
        payload: V2QueryRequest,
        http_request: Request,
        x_application_id: Annotated[str, Header(alias="X-Application-Id")],
        x_user_groups: Annotated[str, Header(alias="X-User-Groups")] = "",
    ) -> StreamingResponse:
        """Run the deterministic tracer and return its events as SSE."""
        del x_user_groups
        _ensure_tenant_available(http_request.app, x_application_id)
        run_id = uuid.uuid4()
        conversation_id = payload.conversation_id or str(uuid.uuid4())

        async def event_generator() -> AsyncIterator[str]:
            configured_pool = getattr(
                http_request.app.state,
                "langgraph_v2_postgres_pool",
                None,
            )
            if configured_pool is None:
                raise RuntimeError("LangGraph v2 PostgreSQL is not configured")
            pool = cast(AsyncConnectionPool[Any], configured_pool)
            repository = RunEventRepository(pool)
            message_repository = ConversationMessageRepository(pool)
            claim = await repository.create_run(
                tenant_id=x_application_id,
                run_id=run_id,
                conversation_id=conversation_id,
                owner_instance_id=_INSTANCE_ID,
            )
            await message_repository.resolve_conversation(
                tenant_id=x_application_id,
                conversation_id=conversation_id,
            )
            await message_repository.persist_user_message(
                tenant_id=x_application_id,
                conversation_id=conversation_id,
                run_id=run_id,
                content=payload.query,
                idempotency_key=f"run:{run_id}:user",
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
            try:
                configured_groundedness_actor = _resolve_groundedness_actor(
                    http_request.app, x_application_id, groundedness_actor
                )
            except Exception as exc:
                yield await _persist_setup_failure(
                    repository,
                    tenant_id=x_application_id,
                    run_id=run_id,
                    owner_instance_id=claim.owner_instance_id,
                    execution_epoch=claim.execution_epoch,
                    message=str(exc) or "Groundedness actor construction failed.",
                )
                return
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
                    repository=PhaseResultRepository(pool),
                    artifact_repository=ArtifactRepository(pool),
                    tenant_id=x_application_id,
                    run_id=run_id,
                    owner_instance_id=claim.owner_instance_id,
                    execution_epoch=claim.execution_epoch,
                    cancellation_check=_resolve_cancellation_check(
                        http_request.app, x_application_id, run_id
                    ),
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
                        thread_id=thread_id_for(
                            x_application_id,
                            conversation_id,
                        ),
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
            heartbeat_task = asyncio.create_task(
                _refresh_claim(
                    repository,
                    x_application_id,
                    run_id,
                    claim.owner_instance_id,
                    claim.execution_epoch,
                )
            )
            try:
                state: TracerState = {
                    "query": payload.query,
                    "conversation_id": conversation_id,
                    "client_request_id": payload.client_request_id,
                    "events": [],
                }
                async for frame in _stream_graph_result(
                    selected_graph,
                    state,
                    graph_config,
                    repository,
                    message_repository,
                    tenant_id=x_application_id,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    owner_instance_id=claim.owner_instance_id,
                    execution_epoch=claim.execution_epoch,
                    answer_chunk_interval_ms=answer_chunk_interval_ms,
                ):
                    yield frame
            finally:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Run-Id": str(run_id),
                "X-Conversation-Id": conversation_id,
            },
        )

    @router.post("/v2/runs/{run_id}/resume/stream")
    async def resume_stream(
        run_id: uuid.UUID,
        http_request: Request,
        x_application_id: Annotated[str, Header(alias="X-Application-Id")],
    ) -> StreamingResponse:
        """Resume one stale or interrupted Run from its exact checkpoint."""
        _ensure_tenant_available(http_request.app, x_application_id)
        configured_pool = getattr(
            http_request.app.state, "langgraph_v2_postgres_pool", None
        )
        if configured_pool is None:
            raise HTTPException(
                status_code=500, detail="LangGraph v2 PostgreSQL is not configured"
            )
        pool = cast(AsyncConnectionPool[Any], configured_pool)
        repository = RunEventRepository(pool)
        message_repository = ConversationMessageRepository(pool)
        try:
            claim = await repository.resume_run(
                tenant_id=x_application_id,
                run_id=run_id,
                owner_instance_id=_INSTANCE_ID,
            )
        except RunNotFound as error:
            raise HTTPException(status_code=404, detail="Run not found") from error
        except ResumeConflict as error:
            raise HTTPException(
                status_code=409, detail="Run is not resumable"
            ) from error
        previous_checkpoint_id = cast(str, claim.checkpoint_id)
        previous_checkpoint_ns = cast(str, claim.checkpoint_ns)

        async def event_generator() -> AsyncIterator[str]:
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
            try:
                configured_groundedness_actor = _resolve_groundedness_actor(
                    http_request.app, x_application_id, groundedness_actor
                )
            except Exception as exc:
                yield await _persist_setup_failure(
                    repository,
                    tenant_id=x_application_id,
                    run_id=run_id,
                    owner_instance_id=claim.owner_instance_id,
                    execution_epoch=claim.execution_epoch,
                    message=str(exc) or "Groundedness actor construction failed.",
                )
                return
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
                        repository=PhaseResultRepository(pool),
                        artifact_repository=ArtifactRepository(pool),
                        tenant_id=x_application_id,
                        run_id=run_id,
                        owner_instance_id=claim.owner_instance_id,
                        execution_epoch=claim.execution_epoch,
                        cancellation_check=_resolve_cancellation_check(
                            http_request.app, x_application_id, run_id
                        ),
                    ),
                    refinement_actor=configured_refinement_actor,
                    retriever=configured_retriever,
                    ranker=configured_ranker,
                    moderation_provider=configured_moderation,
                    answer_actor=configured_answer_actor,
                    groundedness_actor=configured_groundedness_actor,
                )
                graph_config = exact_checkpoint_config(
                    thread_id=thread_id_for(x_application_id, claim.conversation_id),
                    checkpoint_ns="",
                    checkpoint_id=previous_checkpoint_id,
                )
            heartbeat_task = asyncio.create_task(
                _refresh_claim(
                    repository,
                    x_application_id,
                    run_id,
                    claim.owner_instance_id,
                    claim.execution_epoch,
                )
            )
            try:
                async for frame in _stream_graph_result(
                    selected_graph,
                    None,
                    graph_config,
                    repository,
                    message_repository,
                    tenant_id=x_application_id,
                    run_id=run_id,
                    conversation_id=claim.conversation_id,
                    owner_instance_id=claim.owner_instance_id,
                    execution_epoch=claim.execution_epoch,
                    answer_chunk_interval_ms=answer_chunk_interval_ms,
                    initial_sent_keys={
                        event.event_key
                        for event in await repository.list_events(
                            x_application_id, run_id
                        )
                        if not (
                            event.type == "token"
                            and event.event_key.startswith("phase:answer:token:")
                        )
                    },
                    forward_live_events=False,
                ):
                    yield frame
            finally:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Run-Id": str(run_id)},
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


def register_tracer_routes(
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
    answer_chunk_interval_ms: int = ANSWER_CHUNK_INTERVAL_MS,
    resume_enabled: bool = False,
) -> None:
    """Register the test-only tracer routes when explicitly enabled."""
    if enabled:
        router = create_tracer_router(
            graph,
            refinement_actor,
            retriever,
            ranker,
            moderation_provider,
            answer_actor,
            groundedness_actor,
            answer_chunk_interval_ms,
        )
        if not resume_enabled:
            router.routes = [
                route
                for route in router.routes
                if getattr(route, "path", None) != "/v2/runs/{run_id}/resume/stream"
            ]
        app.include_router(router)
