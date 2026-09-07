"""Public sequential-Run isolation coverage for Agent conversations."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.config.models import FlowConfig, LangGraphRuntimeMode, LLMConfig, TenantConfig
from app.langgraph_v2.agent_batch import (
    DispatchBatch,
    SpecialistAttempt,
    SpecialistFindingDraft,
    SpecialistRegistration,
    SpecialistRegistry,
    SpecialistTaskInput,
    TaskProposal,
)
from app.langgraph_v2.agent_coordination import CoordinatorInput, Finish
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    FinancialResearchReport,
    PreparedSynthesis,
)
from app.langgraph_v2.agent_runtime import build_agent_runtime
from app.langgraph_v2.agent_scope import AgentIntentPolicy, SpecialistDescriptor
from app.langgraph_v2.api import GraphRuntimeAdapter
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import (
    AgentCheckpointStateAdapter,
    read_conversation_messages,
    thread_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.conversation_context import ConversationExchange
from app.models.workflow import IntentResult, QueryUnderstandingOutput, ResolvedQuery
from tests.integration.test_langgraph_v2_linear_core import (
    parse_sse,
    persistent_linear_app,
)

_TENANT_ID = "tenant-a"
_SUBJECT_ID = "subject-a"
_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000151")
_IDEMPOTENCY_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000152")
_RUN_ONE_ID = "consecutive-run-1"
_RUN_TWO_ID = "consecutive-run-2"
_RUN_ONE_QUERY = "Compare FUND-ONE with BENCHMARK-ONE."
_RUN_TWO_QUERY = "How did it compare with the benchmark?"
_RUN_ONE_STANDALONE = "Compare FUND-ONE with BENCHMARK-ONE as of 2026-09-06."
_RUN_TWO_STANDALONE = "Compare FUND-ONE with BENCHMARK-ONE as of 2026-09-07."
_RUN_ONE_ANSWER = "RUN-ONE canonical report. [[E:1]] [[E:2]]"
_RUN_TWO_ANSWER = "RUN-TWO canonical report. [[E:1]] [[E:2]]"
_HEADERS = {"X-Application-Id": _TENANT_ID, "X-Subject-Id": _SUBJECT_ID}


@dataclass
class _SecondInitializerObserver:
    """Capture the durable post-initializer state before any second-Run actor."""

    snapshot: dict[str, Any] | None = None

    def capture_if_second_initializer(self, values: Mapping[str, object]) -> None:
        raw_messages = values.get("conversation_messages")
        messages = (
            cast(list[object], raw_messages) if isinstance(raw_messages, list) else None
        )
        if (
            self.snapshot is None
            and values.get("request_id") == _RUN_TWO_ID
            and values.get("query") == _RUN_TWO_QUERY
            and isinstance(messages, list)
            and len(messages) == 3
            and values.get("standalone_query") is None
            and values.get("accepted_batches") == {}
            and values.get("coordination_rounds") == {}
        ):
            self.snapshot = copy.deepcopy(cast(dict[str, Any], values))

    @property
    def second_initializer_committed(self) -> bool:
        return self.snapshot is not None


class _ObservingSaver(AsyncPostgresSaver):
    """Observe the real PostgreSQL checkpoint only after it has committed."""

    observer: _SecondInitializerObserver

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> RunnableConfig:
        result = await super().aput(config, checkpoint, metadata, new_versions)
        values = checkpoint.get("channel_values", {})
        if isinstance(values, Mapping):
            self.observer.capture_if_second_initializer(
                cast(Mapping[str, object], values)
            )
        return result


class _HistoryUnderstanding:
    def __init__(self, observer: _SecondInitializerObserver) -> None:
        self.observer = observer
        self.histories: list[list[ConversationExchange]] = []

    async def understand(
        self,
        query: str,
        history: Sequence[ConversationExchange],
    ) -> QueryUnderstandingOutput:
        self.histories.append(list(history))
        if query == _RUN_ONE_QUERY:
            assert history == []
            standalone_query = _RUN_ONE_STANDALONE
        else:
            assert query == _RUN_TWO_QUERY
            assert self.observer.second_initializer_committed
            assert list(history) == [
                ConversationExchange(user=_RUN_ONE_QUERY, assistant=_RUN_ONE_ANSWER)
            ]
            standalone_query = _RUN_TWO_STANDALONE
        return QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query=query,
                standalone_query=standalone_query,
            ),
            intent=IntentResult(intent="fund_comparison", confidence=1.0),
        )


@dataclass
class _ShapeCoordinator:
    observer: _SecondInitializerObserver
    inputs: list[CoordinatorInput] = field(default_factory=list[CoordinatorInput])

    async def decide(self, input: CoordinatorInput) -> DispatchBatch | Finish:
        self.inputs.append(input)
        if input.standalone_query == _RUN_ONE_STANDALONE:
            if not input.prior_results:
                return DispatchBatch(
                    kind="dispatch",
                    tasks=(
                        TaskProposal(
                            specialist_id="fund-source",
                            objective="Read FUND-ONE source.",
                        ),
                        TaskProposal(
                            specialist_id="benchmark-source",
                            objective="Read BENCHMARK-ONE source.",
                        ),
                    ),
                )
            assert len(input.prior_results) == 2
            return Finish(kind="finish")
        assert input.standalone_query == _RUN_TWO_STANDALONE
        assert self.observer.second_initializer_committed
        assert "RUN-ONE" not in input.model_dump_json()
        if not input.prior_results:
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="combined-comparison",
                        objective="Read FUND-ONE, then compare it with BENCHMARK-ONE.",
                    ),
                ),
            )
        assert len(input.prior_results) == 1
        return Finish(kind="finish")


@dataclass
class _ShapeSpecialist:
    observer: _SecondInitializerObserver
    inputs: list[SpecialistTaskInput] = field(default_factory=list[SpecialistTaskInput])
    combined_steps: list[str] = field(default_factory=list[str])

    async def run(self, input: SpecialistTaskInput, **_: object) -> SpecialistAttempt:
        self.inputs.append(input)
        assert input.context_results == ()
        if input.objective == "Read FUND-ONE source.":
            return _attempt(
                summary="RUN-ONE fund source.",
                evidence=(
                    _evidence(
                        evidence_id="run-one-fund-evidence",
                        request_id=_RUN_ONE_ID,
                        task_id=input.task_id,
                        source="fund",
                    ),
                ),
            )
        if input.objective == "Read BENCHMARK-ONE source.":
            return _attempt(
                summary="RUN-ONE benchmark source.",
                evidence=(
                    _evidence(
                        evidence_id="run-one-benchmark-evidence",
                        request_id=_RUN_ONE_ID,
                        task_id=input.task_id,
                        source="benchmark",
                    ),
                ),
            )
        assert input.objective == "Read FUND-ONE, then compare it with BENCHMARK-ONE."
        assert self.observer.second_initializer_committed
        self.combined_steps.extend(("read-fund", "read-benchmark"))
        return _attempt(
            summary="RUN-TWO combined comparison.",
            evidence=(
                _evidence(
                    evidence_id="run-two-fund-evidence",
                    request_id=_RUN_TWO_ID,
                    task_id=input.task_id,
                    source="fund",
                ),
                _evidence(
                    evidence_id="run-two-benchmark-evidence",
                    request_id=_RUN_TWO_ID,
                    task_id=input.task_id,
                    source="benchmark",
                ),
            ),
        )


def _attempt(
    *, summary: str, evidence: tuple[EvidenceEnvelope, ...]
) -> SpecialistAttempt:
    return SpecialistAttempt(
        finding=SpecialistFindingDraft(
            summary=summary,
            evidence_ids=tuple(item.id for item in evidence),
        ),
        evidence=evidence,
    )


def _evidence(
    *, evidence_id: str, request_id: str, task_id: str, source: str
) -> EvidenceEnvelope:
    return EvidenceEnvelope(
        id=evidence_id,
        tenant_id=_TENANT_ID,
        request_id=request_id,
        task_id=task_id,
        source=source,
        source_url=f"https://fixture.test/{evidence_id}",
        title=f"{source} fixture",
        body=f"{request_id} request-local evidence body",
        excerpt=f"{request_id} {source} evidence excerpt.",
        as_of_date=date(2026, 9, 7),
    )


@dataclass
class _CapturingSynthesis:
    observer: _SecondInitializerObserver
    prepared_inputs: list[PreparedSynthesis] = field(
        default_factory=list[PreparedSynthesis]
    )

    async def synthesize(self, prepared: PreparedSynthesis) -> FinancialResearchReport:
        self.prepared_inputs.append(prepared)
        if prepared.standalone_query == _RUN_ONE_STANDALONE:
            assert tuple(item.id for item in prepared.evidence) == (
                "run-one-fund-evidence",
                "run-one-benchmark-evidence",
            )
            return FinancialResearchReport(markdown_report=_RUN_ONE_ANSWER)
        assert prepared.standalone_query == _RUN_TWO_STANDALONE
        assert self.observer.second_initializer_committed
        assert tuple(item.id for item in prepared.evidence) == (
            "run-two-fund-evidence",
            "run-two-benchmark-evidence",
        )
        assert prepared.calculations == ()
        return FinancialResearchReport(markdown_report=_RUN_TWO_ANSWER)

    async def repair(
        self,
        prepared: PreparedSynthesis,
        *,
        validation_errors: tuple[str, ...],
    ) -> FinancialResearchReport:
        del validation_errors
        return await self.synthesize(prepared)


class _AgentTenantManager:
    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        assert tenant_id == _TENANT_ID
        return TenantConfig(
            kms_app_name="Sequential Agent Tenant",
            application_id=_TENANT_ID,
            ad_groups=[],
            runtime_mode=LangGraphRuntimeMode.AGENT,
            llm_config=LLMConfig(models={}),
            flow_config=FlowConfig(),
        )

    def get_providers(self, tenant_id: str) -> object:
        assert tenant_id == _TENANT_ID
        return object()


def _app(
    database_url: str,
    *,
    observer: _SecondInitializerObserver,
    understanding: _HistoryUnderstanding,
    coordinator: _ShapeCoordinator,
    specialist: _ShapeSpecialist,
    synthesis: _CapturingSynthesis,
) -> FastAPI:
    policy = AgentIntentPolicy(
        intent="fund_comparison",
        description="Compare a fund with its benchmark.",
        specialist_descriptors=(
            SpecialistDescriptor(id="fund-source", description="Fund source"),
            SpecialistDescriptor(id="benchmark-source", description="Benchmark source"),
            SpecialistDescriptor(
                id="combined-comparison", description="Combined comparison"
            ),
        ),
    )
    registry = SpecialistRegistry(
        registrations=(
            SpecialistRegistration(id="fund-source", actor=specialist),
            SpecialistRegistration(id="benchmark-source", actor=specialist),
            SpecialistRegistration(id="combined-comparison", actor=specialist),
        ),
        tenant_eligible_ids=frozenset(
            {"fund-source", "benchmark-source", "combined-comparison"}
        ),
    )

    def factory(
        *,
        app: FastAPI,
        request_context: TrustedRequestContext,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> GraphRuntimeAdapter:
        return build_agent_runtime(
            app,
            request_context=request_context,
            checkpointer=checkpointer,
            query_understanding_actor=understanding,
            coordinator_actor=coordinator,
            specialist_registry=registry,
            intent_policies={policy.intent: policy},
            synthesis_actor=synthesis,
        )

    def saver_factory(conn: Any, *, serde: JsonPlusSerializer) -> _ObservingSaver:
        saver = _ObservingSaver(conn, serde=serde)
        saver.observer = observer
        return saver

    app = persistent_linear_app(
        database_url,
        agent_runtime_factory=factory,
        checkpointer_factory=saver_factory,
    )
    app.state.tenant_manager = _AgentTenantManager()
    return app


def _post(
    client: TestClient,
    *,
    query: str,
    request_id: str,
    conversation_id: UUID = _CONVERSATION_ID,
) -> Any:
    return client.post(
        "/v2/query/stream",
        json={
            "query": query,
            "sessionId": str(conversation_id),
            "clientRequestId": request_id,
        },
        headers=_HEADERS,
    )


def _checkpoint(
    app: FastAPI,
    client: TestClient,
    *,
    conversation_id: UUID = _CONVERSATION_ID,
) -> Any:
    assert client.portal is not None
    return client.portal.call(
        lambda: app.state.langgraph_v2_checkpointer.aget_tuple(
            thread_checkpoint_config(
                thread_id=thread_id_for(
                    _TENANT_ID,
                    _SUBJECT_ID,
                    "agent",
                    str(conversation_id),
                )
            )
        )
    )


def _messages(
    app: FastAPI,
    client: TestClient,
    *,
    conversation_id: UUID = _CONVERSATION_ID,
) -> list[Any]:
    assert client.portal is not None
    return client.portal.call(
        lambda: read_conversation_messages(
            app.state.langgraph_v2_checkpointer,
            thread_checkpoint_config(
                thread_id=thread_id_for(
                    _TENANT_ID,
                    _SUBJECT_ID,
                    "agent",
                    str(conversation_id),
                )
            ),
            state_adapter=AgentCheckpointStateAdapter(),
        )
    )


def _assert_publication(response: Any, checkpoint: Any, *, answer: str) -> None:
    assert response.status_code == 200
    events = parse_sse(response.text)
    done = [event for event in events if event["type"] == "done"]
    citations = [event for event in events if event["type"] == "citations"]
    assert len(done) == len(citations) == 1
    assert done[0]["data"]["answer"] == answer
    assert (
        "".join(event["data"] for event in events if event["type"] == "token") == answer
    )
    assert citations[0]["data"] == done[0]["data"]["citations"]
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    assert state["answer"] == answer
    assert state["final_response"] == done[0]["data"]


def test_consecutive_agent_runs_reset_state_and_change_execution_shape(
    langgraph_v2_migrated_database_url: str,
) -> None:
    observer = _SecondInitializerObserver()
    understanding = _HistoryUnderstanding(observer)
    coordinator = _ShapeCoordinator(observer)
    specialist = _ShapeSpecialist(observer)
    synthesis = _CapturingSynthesis(observer)
    app = _app(
        langgraph_v2_migrated_database_url,
        observer=observer,
        understanding=understanding,
        coordinator=coordinator,
        specialist=specialist,
        synthesis=synthesis,
    )

    with TestClient(app) as client:
        first = _post(client, query=_RUN_ONE_QUERY, request_id=_RUN_ONE_ID)
        first_checkpoint = copy.deepcopy(_checkpoint(app, client))
        second = _post(client, query=_RUN_TWO_QUERY, request_id=_RUN_TWO_ID)
        second_checkpoint = _checkpoint(app, client)
        messages = _messages(app, client)

    _assert_publication(first, first_checkpoint, answer=_RUN_ONE_ANSWER)
    _assert_publication(second, second_checkpoint, answer=_RUN_TWO_ANSWER)
    assert observer.snapshot is not None
    reset = observer.snapshot
    assert reset["query"] == _RUN_TWO_QUERY
    assert reset["conversation_id"] == str(_CONVERSATION_ID)
    assert reset["request_id"] == _RUN_TWO_ID
    assert [
        (message.type, message.text) for message in reset["conversation_messages"]
    ] == [
        ("human", _RUN_ONE_QUERY),
        ("ai", _RUN_ONE_ANSWER),
        ("human", _RUN_TWO_QUERY),
    ]
    assert reset["staged_contributions"] == {}
    assert reset["accepted_batches"] == {}
    assert reset["coordination_rounds"] == {}
    assert reset["dispatched_task"] is None
    assert reset["halted"] is False
    assert reset["coordination_finished"] is False
    assert reset["citations"] == []
    for channel in (
        "standalone_query",
        "intent",
        "research_scope",
        "active_batch",
        "coordination_request_id",
        "coordination_stop_reason",
        "clarification",
        "answer",
        "completion_status",
        "termination_reason",
        "incomplete_research",
        "final_response",
    ):
        assert reset[channel] is None

    assert understanding.histories == [
        [],
        [ConversationExchange(user=_RUN_ONE_QUERY, assistant=_RUN_ONE_ANSWER)],
    ]
    assert specialist.combined_steps == ["read-fund", "read-benchmark"]
    assert len(synthesis.prepared_inputs) == 2
    assert all(
        "conversation_messages" not in input.model_dump_json()
        for input in coordinator.inputs
    )
    assert all(input.context_results == () for input in specialist.inputs)
    assert all(
        "conversation_messages" not in repr(input) for input in specialist.inputs
    )
    assert all(
        "conversation_messages" not in prepared.model_dump_json()
        for prepared in synthesis.prepared_inputs
    )
    assert [(message.type, message.text) for message in messages] == [
        ("human", _RUN_ONE_QUERY),
        ("ai", _RUN_ONE_ANSWER),
        ("human", _RUN_TWO_QUERY),
        ("ai", _RUN_TWO_ANSWER),
    ]
    assert [message.id for message in messages] == [
        f"{_RUN_ONE_ID}:user",
        f"{_RUN_ONE_ID}:assistant",
        f"{_RUN_TWO_ID}:user",
        f"{_RUN_TWO_ID}:assistant",
    ]
    assert second_checkpoint is not None
    second_state = second_checkpoint.checkpoint["channel_values"]
    assert second_state["request_id"] == _RUN_TWO_ID
    assert second_state["staged_contributions"] == {}
    assert second_state["active_batch"] is None
    assert second_state["dispatched_task"] is None
    assert len(second_state["accepted_batches"]) == 1
    assert [
        round_["kind"] for round_ in second_state["coordination_rounds"].values()
    ] == [
        "dispatch",
        "finish",
    ]
    second_control_state = {
        key: value
        for key, value in second_state.items()
        if key != "conversation_messages"
    }
    assert _RUN_ONE_ID not in repr(second_control_state)
    assert "run-one-" not in repr(second_control_state)


def test_agent_request_id_retry_converges_and_conflict_fails_closed(
    langgraph_v2_migrated_database_url: str,
) -> None:
    observer = _SecondInitializerObserver()
    understanding = _HistoryUnderstanding(observer)
    coordinator = _ShapeCoordinator(observer)
    specialist = _ShapeSpecialist(observer)
    synthesis = _CapturingSynthesis(observer)
    app = _app(
        langgraph_v2_migrated_database_url,
        observer=observer,
        understanding=understanding,
        coordinator=coordinator,
        specialist=specialist,
        synthesis=synthesis,
    )

    with TestClient(app) as client:
        first = _post(
            client,
            query=_RUN_ONE_QUERY,
            request_id=_RUN_ONE_ID,
            conversation_id=_IDEMPOTENCY_CONVERSATION_ID,
        )
        first_checkpoint = copy.deepcopy(
            _checkpoint(app, client, conversation_id=_IDEMPOTENCY_CONVERSATION_ID)
        )
        retry = _post(
            client,
            query=_RUN_ONE_QUERY,
            request_id=_RUN_ONE_ID,
            conversation_id=_IDEMPOTENCY_CONVERSATION_ID,
        )
        retry_checkpoint = _checkpoint(
            app, client, conversation_id=_IDEMPOTENCY_CONVERSATION_ID
        )
        conflict = _post(
            client,
            query="Different input",
            request_id=_RUN_ONE_ID,
            conversation_id=_IDEMPOTENCY_CONVERSATION_ID,
        )
        messages = _messages(app, client, conversation_id=_IDEMPOTENCY_CONVERSATION_ID)

    _assert_publication(first, first_checkpoint, answer=_RUN_ONE_ANSWER)
    _assert_publication(retry, retry_checkpoint, answer=_RUN_ONE_ANSWER)
    assert first.headers["x-request-id"] == retry.headers["x-request-id"] == _RUN_ONE_ID
    assert parse_sse(first.text) == parse_sse(retry.text)
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "clientRequestId was already used for a different query"
    }
    assert understanding.histories == [[], []]
    assert len(specialist.inputs) == 4
    assert len(synthesis.prepared_inputs) == 2
    assert [(message.id, message.text) for message in messages] == [
        (f"{_RUN_ONE_ID}:user", _RUN_ONE_QUERY),
        (f"{_RUN_ONE_ID}:assistant", _RUN_ONE_ANSWER),
    ]
    assert first_checkpoint is not None and retry_checkpoint is not None
    assert (
        first_checkpoint.checkpoint["channel_values"]["coordination_rounds"]
        == retry_checkpoint.checkpoint["channel_values"]["coordination_rounds"]
    )
