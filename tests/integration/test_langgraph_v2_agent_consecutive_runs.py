"""Public sequential-Run isolation coverage using the financial golden fixture."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.langgraph_v2.agent_batch import DispatchBatch, TaskProposal
from app.langgraph_v2.agent_coordination import CoordinatorInput, Finish
from app.langgraph_v2.agent_evidence import FinancialResearchReport, PreparedSynthesis
from app.langgraph_v2.checkpointing import (
    AgentCheckpointStateAdapter,
    read_conversation_messages,
    thread_checkpoint_config,
    thread_id_for,
)
from app.langgraph_v2.conversation_context import ConversationExchange
from app.models.workflow import IntentResult, QueryUnderstandingOutput, ResolvedQuery
from tests.integration.test_langgraph_v2_agent_financial_golden_path import (
    FinancialFixture,
    financial_app,
)
from tests.integration.test_langgraph_v2_linear_core import parse_sse

_TENANT_ID = "tenant-a"
_SUBJECT_ID = "subject-a"
_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000151")
_IDEMPOTENCY_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000152")
_RUN_ONE_ID = "consecutive-run-1"
_RUN_TWO_ID = "consecutive-run-2"
_RUN_ONE_QUERY = "Compare FUND-ALPHA with BENCHMARK-OMEGA."
_RUN_TWO_QUERY = "How did it report its holdings and disclosures?"
_RUN_ONE_STANDALONE = "Compare FUND-ALPHA with BENCHMARK-OMEGA as of 2026-09-06."
_RUN_TWO_STANDALONE = "Research FUND-ALPHA holdings and disclosures as of 2026-09-07."
_RUN_ONE_REPORT = "RUN-ONE canonical report. [[E:1]]"
_RUN_ONE_ANSWER = (
    "Incomplete research: one requested task could not complete.\n"
    "- Research FUND\\-ALPHA holdings and disclosures\\.\n\n" + _RUN_ONE_REPORT
)
_RETRY_DIAGNOSTIC = "RUN-ONE-RETRY-DIAGNOSTIC"
_RUN_TWO_ANSWER = "RUN-TWO canonical report. [[E:1]] [[E:2]]"
_INTENT = "financial_golden_path"
_HEADERS = {"X-Application-Id": _TENANT_ID, "X-Subject-Id": _SUBJECT_ID}


@dataclass
class _SecondInitializerObserver:
    """Capture durable second-initializer state before any second-Run actor."""

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
    """Observe real PostgreSQL only after the checkpoint commit returns."""

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


@dataclass
class _HistoryUnderstanding:
    observer: _SecondInitializerObserver
    financial_fixture: FinancialFixture
    histories: list[list[ConversationExchange]] = field(
        default_factory=list[list[ConversationExchange]]
    )

    async def understand(
        self, query: str, history: Sequence[ConversationExchange]
    ) -> QueryUnderstandingOutput:
        self.histories.append(list(history))
        if query == _RUN_ONE_QUERY:
            assert history == []
            self.financial_fixture.configure_run(
                request_id=_RUN_ONE_ID,
                fund_task_dispatch_order=1,
            )
            standalone_query = _RUN_ONE_STANDALONE
        else:
            assert query == _RUN_TWO_QUERY
            assert self.observer.second_initializer_committed
            assert list(history) == [
                ConversationExchange(user=_RUN_ONE_QUERY, assistant=_RUN_ONE_ANSWER)
            ]
            self.financial_fixture.configure_run(
                request_id=_RUN_TWO_ID,
                fund_task_dispatch_order=0,
            )
            standalone_query = _RUN_TWO_STANDALONE
        return QueryUnderstandingOutput(
            resolved_query=ResolvedQuery(
                original_query=query, standalone_query=standalone_query
            ),
            intent=IntentResult(intent=_INTENT, confidence=1.0),
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
                            specialist_id="market-analysis",
                            objective="Analyze FUND-ALPHA against BENCHMARK-OMEGA.",
                        ),
                        TaskProposal(
                            specialist_id="fund-research",
                            objective="Research FUND-ALPHA holdings and disclosures.",
                        ),
                    ),
                )
            assert len(input.prior_results) == 1
            assert len(input.failed_tasks) == 1
            return Finish(kind="finish")
        assert input.standalone_query == _RUN_TWO_STANDALONE
        assert self.observer.second_initializer_committed
        assert _RUN_ONE_ID not in input.model_dump_json()
        if not input.prior_results:
            return DispatchBatch(
                kind="dispatch",
                tasks=(
                    TaskProposal(
                        specialist_id="fund-research",
                        objective="Research FUND-ALPHA holdings and disclosures.",
                    ),
                ),
            )
        assert len(input.prior_results) == 1
        return Finish(kind="finish")


@dataclass
class _CapturingSynthesis:
    observer: _SecondInitializerObserver
    prepared_inputs: list[PreparedSynthesis] = field(
        default_factory=list[PreparedSynthesis]
    )

    async def synthesize(self, prepared: PreparedSynthesis) -> FinancialResearchReport:
        self.prepared_inputs.append(prepared)
        assert prepared.intent == _INTENT
        if prepared.standalone_query == _RUN_ONE_STANDALONE:
            assert tuple(item.id for item in prepared.evidence) == ("price-evidence",)
            assert [item.alias for item in prepared.calculations] == [
                "C:1",
                "C:2",
                "C:3",
            ]
            return FinancialResearchReport(markdown_report=_RUN_ONE_REPORT)
        assert prepared.standalone_query == _RUN_TWO_STANDALONE
        assert self.observer.second_initializer_committed
        assert tuple(item.id for item in prepared.evidence) == (
            "holdings-evidence",
            "report-evidence",
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


@dataclass
class _ConsecutiveRunsFixture:
    app: FastAPI
    observer: _SecondInitializerObserver
    understanding: _HistoryUnderstanding
    coordinator: _ShapeCoordinator
    financial_fixture: FinancialFixture
    synthesis: _CapturingSynthesis


def _fixture(database_url: str) -> _ConsecutiveRunsFixture:
    observer = _SecondInitializerObserver()
    financial_fixture = FinancialFixture(
        alternate_holdings_failure=True,
        request_id=_RUN_ONE_ID,
        failure_request_ids=frozenset({_RUN_ONE_ID}),
        failure_diagnostic=_RETRY_DIAGNOSTIC,
    )
    understanding = _HistoryUnderstanding(observer, financial_fixture)
    coordinator = _ShapeCoordinator(observer)
    synthesis = _CapturingSynthesis(observer)

    def saver_factory(conn: Any, *, serde: JsonPlusSerializer) -> _ObservingSaver:
        saver = _ObservingSaver(conn, serde=serde)
        saver.observer = observer
        return saver

    return _ConsecutiveRunsFixture(
        app=financial_app(
            database_url,
            financial_fixture,
            query_understanding_actor=understanding,
            coordinator_actor=coordinator,
            synthesis_actor=synthesis,
            checkpointer_factory=saver_factory,
        ),
        observer=observer,
        understanding=understanding,
        coordinator=coordinator,
        financial_fixture=financial_fixture,
        synthesis=synthesis,
    )


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
                    _TENANT_ID, _SUBJECT_ID, "agent", str(conversation_id)
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
                    _TENANT_ID, _SUBJECT_ID, "agent", str(conversation_id)
                )
            ),
            state_adapter=AgentCheckpointStateAdapter(),
        )
    )


def _canonical_publication(
    response: Any,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return public results without scheduler-dependent progress ordering."""
    events = parse_sse(response.text)
    return (
        "".join(event["data"] for event in events if event["type"] == "token"),
        [event for event in events if event["type"] == "done"],
        [event for event in events if event["type"] == "citations"],
    )


def _assert_publication(response: Any, checkpoint: Any, *, answer: str) -> None:
    assert response.status_code == 200
    tokens, done, citations = _canonical_publication(response)
    assert len(done) == len(citations) == 1
    assert done[0]["data"]["answer"] == answer
    assert tokens == answer
    assert citations[0]["data"] == done[0]["data"]["citations"]
    assert checkpoint is not None
    state = checkpoint.checkpoint["channel_values"]
    assert state["answer"] == answer
    assert state["final_response"] == done[0]["data"]


def test_consecutive_agent_runs_reset_state_and_change_execution_shape(
    langgraph_v2_migrated_database_url: str,
) -> None:
    fixture = _fixture(langgraph_v2_migrated_database_url)

    with TestClient(fixture.app) as client:
        first = _post(client, query=_RUN_ONE_QUERY, request_id=_RUN_ONE_ID)
        first_checkpoint = copy.deepcopy(_checkpoint(fixture.app, client))
        second = _post(client, query=_RUN_TWO_QUERY, request_id=_RUN_TWO_ID)
        second_checkpoint = _checkpoint(fixture.app, client)
        messages = _messages(fixture.app, client)

    _assert_publication(first, first_checkpoint, answer=_RUN_ONE_ANSWER)
    _assert_publication(second, second_checkpoint, answer=_RUN_TWO_ANSWER)
    assert fixture.observer.snapshot is not None
    reset = fixture.observer.snapshot
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

    assert fixture.understanding.histories == [
        [],
        [ConversationExchange(user=_RUN_ONE_QUERY, assistant=_RUN_ONE_ANSWER)],
    ]
    assert fixture.financial_fixture.tools.provider_calls[-2:] == [
        ("fund", "FUND-ALPHA holdings"),
        ("fund", "FUND-ALPHA report"),
    ]
    assert len(fixture.synthesis.prepared_inputs) == 2
    assert all(input.intent == _INTENT for input in fixture.coordinator.inputs)
    assert all(
        prepared.intent == _INTENT for prepared in fixture.synthesis.prepared_inputs
    )
    assert all(
        "conversation_messages" not in input.model_dump_json()
        for input in fixture.coordinator.inputs
    )
    assert all(
        set(input)
        == {
            "context_results",
            "objective",
            "skill_summaries",
            "task_id",
            "validation_feedback",
        }
        for input in fixture.financial_fixture.specialists.task_prompts
    )
    assert all(
        "conversation_messages" not in prepared.model_dump_json()
        for prepared in fixture.synthesis.prepared_inputs
    )
    second_task_input = fixture.financial_fixture.specialists.task_prompts[-1]
    assert second_task_input["validation_feedback"] is None
    assert _RUN_ONE_ID not in repr(second_task_input)
    assert _RUN_ONE_ID not in fixture.synthesis.prepared_inputs[-1].model_dump_json()
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
    assert first_checkpoint is not None
    first_state = first_checkpoint.checkpoint["channel_values"]
    first_batch = next(iter(first_state["accepted_batches"].values()))
    assert [item["kind"] for item in first_batch["outcomes"]] == [
        "succeeded",
        "failed",
    ]
    assert (
        fixture.financial_fixture.specialists.emitted_failure_diagnostics
        == [_RETRY_DIAGNOSTIC] * 3
    )
    assert first_batch["usage"]["model_requests"] > 0
    assert first_batch["usage"]["completed_tool_calls"] > 0
    assert first_batch["usage"]["tool_attempts"] > 0
    assert len(first_batch["calculations"]) == 3
    assert {item["request_id"] for item in first_batch["calculations"]} == {_RUN_ONE_ID}
    assert {
        evidence_id
        for item in first_batch["calculations"]
        for evidence_id in item["evidence_refs"]
    } == {"price-evidence"}
    assert [
        input["validation_feedback"]
        for input in fixture.financial_fixture.specialists.task_prompts
        if input["task_id"] == first_batch["outcomes"][0]["task_id"]
    ] == [None]
    assert second_checkpoint is not None
    second_state = second_checkpoint.checkpoint["channel_values"]
    assert second_state["request_id"] == _RUN_TWO_ID
    assert second_state["staged_contributions"] == {}
    assert second_state["active_batch"] is None
    assert second_state["dispatched_task"] is None
    assert len(second_state["accepted_batches"]) == 1
    second_batch = next(iter(second_state["accepted_batches"].values()))
    assert len(second_batch["outcomes"]) == 1
    assert second_batch["calculations"] == []
    assert (
        second_batch["usage"]["model_requests"] < first_batch["usage"]["model_requests"]
    )
    assert (
        second_batch["usage"]["tool_attempts"] < first_batch["usage"]["tool_attempts"]
    )
    assert [
        round_["kind"] for round_ in second_state["coordination_rounds"].values()
    ] == ["dispatch", "finish"]
    second_control_state = {
        key: value
        for key, value in second_state.items()
        if key != "conversation_messages"
    }
    assert _RUN_ONE_ID not in repr(second_control_state)
    assert "price-evidence" not in repr(second_control_state)
    assert "C:1" not in repr(second_control_state)
    assert _RETRY_DIAGNOSTIC not in repr(second_control_state)
    assert _RUN_ONE_ID not in second.text
    assert _RETRY_DIAGNOSTIC not in second.text


def test_agent_request_id_retry_converges_and_conflict_fails_closed(
    langgraph_v2_migrated_database_url: str,
) -> None:
    fixture = _fixture(langgraph_v2_migrated_database_url)

    with TestClient(fixture.app) as client:
        first = _post(
            client,
            query=_RUN_ONE_QUERY,
            request_id=_RUN_ONE_ID,
            conversation_id=_IDEMPOTENCY_CONVERSATION_ID,
        )
        first_checkpoint = copy.deepcopy(
            _checkpoint(
                fixture.app,
                client,
                conversation_id=_IDEMPOTENCY_CONVERSATION_ID,
            )
        )
        retry = _post(
            client,
            query=_RUN_ONE_QUERY,
            request_id=_RUN_ONE_ID,
            conversation_id=_IDEMPOTENCY_CONVERSATION_ID,
        )
        retry_checkpoint = _checkpoint(
            fixture.app,
            client,
            conversation_id=_IDEMPOTENCY_CONVERSATION_ID,
        )
        conflict = _post(
            client,
            query="Different input",
            request_id=_RUN_ONE_ID,
            conversation_id=_IDEMPOTENCY_CONVERSATION_ID,
        )
        messages = _messages(
            fixture.app,
            client,
            conversation_id=_IDEMPOTENCY_CONVERSATION_ID,
        )

    _assert_publication(first, first_checkpoint, answer=_RUN_ONE_ANSWER)
    _assert_publication(retry, retry_checkpoint, answer=_RUN_ONE_ANSWER)
    assert first.headers["x-request-id"] == retry.headers["x-request-id"] == _RUN_ONE_ID
    assert _canonical_publication(first) == _canonical_publication(retry)
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "clientRequestId was already used for a different query"
    }
    assert fixture.understanding.histories == [[], []]
    assert len(fixture.financial_fixture.specialists.task_prompts) == 2
    assert len(fixture.synthesis.prepared_inputs) == 2
    assert [(message.id, message.text) for message in messages] == [
        (f"{_RUN_ONE_ID}:user", _RUN_ONE_QUERY),
        (f"{_RUN_ONE_ID}:assistant", _RUN_ONE_ANSWER),
    ]
    assert first_checkpoint is not None and retry_checkpoint is not None
    assert (
        first_checkpoint.checkpoint["channel_values"]
        == retry_checkpoint.checkpoint["channel_values"]
    )
