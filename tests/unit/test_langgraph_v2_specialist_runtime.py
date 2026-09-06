"""No-Tool Specialist PydanticAI actor coverage."""

import asyncio
import json
from datetime import date
from typing import cast

import httpx
import pydantic_ai.models as models
import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.usage import RunUsage

import app.agents.specialist as specialist_module
import app.langgraph_v2.agent_evidence as agent_evidence
from app.agents.specialist import (
    SPECIALIST_MAX_TOKENS,
    SPECIALIST_TIMEOUT_SECONDS,
    PydanticAISpecialistActor,
    create_bound_specialist_actor,
    create_specialist_agent,
)
from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import (
    PriorResultView,
    SpecialistFindingDraft,
    SpecialistTaskInput,
    canonical_context_json_details,
)
from app.langgraph_v2.agent_coordination import MAX_SPECIALIST_CONTEXT_BYTES
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    EvidenceInvocationContext,
    ExpectedToolUnavailability,
    SpecialistToolCapture,
    ToolUnavailable,
    ToolUnavailableReason,
    bind_evidence_tool,
)
from app.langgraph_v2.agent_skills import (
    SkillReference,
    SkillRegistration,
    SpecialistSkillRegistry,
)
from app.langgraph_v2.specialist_retry import (
    RetryDisposition,
    SpecialistInvocationFailure,
    SpecialistModelBoundary,
    SpecialistModelRequestTimeout,
    classify_specialist_failure,
    specialist_usage_limits,
)


def _context() -> EvidenceInvocationContext:
    return EvidenceInvocationContext(
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
        allowed_tool_ids=frozenset({"read_evidence"}),
        allowed_sources=frozenset({"filing"}),
        allowed_queries=frozenset({"Apple revenue", "excerpt"}),
    )


@pytest.fixture(autouse=True)
def disable_real_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep all Specialist actor coverage local to TestModel/FunctionModel."""
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


class _Registry:
    def __init__(self) -> None:
        self.name: str | None = None
        self.kwargs: dict[str, object] | None = None

    def create_agent(self, name: str, **kwargs: object) -> object:
        self.name = name
        self.kwargs = kwargs
        return object()


def test_specialist_factory_disables_tools_and_builtin_retries() -> None:
    registry = _Registry()

    create_specialist_agent(cast(ModelRegistry, registry), model_name="specialist")

    assert registry.name == "specialist"
    assert registry.kwargs is not None
    assert registry.kwargs == {
        "output_type": SpecialistFindingDraft,
        "instructions": registry.kwargs["instructions"],
        "tools": (),
        "retries": 0,
        "tool_retries": 0,
        "output_retries": 0,
        "end_strategy": "early",
    }
    instructions = registry.kwargs["instructions"]
    assert isinstance(instructions, str)
    assert "activate one eligible Skill" in instructions


def test_bound_specialist_factory_uses_exact_frozen_tools() -> None:
    registry = _Registry()

    async def read_evidence(source: str, query: str) -> object:
        del source, query
        return object()

    actor = create_bound_specialist_actor(
        cast(ModelRegistry, registry),
        model_name="specialist",
        tools=(read_evidence,),
        tool_capture=SpecialistToolCapture(),
    )

    assert isinstance(actor, PydanticAISpecialistActor)
    assert registry.kwargs is not None
    assert registry.kwargs["tools"] == (read_evidence,)


@pytest.mark.asyncio
async def test_specialist_activates_a_summary_before_using_an_existing_tool() -> None:
    skill_registry = SpecialistSkillRegistry(
        registrations=(
            SkillRegistration(
                name="filing-analysis",
                version="1",
                description="Read a filing.",
                instructions="FULL-SKILL-INSTRUCTIONS-SENTINEL",
                references=(
                    SkillReference(
                        name="filing-guide",
                        content="FULL-SKILL-REFERENCE-SENTINEL",
                    ),
                ),
                required_tool_ids=frozenset({"read_evidence"}),
            ),
        ),
        tenant_eligible_names=frozenset({"filing-analysis"}),
        shared_skill_names=frozenset({"filing-analysis"}),
    )
    invocation = skill_registry.begin_invocation(
        specialist_skill_names=frozenset(),
        scope_skill_names=frozenset({"filing-analysis"}),
        effective_tool_ids=frozenset({"read_evidence"}),
    )
    calls = 0

    async def read_evidence() -> dict[str, str]:
        return {"excerpt": "already-authorized-tool-result"}

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        business_tools = [
            tool.name for tool in info.function_tools if tool.name != "activate_skill"
        ]
        assert business_tools == ["read_evidence"]
        if calls == 1:
            assert "filing-analysis" in repr(messages)
            assert "FULL-SKILL-INSTRUCTIONS-SENTINEL" not in repr(messages)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="activate_skill",
                        args={"skill_name": "filing-analysis"},
                    )
                ]
            )
        if calls == 2:
            assert "FULL-SKILL-INSTRUCTIONS-SENTINEL" in repr(messages)
            assert "FULL-SKILL-REFERENCE-SENTINEL" in repr(messages)
            return ModelResponse(
                parts=[ToolCallPart(tool_name="read_evidence", args={})]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"summary": "Filing analysis", "evidence_ids": []},
                )
            ]
        )

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(read_evidence, invocation.activation_tool()),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        ),
        skill_invocation=invocation,
    )

    attempt = await actor.run(
        SpecialistTaskInput(
            task_id="task-1",
            objective="Assess the filing.",
        )
    )

    assert calls == 3
    assert attempt.skill_pins == invocation.pins


@pytest.mark.asyncio
async def test_no_tool_specialist_returns_one_structured_finding() -> None:
    model = TestModel(
        call_tools=[],
        custom_output_args={"summary": "No-tool finding", "evidence_ids": []},
    )
    agent = Agent(
        model,
        output_type=SpecialistFindingDraft,
        tools=(),
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )
    actor = PydanticAISpecialistActor(agent)

    with capture_run_messages() as messages:
        finding = await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess market outlook.")
        )

    assert finding.finding == SpecialistFindingDraft(summary="No-tool finding")
    assert finding.messages == tuple(messages)
    assert len(messages) == 3
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].usage.requests == 1
    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.function_tools == []
    assert SPECIALIST_TIMEOUT_SECONDS == 60
    assert SPECIALIST_MAX_TOKENS == 2000


@pytest.mark.asyncio
async def test_specialist_function_model_has_one_request_and_structured_trace() -> None:
    captures: list[tuple[list[ModelMessage], AgentInfo]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captures.append((messages, info))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"summary": "No-tool finding", "evidence_ids": []},
                )
            ]
        )

    agent = Agent(
        FunctionModel(model),
        output_type=SpecialistFindingDraft,
        tools=(),
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )
    actor = PydanticAISpecialistActor(agent)
    with capture_run_messages() as messages:
        finding = await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess market outlook.")
        )

    assert finding.finding == SpecialistFindingDraft(summary="No-tool finding")
    assert finding.messages == tuple(messages)
    assert len(captures) == 1
    assert captures[0][1].function_tools == []
    assert captures[0][1].model_settings == {
        "max_tokens": SPECIALIST_MAX_TOKENS,
        "timeout": SPECIALIST_TIMEOUT_SECONDS,
        "parallel_tool_calls": True,
    }
    assert len(messages) == 3
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].usage.requests == 1

    with capture_run_messages() as result_messages:
        result = await agent.run("Assess market outlook.")
    assert result.output == finding.finding
    assert len(result.new_messages()) == 3
    assert result.usage().requests == 1
    assert len(result_messages) == 3


@pytest.mark.asyncio
async def test_specialist_model_receives_only_materialized_prior_results() -> None:
    prompts: list[dict[str, object]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        user_prompt = next(
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        assert isinstance(user_prompt, str)
        prompts.append(json.loads(user_prompt))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"summary": "Follow-up finding", "evidence_ids": []},
                )
            ]
        )

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )
    prior = PriorResultView(
        task_id="task-prior",
        summary="Stable prior finding.",
        evidence_ids=("evidence-1",),
    )
    context_json_bytes, context_json_sha256 = canonical_context_json_details((prior,))

    await actor.run(
        SpecialistTaskInput(
            task_id="task-follow-up",
            objective="Assess implications.",
            context_results=(prior,),
            context_json_bytes=context_json_bytes,
            context_json_sha256=context_json_sha256,
        )
    )

    assert prompts == [
        {
            "context_results": [prior.model_dump(mode="json")],
            "objective": "Assess implications.",
            "skill_summaries": [],
            "task_id": "task-follow-up",
            "validation_feedback": None,
        }
    ]


@pytest.mark.asyncio
async def test_specialist_non_ascii_context_uses_the_accepted_utf8_serialization() -> (
    None
):
    prompts: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = next(
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        assert isinstance(prompt, str)
        prompts.append(prompt)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"summary": "Follow-up finding", "evidence_ids": []},
                )
            ]
        )

    context = tuple(
        PriorResultView(task_id=f"task-{index}", summary="é" * 2_600)
        for index in range(8)
    )
    context_json_bytes, context_json_sha256 = canonical_context_json_details(context)
    canonical_context = json.dumps(
        [result.model_dump(mode="json") for result in context],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert context_json_bytes == len(canonical_context.encode("utf-8"))
    assert context_json_bytes < MAX_SPECIALIST_CONTEXT_BYTES

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )
    await actor.run(
        SpecialistTaskInput(
            task_id="task-follow-up",
            objective="Assess implications.",
            context_results=context,
            context_json_bytes=context_json_bytes,
            context_json_sha256=context_json_sha256,
        )
    )

    assert f'"context_results":{canonical_context},' in prompts[0]
    assert "\\u00e9" not in prompts[0]


@pytest.mark.asyncio
async def test_specialist_accepts_tool_metadata_only_after_terminal_finding() -> None:
    capture = SpecialistToolCapture()
    calls = 0

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        assert (source, query) == ("filing", "Apple revenue")
        return EvidenceEnvelope(
            id="evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            source="filing",
            source_url="https://example.test/filing",
            title="Annual filing",
            body="BODY-SENTINEL",
            excerpt="Apple revenue grew.",
            as_of_date=date(2026, 9, 6),
        )

    tool = bind_evidence_tool(
        provider,
        context=_context(),
        capture=capture,
    )

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert [tool.name for tool in info.function_tools] == ["read_evidence"]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_evidence",
                        args={"source": "filing", "query": "Apple revenue"},
                    )
                ]
            )
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert len(tool_returns) == 1
        assert tool_returns[0].content == {
            "evidence_id": "evidence-1",
            "excerpt": "Apple revenue grew.",
        }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "summary": "Apple revenue grew.",
                        "evidence_ids": ["evidence-1"],
                    },
                )
            ]
        )

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(tool,),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        ),
        tool_capture=capture,
    )

    attempt = await actor.run(
        SpecialistTaskInput(task_id="task-1", objective="Assess Apple revenue.")
    )

    assert calls == 2
    assert attempt.finding.evidence_ids == ("evidence-1",)
    assert attempt.evidence[0].id == "evidence-1"
    assert capture.evidence == []


@pytest.mark.asyncio
async def test_specialist_keeps_expected_unavailability_after_a_fallback() -> None:
    class SourceUnreachable(Exception):
        pass

    capture = SpecialistToolCapture()
    provider_calls = 0

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        nonlocal provider_calls
        provider_calls += 1
        assert (source, query) == ("filing", "Apple revenue")
        if provider_calls == 1:
            raise SourceUnreachable()
        return EvidenceEnvelope(
            id="evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            source="filing",
            source_url="https://example.test/filing",
            title="Annual filing",
            body="BODY-SENTINEL",
            excerpt="Apple revenue grew.",
            as_of_date=date(2026, 9, 6),
        )

    tool = bind_evidence_tool(
        provider,
        context=_context(),
        capture=capture,
        expected_unavailability=(
            ExpectedToolUnavailability(
                exception_type=SourceUnreachable,
                reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
            ),
        ),
    )
    model_calls = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_evidence",
                        tool_call_id="call-1",
                        args={"source": "filing", "query": "Apple revenue"},
                    )
                ]
            )
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if model_calls == 2:
            assert tool_returns[0].content == ToolUnavailable(
                reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
                requested_coverage="Apple revenue",
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_evidence",
                        tool_call_id="call-2",
                        args={"source": "filing", "query": "Apple revenue"},
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "summary": "Apple revenue grew.",
                        "evidence_ids": ["evidence-1"],
                    },
                )
            ]
        )

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(tool,),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        ),
        tool_capture=capture,
    )

    attempt = await actor.run(
        SpecialistTaskInput(task_id="task-1", objective="Assess Apple revenue.")
    )

    assert provider_calls == 2
    assert attempt.finding.evidence_ids == ("evidence-1",)
    assert len(attempt.unavailability) == 1
    assert attempt.unavailability[0].reason is ToolUnavailableReason.SOURCE_UNREACHABLE
    assert capture.evidence == []
    assert capture.unavailability == []


@pytest.mark.asyncio
async def test_specialist_parallel_timeout_keeps_a_successful_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling_started = asyncio.Event()
    capture = SpecialistToolCapture()

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        assert source == "filing"
        if query == "unavailable coverage":
            await sibling_started.wait()
            await asyncio.sleep(1)
        assert query == "fallback coverage"
        sibling_started.set()
        return EvidenceEnvelope(
            id="evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            source="filing",
            source_url="https://example.test/filing",
            title="Annual filing",
            body="BODY-SENTINEL",
            excerpt="Fallback evidence succeeded.",
            as_of_date=date(2026, 9, 6),
        )

    tool = bind_evidence_tool(
        provider,
        context=_context().model_copy(
            update={
                "allowed_queries": frozenset(
                    {"unavailable coverage", "fallback coverage"}
                )
            }
        ),
        capture=capture,
    )
    monkeypatch.setattr(agent_evidence, "TOOL_TIMEOUT_SECONDS", 0.01)
    model_calls = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_evidence",
                        tool_call_id="call-unavailable",
                        args={"source": "filing", "query": "unavailable coverage"},
                    ),
                    ToolCallPart(
                        tool_name="read_evidence",
                        tool_call_id="call-fallback",
                        args={"source": "filing", "query": "fallback coverage"},
                    ),
                ]
            )
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert len(tool_returns) == 2
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "summary": "Fallback evidence succeeded.",
                        "evidence_ids": ["evidence-1"],
                    },
                )
            ]
        )

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(tool,),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        ),
        tool_capture=capture,
    )

    attempt = await asyncio.wait_for(
        actor.run(SpecialistTaskInput(task_id="task-1", objective="Assess.")),
        timeout=1,
    )

    assert attempt.finding.evidence_ids == ("evidence-1",)
    assert [item.id for item in attempt.evidence] == ["evidence-1"]
    assert [item.reason for item in attempt.unavailability] == [
        ToolUnavailableReason.CALL_TIMEOUT
    ]


@pytest.mark.asyncio
async def test_specialist_discards_tool_metadata_when_model_fails() -> None:
    class SourceUnreachable(Exception):
        pass

    capture = SpecialistToolCapture()

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise SourceUnreachable()

    tool = bind_evidence_tool(
        provider,
        context=_context(),
        capture=capture,
        expected_unavailability=(
            ExpectedToolUnavailability(
                exception_type=SourceUnreachable,
                reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
            ),
        ),
    )
    calls = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_evidence",
                        args={"source": "filing", "query": "excerpt"},
                    )
                ]
            )
        raise RuntimeError("model failed")

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(tool,),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        ),
        tool_capture=capture,
    )

    with pytest.raises(Exception):
        await actor.run(SpecialistTaskInput(task_id="task-1", objective="Assess."))

    assert actor.tool_capture.evidence == []
    assert actor.tool_capture.unavailability == []


@pytest.mark.asyncio
async def test_specialist_charges_a_provider_failure_without_a_usage_response() -> None:
    model_calls = 0

    async def lookup() -> str:
        return "available"

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="lookup", args={})])
        raise ModelHTTPError(429, "function")

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(lookup,),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )
    usage = RunUsage()

    with pytest.raises(SpecialistInvocationFailure) as raised:
        await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess."),
            usage=usage,
        )

    assert model_calls == 2
    assert usage.requests == 2
    assert raised.value.facts.boundary is SpecialistModelBoundary.UNKNOWN
    assert (
        classify_specialist_failure(raised.value.error, facts=raised.value.facts)
        is None
    )


@pytest.mark.asyncio
async def test_specialist_uses_the_active_model_override_for_retry_boundary() -> None:
    def overridden_model(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        del messages, info
        raise ModelHTTPError(429, "function")

    http_client = httpx.AsyncClient()
    provider = AzureProvider(
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-02-01",
        api_key="test-key",
        http_client=http_client,
    )
    agent = Agent(
        OpenAIChatModel("deployment", provider=provider),
        output_type=SpecialistFindingDraft,
        retries=0,
        tool_retries=0,
        output_retries=0,
        end_strategy="early",
    )
    actor = PydanticAISpecialistActor(agent)

    try:
        with agent.override(model=FunctionModel(overridden_model)):
            with pytest.raises(SpecialistInvocationFailure) as raised:
                await actor.run(
                    SpecialistTaskInput(task_id="task-1", objective="Assess.")
                )
    finally:
        await http_client.aclose()

    assert raised.value.facts.boundary is SpecialistModelBoundary.UNKNOWN
    assert (
        classify_specialist_failure(raised.value.error, facts=raised.value.facts)
        is None
    )


@pytest.mark.asyncio
async def test_specialist_retries_only_its_own_model_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def delayed_model(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        del messages, info
        await asyncio.sleep(1)
        raise AssertionError("model deadline should have cancelled this request")

    monkeypatch.setattr(specialist_module, "SPECIALIST_TIMEOUT_SECONDS", 0.01)
    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(delayed_model),
            output_type=SpecialistFindingDraft,
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )
    usage = RunUsage()

    with pytest.raises(SpecialistInvocationFailure) as raised:
        await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess."),
            usage=usage,
        )

    assert isinstance(raised.value.error, SpecialistModelRequestTimeout)
    assert usage.requests == 1
    assert (
        classify_specialist_failure(raised.value.error, facts=raised.value.facts)
        is RetryDisposition.RETRY
    )


@pytest.mark.asyncio
async def test_specialist_preserves_a_provider_timeout_as_fatal() -> None:
    def provider_timeout(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        del messages, info
        raise TimeoutError("provider timeout")

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(provider_timeout),
            output_type=SpecialistFindingDraft,
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )
    usage = RunUsage()

    with pytest.raises(SpecialistInvocationFailure) as raised:
        await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess."),
            usage=usage,
        )

    assert type(raised.value.error) is TimeoutError
    assert usage.requests == 1
    assert (
        classify_specialist_failure(raised.value.error, facts=raised.value.facts)
        is None
    )


@pytest.mark.asyncio
async def test_specialist_only_converts_proven_request_limit_exhaustion() -> None:
    def unexpected_usage_limit(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        del messages, info
        raise UsageLimitExceeded("unproven")

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(unexpected_usage_limit),
            output_type=SpecialistFindingDraft,
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )
    with pytest.raises(SpecialistInvocationFailure) as unexpected:
        await actor.run(SpecialistTaskInput(task_id="task-1", objective="Assess."))

    assert not unexpected.value.facts.count_limit_exhausted
    assert (
        classify_specialist_failure(
            unexpected.value.error, facts=unexpected.value.facts
        )
        is None
    )

    edge_usage = RunUsage(requests=11)
    with pytest.raises(SpecialistInvocationFailure) as edge:
        await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess."),
            usage=edge_usage,
        )

    assert edge_usage.requests == 12
    assert not edge.value.facts.count_limit_exhausted
    assert edge.value.facts.unreturned_model_requests == 1
    assert classify_specialist_failure(edge.value.error, facts=edge.value.facts) is None

    limit_actor = PydanticAISpecialistActor(
        Agent(
            TestModel(),
            output_type=SpecialistFindingDraft,
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )
    usage = RunUsage(requests=12)
    with pytest.raises(SpecialistInvocationFailure) as exhausted:
        await limit_actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess."),
            usage=usage,
            usage_limits=specialist_usage_limits(),
        )

    assert exhausted.value.facts.count_limit_exhausted
    assert (
        classify_specialist_failure(exhausted.value.error, facts=exhausted.value.facts)
        is RetryDisposition.TASK_FAILED
    )


@pytest.mark.asyncio
async def test_specialist_does_not_treat_a_parallel_business_usage_limit_as_a_gate() -> (
    None
):
    async def completed_tool() -> str:
        return "completed"

    async def business_limit_tool() -> str:
        raise UsageLimitExceeded("business quota")

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name="completed_tool", args={}),
                ToolCallPart(tool_name="business_limit_tool", args={}),
            ]
        )

    actor = PydanticAISpecialistActor(
        Agent(
            FunctionModel(model),
            output_type=SpecialistFindingDraft,
            tools=(completed_tool, business_limit_tool),
            retries=0,
            tool_retries=0,
            output_retries=0,
            end_strategy="early",
        )
    )
    usage = RunUsage(tool_calls=6)

    with pytest.raises(SpecialistInvocationFailure) as raised:
        await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess."),
            usage=usage,
            usage_limits=specialist_usage_limits(),
        )

    assert usage.tool_calls == 7
    assert not raised.value.facts.count_limit_exhausted
    assert (
        classify_specialist_failure(raised.value.error, facts=raised.value.facts)
        is None
    )
