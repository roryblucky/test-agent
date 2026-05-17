"""Unit tests for AgentHandler mode dispatch."""

from collections.abc import Callable
from typing import Any, Self
from unittest.mock import MagicMock

import pytest
from pydantic_ai.usage import RunUsage

from app.config.models import (
    AgentConfig,
    FlowConfig,
    FlowStep,
    FlowStepType,
    LLMConfig,
    TenantConfig,
)
from app.models.workflow import (
    IntentResult,
    PlannerOutput,
    ToolCallRecord,
    ToolObservation,
)
from app.services.flow_context import FlowContext
from app.services.handlers.agent import AgentHandler
from app.services.tenant_manager import TenantProviders
from app.skills.schema import SkillDefinition, SkillMetadata


class FakeAgentStream:
    """Async context manager that mimics the small stream API used by AgentHandler."""

    def __init__(
        self,
        output: Any,
        chunks: list[Any] | None = None,
        on_enter: Callable[[Any], None] | None = None,
        deps: Any | None = None,
    ) -> None:
        self._output = output
        self._chunks = chunks or []
        self._on_enter = on_enter
        self._deps = deps

    async def __aenter__(self) -> Self:
        if self._on_enter:
            self._on_enter(self._deps)
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def stream_output(self, debounce_by: float = 0.01):
        for chunk in self._chunks:
            yield chunk

    async def get_output(self) -> Any:
        return self._output

    def usage(self) -> RunUsage:
        return RunUsage()

    def new_messages(self) -> list[Any]:
        return []


class FakeAgent:
    """Fake Pydantic AI agent for handler tests."""

    def __init__(
        self,
        output: Any,
        chunks: list[Any] | None = None,
        on_enter: Callable[[Any], None] | None = None,
    ) -> None:
        self.output = output
        self.chunks = chunks or []
        self.on_enter = on_enter
        self.last_prompt: str | None = None
        self.last_deps: Any | None = None
        self.last_message_history: Any | None = None

    def run_stream(self, prompt: str, *, deps: Any, message_history: Any = None):
        self.last_prompt = prompt
        self.last_deps = deps
        self.last_message_history = message_history
        return FakeAgentStream(
            self.output,
            chunks=self.chunks,
            on_enter=self.on_enter,
            deps=deps,
        )


def _tenant_config() -> TenantConfig:
    return TenantConfig(
        kmsAppName="Agent Mode Test App",
        applicationId="agent-mode-test",
        adGroups=["group1"],
        llm_config=LLMConfig(models={}),
        flow_config=FlowConfig(),
    )


def _handler() -> AgentHandler:
    return AgentHandler(
        registry=MagicMock(),
        providers=TenantProviders(),
        cfg=_tenant_config(),
        skill_registry=None,
    )


def _skill(
    name: str,
    *,
    allowed_tools: list[str],
    required_tools: list[str],
) -> SkillDefinition:
    return SkillDefinition(
        metadata=SkillMetadata(
            name=name,
            description=f"{name} skill",
            allowed_tools=allowed_tools,
            required_tools=required_tools,
        ),
        instructions="Use required tools.",
        tenant_id="agent-mode-test",
        source_path=f"/skills/{name}/SKILL.md",
    )


@pytest.mark.asyncio
async def test_agent_without_mode_defaults_to_supervisor(mock_emitter) -> None:
    """Missing agent mode preserves the old supervisor-style final answer path."""
    handler = _handler()
    agent_config = AgentConfig(llmType="fast")
    fake_agent = FakeAgent(
        output="final answer",
        chunks=["final", "final answer"],
    )
    handler._agent_cache[
        handler._cache_key("supervisor", agent_config)
    ] = fake_agent

    ctx = FlowContext(query="answer this", emitter=mock_emitter)
    step = FlowStep(
        type=FlowStepType.AGENT,
        agentConfig=agent_config,
    )

    result = await handler.handle(ctx, step)

    assert result.llm_response == "final answer"
    assert result.planner_output is None
    assert fake_agent.last_prompt == "answer this"
    assert fake_agent.last_deps.flow_context is ctx
    assert fake_agent.last_deps.tenant_id == "agent-mode-test"
    mock_emitter.emit_token.assert_any_await("final")
    mock_emitter.emit_step_completed.assert_any_await(
        "agent:orchestration",
        {"output_length": len("final answer")},
    )


@pytest.mark.asyncio
async def test_agent_planner_writes_planner_output_without_final_answer(
    mock_emitter,
) -> None:
    """Planner mode writes PlannerOutput and does not populate llm_response."""
    handler = _handler()
    agent_config = AgentConfig(llmType="fast")

    def _record_runtime_context(deps: Any) -> None:
        deps.activated_skill_names.append("generic-search")
        deps.flow_context.tool_calls.append(
            ToolCallRecord(
                tool_name="search_documents",
                input_payload={"query": "find evidence"},
                status="success",
                output_evidence_ids=["ev1"],
            )
        )
        deps.flow_context.tool_observations.append(
            ToolObservation(
                tool_name="search_documents",
                status="success",
                evidence_ids=["ev1"],
            )
        )

    fake_agent = FakeAgent(
        output=PlannerOutput(
            can_synthesize=True,
            reason="Evidence is available.",
        ),
        on_enter=_record_runtime_context,
    )
    handler._agent_cache[handler._cache_key("planner", agent_config)] = fake_agent

    ctx = FlowContext(
        query="find evidence",
        emitter=mock_emitter,
        message_history=["previous user turn"],
    )
    ctx.intent = IntentResult(
        intent="knowledge_query",
        confidence=0.91,
        candidate_skills=["generic-search"],
    )
    step = FlowStep(
        type=FlowStepType.AGENT,
        mode="planner",
        agentConfig=agent_config,
    )

    result = await handler.handle(ctx, step)

    assert result.llm_response is None
    assert fake_agent.last_message_history is None
    assert '"original_query": "find evidence"' in (fake_agent.last_prompt or "")
    assert '"standalone_query": "find evidence"' in (fake_agent.last_prompt or "")
    assert '"intent": "knowledge_query"' in (fake_agent.last_prompt or "")
    assert '"candidate_skills": [' in (fake_agent.last_prompt or "")
    assert result.planner_output == PlannerOutput(
        can_synthesize=True,
        reason="Evidence is available.",
        active_skills=["generic-search"],
        used_tools=["search_documents"],
        evidence_ids=["ev1"],
    )
    assert result.active_skills == ["generic-search"]
    mock_emitter.emit_token.assert_not_awaited()
    mock_emitter.emit_step_completed.assert_any_await(
        "agent:planner",
        {
            "can_synthesize": True,
            "evidence_count": 1,
            "missing_evidence_count": 0,
            "used_tools": ["search_documents"],
        },
    )


@pytest.mark.asyncio
async def test_agent_planner_reports_required_tool_missing(
    mock_emitter,
) -> None:
    """Planner output fails closed when an activated skill requires unavailable tools."""
    skill = _skill(
        "required-search",
        allowed_tools=["search_documents", "rank_documents"],
        required_tools=["search_documents", "rank_documents"],
    )
    skill_registry = MagicMock()
    skill_registry.get_activated_skill.return_value = skill
    handler = AgentHandler(
        registry=MagicMock(),
        providers=TenantProviders(),
        cfg=_tenant_config(),
        skill_registry=skill_registry,
    )
    agent_config = AgentConfig(
        llmType="fast",
        buildInTools=["search_documents"],
    )

    def _activate_skill(deps: Any) -> None:
        deps.activated_skill_names.append("required-search")

    fake_agent = FakeAgent(
        output=PlannerOutput(
            active_skills=["required-search"],
            can_synthesize=True,
            reason="Initial plan is ready.",
        ),
        on_enter=_activate_skill,
    )
    handler._agent_cache[handler._cache_key("planner", agent_config)] = fake_agent

    ctx = FlowContext(query="find evidence", emitter=mock_emitter)
    step = FlowStep(
        type=FlowStepType.AGENT,
        mode="planner",
        agentConfig=agent_config,
    )

    result = await handler.handle(ctx, step)

    assert result.planner_output is not None
    assert result.planner_output.can_synthesize is False
    assert result.planner_output.required_tools_missing == ["rank_documents"]
    assert "Required tools unavailable: rank_documents." in (
        result.planner_output.reason
    )


@pytest.mark.asyncio
async def test_agent_unknown_mode_fails_closed(mock_emitter) -> None:
    """Unknown agent modes fail before any fallback behavior runs."""
    handler = _handler()
    ctx = FlowContext(query="test", emitter=mock_emitter)
    step = FlowStep(
        type=FlowStepType.AGENT,
        mode="unknown",
        agentConfig=AgentConfig(llmType="fast"),
    )

    with pytest.raises(ValueError, match="Unknown agent mode"):
        await handler.handle(ctx, step)


def test_agent_warmup_cache_key_includes_mode(monkeypatch) -> None:
    """Warmup builds separate cached agents for supervisor and planner."""
    handler = _handler()
    built_modes: list[str] = []

    def _fake_build(agent_config: AgentConfig, mode: str) -> FakeAgent:
        built_modes.append(mode)
        return FakeAgent(output="ok")

    monkeypatch.setattr(handler, "_build_tenant_agent", _fake_build)
    agent_config = AgentConfig(llmType="fast")

    handler.warmup(
        [
            FlowStep(type=FlowStepType.AGENT, agentConfig=agent_config),
            FlowStep(type=FlowStepType.AGENT, mode="planner", agentConfig=agent_config),
        ]
    )

    assert built_modes == ["supervisor", "planner"]
    assert handler._cache_key("supervisor", agent_config) in handler._agent_cache
    assert handler._cache_key("planner", agent_config) in handler._agent_cache
