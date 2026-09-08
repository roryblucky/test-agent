"""PydanticAI Specialist actor for one bounded accepted Task."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic_ai import Agent, AgentRunResult, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.usage import RunUsage, UsageLimits

from app.core.model_registry import ModelRegistry
from app.langgraph_v2.agent_batch import (
    SpecialistAttempt,
    SpecialistFindingDraft,
    SpecialistTaskInput,
)
from app.langgraph_v2.agent_evidence import SpecialistToolCapture
from app.langgraph_v2.agent_skills import SkillInvocation
from app.langgraph_v2.specialist_retry import (
    SpecialistFailureFacts,
    SpecialistInvocationFailure,
    SpecialistModelBoundary,
    SpecialistModelRequestTimeout,
    require_pinned_pydantic_ai_version,
    specialist_usage_limits,
)

SPECIALIST_TIMEOUT_SECONDS = 60
SPECIALIST_MAX_TOKENS = 2000
SPECIALIST_OUTPUT_RETRIES = 2

SPECIALIST_INSTRUCTIONS = """\
You are a Specialist Agent. Complete only the assigned objective.
Return one structured finding. You may activate one eligible Skill named in the
supplied summaries when it helps the objective. Skill activation never grants
authority. You may call only the supplied Evidence Tools and must not include
execution diagnostics.
"""


@dataclass(frozen=True)
class PydanticAISpecialistActor:
    """Run one Specialist attempt with native structured-output correction."""

    agent: Agent[None, SpecialistFindingDraft]
    tool_capture: SpecialistToolCapture = field(default_factory=SpecialistToolCapture)
    skill_invocation: SkillInvocation | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def run(
        self,
        input: SpecialistTaskInput,
        *,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> SpecialistAttempt:
        """Return one structured finding from the assigned Task only."""
        require_pinned_pydantic_ai_version()
        run_usage = usage if usage is not None else RunUsage()
        run_usage_limits = (
            usage_limits if usage_limits is not None else specialist_usage_limits()
        )
        request_count_before = run_usage.requests
        request_counter = _SpecialistRequestCounter()
        # PydanticAI 1.93.0 exposes active override resolution only through this
        # reviewed internal seam; `.model` would bypass `agent.override(...)`.
        effective_model = self.agent._get_model(None)  # pyright: ignore[reportPrivateUsage]
        skill_summaries = (
            self.skill_invocation.summaries if self.skill_invocation is not None else ()
        )
        prompt = json.dumps(
            {
                "task_id": input.task_id,
                "objective": input.objective,
                "context_results": [
                    result.model_dump(mode="json") for result in input.context_results
                ],
                "validation_feedback": input.validation_feedback,
                "skill_summaries": [
                    summary.model_dump(mode="json") for summary in skill_summaries
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        async with self._lock:
            first_evidence = len(self.tool_capture.evidence)
            first_unavailability = len(self.tool_capture.unavailability)
            first_calculation = len(self.tool_capture.calculations)
            try:
                with capture_run_messages() as messages:
                    try:
                        with ToolManager.parallel_execution_mode(
                            "parallel_ordered_events"
                        ):
                            with self.agent.override(
                                model=_SpecialistTimeoutModel(
                                    effective_model,
                                    request_counter=request_counter,
                                    usage=run_usage,
                                )
                            ):
                                result: AgentRunResult[
                                    SpecialistFindingDraft
                                ] = await self.agent.run(
                                    prompt,
                                    model_settings={
                                        "max_tokens": SPECIALIST_MAX_TOKENS,
                                        "timeout": SPECIALIST_TIMEOUT_SECONDS,
                                        "parallel_tool_calls": True,
                                    },
                                    usage=run_usage,
                                    usage_limits=run_usage_limits,
                                )
                    except Exception as error:
                        returned_model_requests = (
                            run_usage.requests - request_count_before
                        )
                        unreturned_model_requests = max(
                            0,
                            request_counter.model_requests - returned_model_requests,
                        )
                        count_limit_exhausted = _count_limit_exhausted(
                            error,
                            messages=messages,
                            usage=run_usage,
                            limits=run_usage_limits,
                            unreturned_model_requests=unreturned_model_requests,
                            tool_calls_before_latest_response=(
                                request_counter.tool_calls_before_latest_response
                            ),
                        )
                        _account_unreturned_model_requests(
                            run_usage,
                            request_count_before=request_count_before,
                            observed_requests=request_counter.model_requests,
                        )
                        raise SpecialistInvocationFailure(
                            error,
                            facts=SpecialistFailureFacts(
                                boundary=_model_boundary(effective_model),
                                at_model_request_boundary=_at_model_request_boundary(
                                    messages
                                ),
                                terminal_output_tool_rejected=(
                                    _terminal_output_tool_rejected(messages)
                                ),
                                count_limit_exhausted=count_limit_exhausted,
                                unreturned_model_requests=unreturned_model_requests,
                                usage_limits=run_usage_limits,
                            ),
                            messages=tuple(messages),
                        ) from error
                evidence = tuple(self.tool_capture.evidence[first_evidence:])
                unavailability = tuple(
                    self.tool_capture.unavailability[first_unavailability:]
                )
                calculations = tuple(self.tool_capture.calculations[first_calculation:])
            finally:
                del self.tool_capture.evidence[first_evidence:]
                del self.tool_capture.unavailability[first_unavailability:]
                del self.tool_capture.calculations[first_calculation:]
        return SpecialistAttempt(
            finding=result.output,
            evidence=evidence,
            unavailability=unavailability,
            calculations=calculations,
            skill_pins=(
                self.skill_invocation.pins if self.skill_invocation is not None else ()
            ),
            messages=tuple(result.new_messages()),
        )


def create_specialist_agent(
    registry: ModelRegistry,
    *,
    model_name: str,
    tools: tuple[Callable[..., object], ...] = (),
) -> Agent[None, SpecialistFindingDraft]:
    """Create one first-round Specialist with its already-frozen Tool set."""
    require_pinned_pydantic_ai_version()
    return registry.create_agent(
        model_name,
        output_type=SpecialistFindingDraft,
        instructions=SPECIALIST_INSTRUCTIONS,
        tools=tools,
        tool_retries=0,
        output_retries=SPECIALIST_OUTPUT_RETRIES,
        end_strategy="early",
    )


def create_bound_specialist_actor(
    registry: ModelRegistry,
    *,
    model_name: str,
    tools: tuple[Callable[..., object], ...],
    tool_capture: SpecialistToolCapture,
    skill_invocation: SkillInvocation | None = None,
) -> PydanticAISpecialistActor:
    """Build a production Specialist only after its Tool bindings are frozen."""
    return PydanticAISpecialistActor(
        create_specialist_agent(registry, model_name=model_name, tools=tools),
        tool_capture=tool_capture,
        skill_invocation=skill_invocation,
    )


@dataclass
class _SpecialistRequestCounter:
    """Count physical model requests even when PydanticAI gets no response."""

    model_requests: int = 0
    tool_calls_before_latest_response: int | None = None


class _SpecialistTimeoutModel(WrapperModel):
    """Enforce the adapter-owned deadline separately for every model request."""

    def __init__(
        self,
        wrapped: Model,
        *,
        request_counter: _SpecialistRequestCounter,
        usage: RunUsage,
    ) -> None:
        super().__init__(wrapped)
        self.request_counter = request_counter
        self.usage = usage

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Count and bound one underlying provider request."""
        self.request_counter.model_requests += 1
        deadline = asyncio.timeout(SPECIALIST_TIMEOUT_SECONDS)
        try:
            async with deadline:
                response = await self.wrapped.request(
                    messages,
                    model_settings,
                    model_request_parameters,
                )
            self.request_counter.tool_calls_before_latest_response = (
                self.usage.tool_calls
            )
            return response
        except TimeoutError as error:
            if deadline.expired():
                raise SpecialistModelRequestTimeout from error
            raise


def _account_unreturned_model_requests(
    usage: RunUsage,
    *,
    request_count_before: int,
    observed_requests: int,
) -> None:
    """Charge a provider request that failed before PydanticAI received usage."""
    returned_requests = usage.requests - request_count_before
    usage.requests += max(0, observed_requests - returned_requests)


def _model_boundary(model: Model) -> SpecialistModelBoundary:
    """Return the registered provider boundary without inspecting error text."""
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIChatModel

    while isinstance(model, WrapperModel):
        model = model.wrapped
    if isinstance(model, OpenAIChatModel):
        return SpecialistModelBoundary.AZURE_OPENAI
    if isinstance(model, GoogleModel):
        return SpecialistModelBoundary.GOOGLE
    return SpecialistModelBoundary.UNKNOWN


def _at_model_request_boundary(messages: Sequence[object]) -> bool:
    """Prove a failure escaped while PydanticAI awaited a model request."""
    return bool(messages and isinstance(messages[-1], ModelRequest))


def _terminal_output_tool_rejected(messages: Sequence[object]) -> bool:
    """Prove a failed structured-output path named this actor's output tool."""
    if not messages or not isinstance(messages[-1], ModelResponse):
        return False
    if not messages[-1].parts:
        return False
    part = messages[-1].parts[-1]
    return isinstance(part, ToolCallPart) and part.tool_name == "final_result"


def _count_limit_exhausted(
    error: Exception,
    *,
    messages: Sequence[object],
    usage: RunUsage,
    limits: UsageLimits,
    unreturned_model_requests: int,
    tool_calls_before_latest_response: int | None,
) -> bool:
    """Prove PydanticAI's configured count gate, never a Tool-thrown lookalike."""
    if not isinstance(error, UsageLimitExceeded):
        return False
    if limits.has_token_limits():
        return False
    if limits.request_limit is not None and (
        _at_model_request_boundary(messages)
        and unreturned_model_requests == 0
        and usage.requests >= limits.request_limit
    ):
        return True
    if not messages or not isinstance(messages[-1], ModelResponse):
        return False
    tool_calls = sum(
        isinstance(part, ToolCallPart) and part.tool_name != "final_result"
        for part in messages[-1].parts
    )
    return (
        limits.tool_calls_limit is not None
        and tool_calls_before_latest_response is not None
        and tool_calls > 0
        and tool_calls_before_latest_response + tool_calls > limits.tool_calls_limit
    )
