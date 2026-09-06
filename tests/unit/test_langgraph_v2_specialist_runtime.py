"""No-Tool Specialist PydanticAI actor coverage."""

import pydantic_ai.models as models
import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from app.agents.specialist import (
    SPECIALIST_MAX_TOKENS,
    SPECIALIST_TIMEOUT_SECONDS,
    PydanticAISpecialistActor,
)
from app.langgraph_v2.agent_batch import SpecialistFindingDraft, SpecialistTaskInput


@pytest.fixture(autouse=True)
def disable_real_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep all Specialist actor coverage local to TestModel/FunctionModel."""
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


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

    assert finding == SpecialistFindingDraft(summary="No-tool finding")
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
    with capture_run_messages() as messages:
        finding = await actor.run(
            SpecialistTaskInput(task_id="task-1", objective="Assess market outlook.")
        )

    assert finding == SpecialistFindingDraft(summary="No-tool finding")
    assert len(captures) == 1
    assert captures[0][1].function_tools == []
    assert captures[0][1].model_settings == {"max_tokens": SPECIALIST_MAX_TOKENS}
    assert len(messages) == 3
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].usage.requests == 1
