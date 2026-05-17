"""Question refinement agent.

Uses a *fast* model to rewrite and optimise the user's raw question
before retrieval.
"""

from __future__ import annotations

from pydantic_ai import Agent

from app.core.model_registry import ModelRegistry
from app.models.domain import RefinedQuestion

DEFAULT_INSTRUCTIONS = """\
You are a question refinement assistant for a knowledge management system.
Given the user's raw question, you must:
1. Rewrite it to be clearer and more specific.
2. Extract key search terms / keywords.
Return a structured result with `refined_query` and `keywords`.
Do NOT answer the question — only refine it for downstream retrieval.
"""


def create_refine_agent(
    registry: ModelRegistry,
    model_name: str = "fast",
    instructions: str | None = None,
) -> Agent[None, RefinedQuestion]:
    """Create a question-refinement agent with the given model.

    Args:
        registry: Model registry for resolving model names.
        model_name: Named model from ``llmConfig.models``.
        instructions: Optional system prompt override.  If ``None``,
            falls back to the default refine-question instructions.
    """
    from app.agents.history_processors import filter_thinking, trim_history

    return registry.create_agent(
        model_name,
        output_type=RefinedQuestion,
        instructions=instructions or DEFAULT_INSTRUCTIONS,
        history_processors=[trim_history(20), filter_thinking()],
    )
