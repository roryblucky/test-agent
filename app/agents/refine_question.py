"""Question refinement agent.

Uses a *fast* model to rewrite and optimise the user's raw question
before retrieval.
"""

from __future__ import annotations

from pydantic_ai import Agent

from app.core.model_registry import ModelRegistry
from app.models.domain import RefinedQuestion

_INSTRUCTIONS = """\
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
) -> Agent[None, RefinedQuestion]:
    """Create a question-refinement agent with the given model."""
    return registry.create_agent(
        model_name,
        output_type=RefinedQuestion,
        instructions=_INSTRUCTIONS,
    )
