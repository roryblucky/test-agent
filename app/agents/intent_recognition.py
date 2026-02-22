"""Intent recognition agent.

Uses a designated model (typically *fast* / *intent*) to classify the
user's query into one of the supported intent categories.
"""

from __future__ import annotations

from pydantic_ai import Agent

from app.core.model_registry import ModelRegistry
from app.models.domain import IntentResult

_INSTRUCTIONS = """\
You are an intent classification assistant.
Given the user's query, classify it into one of the following intents:
- knowledge_query: The user wants to retrieve knowledge from the document base.
- chitchat: General conversation not requiring document retrieval.
- code_help: The user needs help with code or technical implementation.
- comparison: The user wants to compare two or more concepts.
- summarization: The user wants a summary of a topic.

Return the intent, a confidence score (0-1), and optional sub-intents.
"""


def create_intent_agent(
    registry: ModelRegistry,
    model_name: str = "intent",
) -> Agent[None, IntentResult]:
    """Create an intent-recognition agent with the given model."""
    from app.agents.history_processors import filter_thinking, trim_history

    return registry.create_agent(
        model_name,
        output_type=IntentResult,
        instructions=_INSTRUCTIONS,
        history_processors=[trim_history(20), filter_thinking()],
    )
