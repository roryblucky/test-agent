"""Intent recognition agent.

Uses a designated model (typically *fast* / *intent*) to classify the
user's query into one of the supported intent categories.
"""

from __future__ import annotations

from pydantic_ai import Agent

from app.core.model_registry import ModelRegistry
from app.models.domain import IntentResult
from app.models.workflow import IntentCatalogItem

# Runtime intent catalogs are tenant/domain business extensions only.
# Platform common intents live in agent instructions, not in per-run catalog input.
DEFAULT_INTENT_CATALOG: tuple[IntentCatalogItem, ...] = ()

DEFAULT_INSTRUCTIONS = """\
You are an intent classification assistant.
Given the user's query, classify it into one of the following intents:
- knowledge_query: The user wants to retrieve knowledge from the document base.
- chitchat: General conversation not requiring document retrieval.
- code_help: The user needs help with code or technical implementation.
- comparison: The user wants to compare two or more concepts.
- summarization: The user wants a summary of a topic.
- summarize_history: The user wants to summarize prior conversation.
- revise_previous: The user wants to revise or rewrite a prior answer.
- continue_previous: The user wants to continue a prior conversation thread.
- compare_previous: The user wants to compare against prior conversation content.

Return the intent, a confidence score (0-1), and optional sub-intents.
"""


def create_intent_agent(
    registry: ModelRegistry,
    model_name: str = "intent",
    instructions: str | None = None,
) -> Agent[None, IntentResult]:
    """Create an intent-recognition agent with the given model.

    Args:
        registry: Model registry for resolving model names.
        model_name: Named model from ``llmConfig.models``.
        instructions: Optional system prompt override.  If ``None``,
            falls back to the default intent-classification instructions.
    """
    from app.agents.history_processors import filter_thinking, trim_history

    return registry.create_agent(
        model_name,
        output_type=IntentResult,
        instructions=instructions or DEFAULT_INSTRUCTIONS,
        history_processors=[trim_history(20), filter_thinking()],
    )
