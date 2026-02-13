"""RAG answer generation agent.

Uses a *pro* model to synthesise an answer from retrieved context.
"""

from __future__ import annotations

from pydantic_ai import Agent

from app.core.model_registry import ModelRegistry

_INSTRUCTIONS = """\
You are a knowledgeable assistant for an enterprise knowledge management system.
Answer the user's question **strictly based on the provided reference documents**.
- If the documents do not contain enough information, say so honestly.
- Cite which document(s) you used.
- Be concise but thorough.
"""


def create_rag_answer_agent(
    registry: ModelRegistry,
    model_name: str = "pro",
) -> Agent[None, str]:
    """Create a RAG answer agent that produces a text response."""
    return registry.create_agent(
        model_name,
        output_type=str,
        instructions=_INSTRUCTIONS,
    )
