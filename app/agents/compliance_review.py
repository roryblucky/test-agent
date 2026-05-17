"""Compliance review agent for buffered answer release.

Architecture (informed by pydantic-ai best practices):

- **Static ``instructions``** (reviewer identity, guardrails,
  tenant/domain contracts) are set at Agent build time for caching.
- **Dynamic ``@agent.instructions``** appends per-request data
  (draft answer + evidence) and is NOT retained in ``message_history``.

We use ``instructions`` (not ``system_prompt``) per pydantic-ai guidance:
``system_prompt`` content persists in message history, which would cause
draft answers from earlier reviews to leak into subsequent requests.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from app.core.model_registry import ModelRegistry
from app.models.workflow import ComplianceReviewResult


@dataclass
class ComplianceReviewDeps:
    """Dependencies for compliance review.

    ``reference_data`` carries the draft answer and evidence to review.
    """

    reference_data: str


def create_compliance_review_agent(
    registry: ModelRegistry,
    model_name: str = "fast",
    instructions: str | None = None,
) -> Agent[ComplianceReviewDeps, ComplianceReviewResult]:
    """Create a compliance review agent with cacheable static instructions.

    Args:
        registry: Model registry for resolving model names.
        model_name: Named model from ``llmConfig.models``.
        instructions: Static system prompt (reviewer identity + guardrails +
            contracts).  Set at build time for API-level caching.
    """
    from app.agents.history_processors import filter_thinking, trim_history

    agent = registry.create_agent(
        model_name,
        output_type=ComplianceReviewResult,
        deps_type=ComplianceReviewDeps,
        instructions=instructions,
        history_processors=[trim_history(10), filter_thinking()],
    )

    @agent.instructions
    def inject_review_data(ctx: RunContext[ComplianceReviewDeps]) -> str:
        data = ctx.deps.reference_data
        if data:
            return f"<reference_data>\n{data}\n</reference_data>"
        return ""

    return agent
