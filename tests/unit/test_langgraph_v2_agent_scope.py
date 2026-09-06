"""Trusted Agent intent-to-scope policy coverage."""

from app.langgraph_v2.agent_scope import (
    AgentIntentPolicy,
    SpecialistDescriptor,
    resolve_research_scope,
)
from app.models.workflow import IntentResult


def test_scope_uses_only_trusted_policy_not_model_metadata() -> None:
    policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
        specialist_descriptors=(
            SpecialistDescriptor(id="market-data", description="Market data"),
            SpecialistDescriptor(id="news", description="News"),
        ),
    )
    intent = IntentResult(
        intent="market_outlook",
        confidence=0.9,
        metadata={
            "tools": ["exfiltrate"],
            "skills": ["unsafe"],
            "sources": ["private"],
            "filters": {"expand": True},
        },
    )

    scope = resolve_research_scope(intent, {policy.intent: policy})

    assert scope.intent == "market_outlook"
    assert scope.specialist_descriptors == policy.specialist_descriptors
    assert not hasattr(scope, "tools")


def test_scope_rejects_unknown_intent() -> None:
    policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
        specialist_descriptors=(),
    )

    try:
        resolve_research_scope(
            IntentResult(intent="made-up", confidence=0.9),
            {policy.intent: policy},
        )
    except ValueError as error:
        assert str(error) == "Agent Intent is not configured"
    else:
        raise AssertionError("unknown Intent must fail closed")
