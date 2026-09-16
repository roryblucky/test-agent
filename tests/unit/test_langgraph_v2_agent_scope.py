"""Trusted Agent intent-to-scope policy coverage."""

from app.langgraph_v2.agent_scope import (
    AgentIntentPolicy,
    SpecialistDescriptor,
    resolve_research_scope,
)
from app.models.workflow import IntentResult


def test_two_intents_share_catalog_specialists_but_keep_distinct_data_scope() -> None:
    descriptors = (
        SpecialistDescriptor(id="market-data", description="Market data"),
        SpecialistDescriptor(id="news", description="News"),
    )
    market_policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
        allowed_tool_ids=frozenset({"market-search"}),
        allowed_skill_names=frozenset({"filing-analysis"}),
        allowed_sources=frozenset({"filing"}),
        allowed_queries=frozenset({"Apple revenue"}),
        max_evidence_age_days=7,
    )
    legal_policy = AgentIntentPolicy(
        intent="legal_risk",
        description="Assess legal risk.",
        allowed_tool_ids=frozenset({"legal-search"}),
        allowed_sources=frozenset({"court-record"}),
        allowed_queries=frozenset({"Apple litigation"}),
        max_evidence_age_days=30,
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

    policies = {
        market_policy.intent: market_policy,
        legal_policy.intent: legal_policy,
    }
    market_scope = resolve_research_scope(
        intent,
        policies,
        specialist_descriptors=descriptors,
    )
    legal_scope = resolve_research_scope(
        IntentResult(intent="legal_risk", confidence=0.9),
        policies,
        specialist_descriptors=descriptors,
    )

    assert market_scope.specialist_descriptors == descriptors
    assert legal_scope.specialist_descriptors == descriptors
    assert market_scope.allowed_tool_ids == frozenset({"market-search"})
    assert legal_scope.allowed_tool_ids == frozenset({"legal-search"})
    assert market_scope.allowed_sources == frozenset({"filing"})
    assert legal_scope.allowed_sources == frozenset({"court-record"})
    assert market_scope.allowed_queries == frozenset({"Apple revenue"})
    assert legal_scope.allowed_queries == frozenset({"Apple litigation"})
    assert market_scope.max_evidence_age_days == 7
    assert legal_scope.max_evidence_age_days == 30
    assert market_scope.allowed_skill_names == frozenset({"filing-analysis"})
    assert not hasattr(market_scope, "tools")


def test_scope_rejects_unknown_intent() -> None:
    policy = AgentIntentPolicy(
        intent="market_outlook",
        description="Assess market conditions.",
    )

    try:
        resolve_research_scope(
            IntentResult(intent="made-up", confidence=0.9),
            {policy.intent: policy},
            specialist_descriptors=(),
        )
    except ValueError as error:
        assert str(error) == "Agent Intent is not configured"
    else:
        raise AssertionError("unknown Intent must fail closed")
