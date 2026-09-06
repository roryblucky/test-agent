"""Public Evidence cache and report-gate coverage for Agent research."""

import pytest

from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    FinancialResearchReport,
    RequestEvidenceCatalog,
    bind_evidence_tool,
    prepare_synthesis,
    publish_report,
)


def _evidence(
    *,
    id: str = "evidence-1",
    body: str = "Apple revenue grew.",
    **overrides: str,
) -> EvidenceEnvelope:
    values = {
        "id": id,
        "tenant_id": "tenant-a",
        "request_id": "request-1",
        "task_id": "task-1",
        "source": "filing",
        "source_url": "https://example.test/filing",
        "title": "Annual filing",
        "body": body,
        "excerpt": "Apple revenue grew.",
    }
    return EvidenceEnvelope.model_validate({**values, **overrides})


def test_only_accepted_referenced_evidence_is_prepared_and_published() -> None:
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(), _evidence(id="orphan-1", body="Do not publish.")),
        finding_evidence_ids=("evidence-1",),
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
    )

    prepared = prepare_synthesis(
        standalone_query="Apple outlook",
        intent="market_outlook",
        accepted_evidence_ids=("evidence-1",),
        catalog=catalog,
        tenant_id="tenant-a",
        request_id="request-1",
    )
    published = publish_report(
        FinancialResearchReport(markdown="Apple grew. [[E:1]]"),
        prepared,
    )

    assert [item.id for item in prepared.evidence] == ["evidence-1"]
    assert published.citations[0].evidence_id == "evidence-1"
    assert published.citations[0].source == "filing"


def test_evidence_cache_is_idempotent_but_conflicts_and_unaccepted_ids_fail_closed() -> None:
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(),),
        finding_evidence_ids=("evidence-1",),
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
    )
    catalog.accept_referenced(
        (_evidence(),),
        finding_evidence_ids=("evidence-1",),
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
    )
    with pytest.raises(ValueError, match="Evidence body conflicts"):
        catalog.accept_referenced(
            (_evidence(body="Different body."),),
            finding_evidence_ids=("evidence-1",),
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
        )
    with pytest.raises(ValueError, match="Evidence is not accepted"):
        prepare_synthesis(
            standalone_query="Apple outlook",
            intent="market_outlook",
            accepted_evidence_ids=("missing",),
            catalog=catalog,
            tenant_id="tenant-a",
            request_id="request-1",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": "tenant-b"},
        {"request_id": "request-2"},
        {"task_id": "task-2"},
    ],
)
def test_evidence_cache_rejects_cross_scope_provenance(
    overrides: dict[str, str],
) -> None:
    catalog = RequestEvidenceCatalog()

    with pytest.raises(ValueError, match="Evidence provenance is not eligible"):
        catalog.accept_referenced(
            (_evidence(**overrides),),
            finding_evidence_ids=("evidence-1",),
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
        )


@pytest.mark.asyncio
async def test_evidence_tool_freezes_source_before_provider_access() -> None:
    calls: list[tuple[str, str]] = []

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        calls.append((source, query))
        return _evidence()

    tool = bind_evidence_tool(
        provider,
        allowed_sources=frozenset({"filing"}),
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
    )

    returned = await tool("filing", "Apple revenue")

    assert returned.return_value == {
        "evidence_id": "evidence-1",
        "excerpt": "Apple revenue grew.",
    }
    assert returned.metadata == _evidence()
    assert calls == [("filing", "Apple revenue")]
    with pytest.raises(ValueError, match="Evidence source is not eligible"):
        await tool("private", "Apple revenue")
    assert calls == [("filing", "Apple revenue")]


@pytest.mark.parametrize("markdown", ["Apple grew. [[E:x]]", "Apple grew. [[E:2]]"])
def test_report_gate_rejects_malformed_or_unknown_evidence_marker(markdown: str) -> None:
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(),),
        finding_evidence_ids=("evidence-1",),
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
    )
    prepared = prepare_synthesis(
        standalone_query="Apple outlook",
        intent="market_outlook",
        accepted_evidence_ids=("evidence-1",),
        catalog=catalog,
        tenant_id="tenant-a",
        request_id="request-1",
    )

    with pytest.raises(ValueError, match="Evidence marker"):
        publish_report(FinancialResearchReport(markdown=markdown), prepared)
