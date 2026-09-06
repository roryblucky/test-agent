"""Public Evidence cache and report-gate coverage for Agent research."""

import asyncio
import json
from datetime import UTC, date, datetime

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

import app.langgraph_v2.agent_evidence as agent_evidence
from app.langgraph_v2.agent_evidence import (
    EvidenceEnvelope,
    EvidenceInvocationContext,
    ExpectedToolUnavailability,
    FinancialResearchReport,
    RequestEvidenceCatalog,
    SpecialistToolCapture,
    ToolUnavailable,
    ToolUnavailableReason,
    bind_evidence_tool,
    prepare_synthesis,
    publish_report,
)


def _context() -> EvidenceInvocationContext:
    return EvidenceInvocationContext(
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
        allowed_tool_ids=frozenset({"filing-reader"}),
        allowed_sources=frozenset({"filing"}),
        allowed_queries=frozenset({"Apple revenue"}),
    )


def _evidence(
    *,
    id: str = "evidence-1",
    body: str = "Apple revenue grew.",
    **overrides: object,
) -> EvidenceEnvelope:
    values: dict[str, object] = {
        "id": id,
        "tenant_id": "tenant-a",
        "request_id": "request-1",
        "task_id": "task-1",
        "source": "filing",
        "source_url": "https://example.test/filing",
        "title": "Annual filing",
        "body": body,
        "excerpt": "Apple revenue grew.",
        "as_of_date": date(2026, 9, 6),
    }
    return EvidenceEnvelope.model_validate({**values, **overrides})


def _tool_context(
    *, tool_call_id: str = "call-1", tool_name: str = "read_evidence"
) -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        tool_call_id=tool_call_id,
        tool_name=tool_name,
    )


@pytest.mark.asyncio
async def test_expected_tool_unavailability_becomes_bounded_model_data() -> None:
    class SourceUnreachable(Exception):
        pass

    capture = SpecialistToolCapture()

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise SourceUnreachable("provider details must not escape")

    tool = bind_evidence_tool(
        provider,
        context=_context(),
        tool_id="filing-reader",
        expected_unavailability=(
            ExpectedToolUnavailability(
                exception_type=SourceUnreachable,
                reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
            ),
        ),
        capture=capture,
        now=lambda: datetime(2026, 9, 6, 12, tzinfo=UTC),
    )

    returned = await tool(
        _tool_context(tool_name="filing-reader"), "filing", "Apple revenue"
    )

    assert returned.return_value == ToolUnavailable(
        reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
        requested_coverage="Apple revenue",
    )
    assert len(capture.unavailability) == 1
    assert capture.unavailability[0].model_dump() == {
        "id": capture.unavailability[0].id,
        "tenant_id": "tenant-a",
        "request_id": "request-1",
        "task_id": "task-1",
        "attempt": 1,
        "tool_call_id": "call-1",
        "tool_id": "filing-reader",
        "source": "filing",
        "observed_at": datetime(2026, 9, 6, 12, tzinfo=UTC),
        "reason": ToolUnavailableReason.SOURCE_UNREACHABLE,
        "requested_coverage": "Apple revenue",
    }


def test_only_accepted_referenced_evidence_is_prepared_and_published() -> None:
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(), _evidence(id="orphan-1", body="Do not publish.")),
        finding_evidence_ids=("evidence-1",),
        context=_context(),
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
        FinancialResearchReport(markdown_report="Apple grew. [[E:1]]"),
        prepared,
    )

    assert [item.id for item in prepared.evidence] == ["evidence-1"]
    assert published.citations[0].evidence_id == "evidence-1"
    assert published.citations[0].source == "filing"


def test_evidence_cache_is_idempotent_but_conflicts_and_unaccepted_ids_fail_closed() -> (
    None
):
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(),),
        finding_evidence_ids=("evidence-1",),
        context=_context(),
    )
    catalog.accept_referenced(
        (_evidence(),),
        finding_evidence_ids=("evidence-1",),
        context=_context(),
    )
    with pytest.raises(ValueError, match="Evidence body conflicts"):
        catalog.accept_referenced(
            (_evidence(body="Different body."),),
            finding_evidence_ids=("evidence-1",),
            context=_context(),
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


def test_evidence_cache_ignores_noncanonical_raw_payload_for_idempotence() -> None:
    catalog = RequestEvidenceCatalog()
    first = _evidence(raw_provider_payload="provider-response-a")
    duplicate = _evidence(raw_provider_payload="provider-response-b")

    for evidence in (first, duplicate):
        catalog.accept_referenced(
            (evidence,),
            finding_evidence_ids=("evidence-1",),
            context=_context(),
        )

    accepted = catalog.resolve(
        "evidence-1", tenant_id="tenant-a", request_id="request-1"
    )
    assert accepted.body == first.body


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
            context=_context(),
        )


@pytest.mark.asyncio
async def test_evidence_tool_freezes_source_before_provider_access() -> None:
    calls: list[tuple[str, str]] = []

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        calls.append((source, query))
        return _evidence()

    tool = bind_evidence_tool(
        provider,
        context=_context(),
    )

    returned = await tool(_tool_context(), "filing", "Apple revenue")

    assert returned.return_value == {
        "evidence_id": "evidence-1",
        "excerpt": "Apple revenue grew.",
    }
    assert returned.metadata == _evidence()
    assert calls == [("filing", "Apple revenue")]
    with pytest.raises(ValueError, match="Evidence source is not eligible"):
        await tool(_tool_context(), "private", "Apple revenue")
    assert calls == [("filing", "Apple revenue")]


@pytest.mark.asyncio
async def test_evidence_tool_projects_oversized_success_as_unavailable() -> None:
    capture = SpecialistToolCapture()

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        return _evidence(excerpt="🐍" * 2048)

    tool = bind_evidence_tool(
        provider,
        context=_context(),
        capture=capture,
    )

    returned = await tool(_tool_context(), "filing", "Apple revenue")

    assert returned.return_value == ToolUnavailable(
        reason=ToolUnavailableReason.RESPONSE_UNUSABLE,
        requested_coverage="Apple revenue",
    )
    assert capture.evidence == []
    assert len(capture.unavailability) == 1


@pytest.mark.asyncio
async def test_unavailable_return_keeps_the_4_kib_boundary_and_projects_oversize_gap() -> (
    None
):
    class SourceUnreachable(Exception):
        pass

    template = ToolUnavailable(
        reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
        requested_coverage="x",
    )
    exact_coverage = "x" * (
        4 * 1024
        - len(
            json.dumps(
                template.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        + 1
    )
    too_large_coverage = f"{exact_coverage}x"
    assert len(
        json.dumps(
            ToolUnavailable(
                reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
                requested_coverage=exact_coverage,
            ).model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ) == 4 * 1024
    assert len(
        json.dumps(
            ToolUnavailable(
                reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
                requested_coverage=too_large_coverage,
            ).model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ) == 4 * 1024 + 1
    context = _context().model_copy(
        update={"allowed_queries": frozenset({exact_coverage, too_large_coverage})}
    )

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise SourceUnreachable()

    tool = bind_evidence_tool(
        provider,
        context=context,
        expected_unavailability=(
            ExpectedToolUnavailability(
                exception_type=SourceUnreachable,
                reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
            ),
        ),
    )

    exact = await tool(_tool_context(), "filing", exact_coverage)
    too_large = await tool(
        _tool_context(tool_call_id="call-2"), "filing", too_large_coverage
    )

    unusable = ToolUnavailable(
        reason=ToolUnavailableReason.RESPONSE_UNUSABLE,
        requested_coverage="Requested coverage could not be safely projected.",
    )
    assert exact.return_value == unusable
    assert too_large.return_value == unusable


@pytest.mark.asyncio
async def test_unknown_and_provider_timeout_fail_closed() -> None:
    async def unknown_provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise RuntimeError("unknown provider failure")

    unknown_tool = bind_evidence_tool(unknown_provider, context=_context())
    with pytest.raises(RuntimeError, match="unknown provider failure"):
        await unknown_tool(_tool_context(), "filing", "Apple revenue")

    async def timeout_provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise TimeoutError("provider timeout is not binding-owned")

    timeout_tool = bind_evidence_tool(timeout_provider, context=_context())
    with pytest.raises(TimeoutError, match="provider timeout is not binding-owned"):
        await timeout_tool(_tool_context(), "filing", "Apple revenue")


@pytest.mark.asyncio
async def test_binding_owned_timeout_becomes_call_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SpecialistToolCapture()

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        await asyncio.sleep(1)
        raise AssertionError("binding timeout must cancel the provider")

    assert agent_evidence.TOOL_TIMEOUT_SECONDS == 20
    monkeypatch.setattr(agent_evidence, "TOOL_TIMEOUT_SECONDS", 0.001)
    tool = bind_evidence_tool(
        provider,
        context=_context(),
        capture=capture,
    )

    returned = await tool(_tool_context(), "filing", "Apple revenue")

    assert returned.return_value == ToolUnavailable(
        reason=ToolUnavailableReason.CALL_TIMEOUT,
        requested_coverage="Apple revenue",
    )
    assert capture.unavailability[0].reason is ToolUnavailableReason.CALL_TIMEOUT


@pytest.mark.parametrize(
    "markdown",
    ["Apple grew. [[E:x]]", "Apple grew. [[E:2]]", "Apple grew. [[E:1]] [[ E:2]]"],
)
def test_report_gate_rejects_malformed_or_unknown_evidence_marker(
    markdown: str,
) -> None:
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(),),
        finding_evidence_ids=("evidence-1",),
        context=_context(),
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
        publish_report(FinancialResearchReport(markdown_report=markdown), prepared)


def test_evidence_cache_does_not_commit_before_all_references_validate() -> None:
    catalog = RequestEvidenceCatalog()

    with pytest.raises(ValueError, match="Evidence provenance is missing"):
        catalog.accept_referenced(
            (_evidence(),),
            finding_evidence_ids=("evidence-1", "missing"),
            context=_context(),
        )

    with pytest.raises(ValueError, match="Evidence is not accepted"):
        catalog.resolve("evidence-1", tenant_id="tenant-a", request_id="request-1")


def test_synthesis_rejects_evidence_older_than_scope_freshness_window() -> None:
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(as_of_date=date(2026, 8, 29)),),
        finding_evidence_ids=("evidence-1",),
        context=_context(),
    )

    with pytest.raises(ValueError, match="Evidence is not accepted"):
        prepare_synthesis(
            standalone_query="Apple outlook",
            intent="market_outlook",
            accepted_evidence_ids=("evidence-1",),
            catalog=catalog,
            tenant_id="tenant-a",
            request_id="request-1",
            as_of_date=date(2026, 9, 6),
            max_evidence_age_days=7,
        )
