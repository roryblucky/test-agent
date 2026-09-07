"""Public Evidence cache and report-gate coverage for Agent research."""

import asyncio
import hashlib
import json
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import cast

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
from app.langgraph_v2.calculations import (
    CalculationArtifactInvalid,
    CalculationExecutionContext,
    CalculationExecutor,
    CalculationMethod,
    CalculationRequest,
    PriceObservation,
    TrustedPriceSeries,
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


@pytest.mark.asyncio
async def test_concurrent_siblings_converge_on_the_same_evidence_id_and_body() -> None:
    catalog = RequestEvidenceCatalog()
    first_context = _context()
    second_context = first_context.model_copy(update={"task_id": "task-2"})

    await asyncio.gather(
        asyncio.to_thread(
            catalog.accept_referenced,
            (_evidence(id="shared-evidence", body="same body"),),
            finding_evidence_ids=("shared-evidence",),
            context=first_context,
        ),
        asyncio.to_thread(
            catalog.accept_referenced,
            (
                _evidence(
                    id="shared-evidence",
                    body="same body",
                    task_id="task-2",
                ),
            ),
            finding_evidence_ids=("shared-evidence",),
            context=second_context,
        ),
    )
    accepted = catalog.resolve(
        "shared-evidence",
        tenant_id="tenant-a",
        request_id="request-1",
        accepted_evidence_ids=("shared-evidence",),
    )
    assert accepted.body == "same body"
    assert accepted.task_id == "task-1"
    assert accepted.raw_provider_payload is None


@pytest.mark.asyncio
async def test_concurrent_siblings_reject_the_same_evidence_id_with_different_body() -> (
    None
):
    catalog = RequestEvidenceCatalog()
    first_context = _context()
    second_context = first_context.model_copy(update={"task_id": "task-2"})

    results = await asyncio.gather(
        asyncio.to_thread(
            catalog.accept_referenced,
            (_evidence(id="shared-evidence", body="first body"),),
            finding_evidence_ids=("shared-evidence",),
            context=first_context,
        ),
        asyncio.to_thread(
            catalog.accept_referenced,
            (
                _evidence(
                    id="shared-evidence",
                    body="second body",
                    task_id="task-2",
                ),
            ),
            finding_evidence_ids=("shared-evidence",),
            context=second_context,
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, ValueError) for result in results) == 1


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


def test_synthesis_receives_value_free_calculation_aliases_and_code_renders_them() -> (
    None
):
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(),), finding_evidence_ids=("evidence-1",), context=_context()
    )
    provisional_executor = CalculationExecutor(
        (
            TrustedPriceSeries(
                ref="apple-series",
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(as_of=date(2026, 1, 3), value=Decimal("110")),
                ),
                evidence_refs=("evidence-1",),
                evidence_hashes=(hashlib.sha256(b"Apple revenue grew.").hexdigest(),),
            ),
        )
    )
    artifact = provisional_executor.execute(
        CalculationRequest(
            method=CalculationMethod.PERIOD_RETURN,
            version="v1",
            series_ref="apple-series",
        ),
        context=CalculationExecutionContext(
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            attempt=1,
            tool_id="calculator",
        ),
    )
    prepared = prepare_synthesis(
        standalone_query="Apple outlook",
        intent="market_outlook",
        accepted_evidence_ids=("evidence-1",),
        catalog=catalog,
        tenant_id="tenant-a",
        request_id="request-1",
        accepted_calculations=(artifact,),
    )
    published = publish_report(
        FinancialResearchReport(markdown_report="The return was [[C:1]]. [[E:1]]"),
        prepared,
    )

    prompt = json.dumps(prepared.model_dump(mode="json"), sort_keys=True)
    assert prepared.calculations[0].alias == "C:1"
    assert "canonical_value" not in prompt
    assert artifact.id not in prompt
    assert "10.0000%" not in prompt
    assert "[[C:1]]" not in published.answer
    assert "10.0000% (period return; percent; USD; 2026-01-02 to 2026-01-03" in (
        published.answer
    )
    assert [citation.evidence_id for citation in published.citations] == ["evidence-1"]
    with pytest.raises(ValueError, match="Calculation marker is not eligible"):
        publish_report(
            FinancialResearchReport(markdown_report="[[C:2]] [[E:1]]"), prepared
        )
    assert (
        publish_report(
            FinancialResearchReport(markdown_report="Apple grew. [[E:1]]"), prepared
        ).answer
        == "Apple grew. [[E:1]]"
    )
    for markdown in (
        "[[C:1]] [[C:1]] [[E:1]]",
        "[[C:x]] [[E:1]]",
        "[[C:01]] [[E:1]]",
        "[[ C:1]] [[E:1]]",
    ):
        with pytest.raises(ValueError, match="Calculation marker"):
            publish_report(FinancialResearchReport(markdown_report=markdown), prepared)
    with pytest.raises(ValueError, match="projection is invalid"):
        publish_report(
            FinancialResearchReport(markdown_report="[[C:1]] [[E:1]]"),
            prepared.model_copy(update={"calculations": ()}),
        )
    foreign_artifact = provisional_executor.execute(
        CalculationRequest(
            method=CalculationMethod.PERIOD_RETURN,
            version="v1",
            series_ref="apple-series",
        ),
        context=CalculationExecutionContext(
            tenant_id="tenant-b",
            request_id="request-1",
            task_id="task-1",
            attempt=1,
            tool_id="calculator",
        ),
    )
    with pytest.raises(CalculationArtifactInvalid, match="provenance is invalid"):
        prepare_synthesis(
            standalone_query="Apple outlook",
            intent="market_outlook",
            accepted_evidence_ids=("evidence-1",),
            catalog=catalog,
            tenant_id="tenant-a",
            request_id="request-1",
            accepted_calculations=(foreign_artifact,),
        )


def test_prepared_calculations_enforce_the_projection_count_limit() -> None:
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(),), finding_evidence_ids=("evidence-1",), context=_context()
    )
    executor = CalculationExecutor(
        tuple(
            TrustedPriceSeries(
                ref=f"apple-series-{index:02}",
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(as_of=date(2026, 1, 3), value=Decimal("110")),
                ),
                evidence_refs=("evidence-1",),
                evidence_hashes=(hashlib.sha256(b"Apple revenue grew.").hexdigest(),),
            )
            for index in range(34)
        )
    )
    context = CalculationExecutionContext(
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
        attempt=1,
    )
    at_count_limit = tuple(
        executor.execute(
            CalculationRequest(
                method=CalculationMethod.PERIOD_RETURN,
                version="v1",
                series_ref=f"apple-series-{index:02}",
            ),
            context=context,
        )
        for index in range(32)
    )

    prepared = prepare_synthesis(
        standalone_query="Apple outlook",
        intent="market_outlook",
        accepted_evidence_ids=("evidence-1",),
        catalog=catalog,
        tenant_id="tenant-a",
        request_id="request-1",
        accepted_calculations=at_count_limit,
    )

    assert len(prepared.calculations) == 32
    with pytest.raises(ValueError, match="Prepared Calculation count exceeds 32"):
        prepare_synthesis(
            standalone_query="Apple outlook",
            intent="market_outlook",
            accepted_evidence_ids=("evidence-1",),
            catalog=catalog,
            tenant_id="tenant-a",
            request_id="request-1",
            accepted_calculations=(
                *at_count_limit,
                executor.execute(
                    CalculationRequest(
                        method=CalculationMethod.PERIOD_RETURN,
                        version="v1",
                        series_ref="apple-series-32",
                    ),
                    context=context,
                ),
            ),
        )


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
        "evidence-1",
        tenant_id="tenant-a",
        request_id="request-1",
        accepted_evidence_ids=("evidence-1",),
    )
    assert accepted.body == first.body


def test_evidence_cache_keeps_orphans_diagnostic_only() -> None:
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(), _evidence(id="orphan-1", body="diagnostic only")),
        finding_evidence_ids=("evidence-1",),
        context=_context(),
    )
    assert (
        catalog.resolve(
            "evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            accepted_evidence_ids=("evidence-1",),
        ).body
        == "Apple revenue grew."
    )
    with pytest.raises(ValueError, match="Evidence is not accepted"):
        catalog.resolve(
            "orphan-1",
            tenant_id="tenant-a",
            request_id="request-1",
            accepted_evidence_ids=("evidence-1",),
        )


def test_referenced_evidence_is_ineligible_without_checkpointed_accepted_state() -> (
    None
):
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(),),
        finding_evidence_ids=("evidence-1",),
        context=_context(),
    )

    with pytest.raises(ValueError, match="Evidence is not accepted"):
        catalog.resolve(
            "evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            accepted_evidence_ids=(),
        )

    assert (
        catalog.resolve(
            "evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            accepted_evidence_ids=("evidence-1",),
        ).body
        == "Apple revenue grew."
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
            context=_context(),
        )


@pytest.mark.asyncio
async def test_evidence_tool_freezes_source_before_provider_access() -> None:
    calls: list[tuple[str, str]] = []
    capture = SpecialistToolCapture()

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        calls.append((source, query))
        return _evidence()

    tool = bind_evidence_tool(
        provider,
        context=_context(),
        capture=capture,
    )

    returned = await tool(_tool_context(), "filing", "Apple revenue")

    assert returned.return_value == {
        "evidence_id": "evidence-1",
        "excerpt": "Apple revenue grew.",
    }
    assert returned.metadata is None
    assert capture.evidence == [_evidence()]
    assert calls == [("filing", "Apple revenue")]
    with pytest.raises(ValueError, match="Evidence source is not eligible"):
        await tool(_tool_context(), "private", "Apple revenue")
    assert calls == [("filing", "Apple revenue")]


@pytest.mark.asyncio
async def test_evidence_tool_projects_an_unusable_provider_response() -> None:
    capture = SpecialistToolCapture()

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        return cast(EvidenceEnvelope, {"unexpected": "provider response"})

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
    assert capture.unavailability[0].reason is ToolUnavailableReason.RESPONSE_UNUSABLE


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
    [
        "Apple grew.",
        "Apple grew. [[E:1]] [[E:1]]",
        "Apple grew. [[E:x]]",
        "Apple grew. [[E:2]]",
        "Apple grew. [[E:1]] [[ E:2]]",
    ],
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


def test_prepared_evidence_enforces_the_projection_count_limit() -> None:
    catalog = RequestEvidenceCatalog()
    evidence_ids = tuple(f"evidence-{index}" for index in range(65))
    catalog.accept_referenced(
        tuple(_evidence(id=evidence_id) for evidence_id in evidence_ids),
        finding_evidence_ids=evidence_ids,
        context=_context(),
    )

    prepared = prepare_synthesis(
        standalone_query="Apple outlook",
        intent="market_outlook",
        accepted_evidence_ids=evidence_ids[:64],
        catalog=catalog,
        tenant_id="tenant-a",
        request_id="request-1",
    )

    assert len(prepared.evidence) == 64
    with pytest.raises(ValueError, match="Prepared Evidence count exceeds 64"):
        prepare_synthesis(
            standalone_query="Apple outlook",
            intent="market_outlook",
            accepted_evidence_ids=evidence_ids,
            catalog=catalog,
            tenant_id="tenant-a",
            request_id="request-1",
        )


def test_prepared_synthesis_rejects_duplicate_evidence_and_missing_bodies() -> None:
    catalog = RequestEvidenceCatalog()
    evidence = _evidence()
    catalog.accept_referenced(
        (evidence,), finding_evidence_ids=(evidence.id,), context=_context()
    )

    with pytest.raises(ValueError, match="identifiers must be unique"):
        prepare_synthesis(
            standalone_query="Apple outlook",
            intent="market_outlook",
            accepted_evidence_ids=(evidence.id, evidence.id),
            catalog=catalog,
            tenant_id="tenant-a",
            request_id="request-1",
        )

    missing_body = evidence.model_copy(update={"body": ""})
    recovery_catalog = RequestEvidenceCatalog()
    recovery_catalog.accept_referenced(
        (missing_body,),
        finding_evidence_ids=(missing_body.id,),
        context=_context(),
    )
    with pytest.raises(ValueError, match="body is unavailable"):
        prepare_synthesis(
            standalone_query="Apple outlook",
            intent="market_outlook",
            accepted_evidence_ids=(missing_body.id,),
            catalog=recovery_catalog,
            tenant_id="tenant-a",
            request_id="request-1",
        )


def test_report_gate_requires_each_prepared_evidence_once() -> None:
    catalog = RequestEvidenceCatalog()
    first = _evidence(id="evidence-1")
    second = _evidence(id="evidence-2")
    catalog.accept_referenced(
        (first, second),
        finding_evidence_ids=(first.id, second.id),
        context=_context(),
    )
    prepared = prepare_synthesis(
        standalone_query="Apple outlook",
        intent="market_outlook",
        accepted_evidence_ids=(first.id, second.id),
        catalog=catalog,
        tenant_id="tenant-a",
        request_id="request-1",
    )

    with pytest.raises(ValueError, match="Evidence marker is missing"):
        publish_report(
            FinancialResearchReport(markdown_report="Apple grew. [[E:1]]"), prepared
        )

    published = publish_report(
        FinancialResearchReport(markdown_report="Second [[E:2]], first [[E:1]]"),
        prepared,
    )
    assert [citation.evidence_id for citation in published.citations] == [
        "evidence-2",
        "evidence-1",
    ]


def test_evidence_cache_does_not_commit_before_all_references_validate() -> None:
    catalog = RequestEvidenceCatalog()

    with pytest.raises(ValueError, match="Evidence provenance is missing"):
        catalog.accept_referenced(
            (_evidence(),),
            finding_evidence_ids=("evidence-1", "missing"),
            context=_context(),
        )

    with pytest.raises(ValueError, match="Evidence is not accepted"):
        catalog.resolve(
            "evidence-1",
            tenant_id="tenant-a",
            request_id="request-1",
            accepted_evidence_ids=(),
        )


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
