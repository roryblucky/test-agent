"""Public Evidence cache and report-gate coverage for Agent research."""

import asyncio
import hashlib
import json
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

import app.langgraph_v2.agent_evidence as agent_evidence
from app.langgraph_v2.agent_evidence import (
    EvidenceCacheCapacityExceeded,
    EvidenceEnvelope,
    EvidenceInvocationContext,
    ExpectedToolUnavailability,
    FinancialResearchReport,
    PreparedCalculation,
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
async def test_concurrent_identical_evidence_bodies_are_counted_once() -> None:
    catalog = RequestEvidenceCatalog()

    await asyncio.gather(
        asyncio.to_thread(
            catalog.accept_referenced,
            (_evidence(id="evidence-1", body="same body"),),
            finding_evidence_ids=("evidence-1",),
            context=_context(),
        ),
        asyncio.to_thread(
            catalog.accept_referenced,
            (_evidence(id="evidence-2", body="same body"),),
            finding_evidence_ids=("evidence-2",),
            context=_context(),
        ),
    )

    assert catalog.cached_body_bytes == len(b"same body")


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
    assert catalog.cached_body_bytes == len(b"same body")
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


def test_prepared_calculations_enforce_exact_projection_count_and_size_limits() -> None:
    catalog = RequestEvidenceCatalog()
    catalog.accept_referenced(
        (_evidence(),), finding_evidence_ids=("evidence-1",), context=_context()
    )
    executor = CalculationExecutor(
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
    artifact = executor.execute(
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
        ),
    )
    projection = PreparedCalculation(
        alias="C:32",
        method=artifact.method,
        unit=artifact.unit,
        currency=artifact.currency,
        period_start=artifact.period_start,
        period_end=artifact.period_end,
        as_of_date=artifact.as_of_date,
        assumptions=artifact.assumptions,
    )
    padded_currency = "x" * (2 * 1024 - projection.canonical_json_size() + len("USD"))
    executor = CalculationExecutor(
        tuple(
            TrustedPriceSeries(
                ref=f"apple-series-{index:02}",
                instrument_id="AAPL",
                currency=padded_currency,
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
    assert prepared.calculations[-1].canonical_json_size() == 2 * 1024
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
    with pytest.raises(ValueError, match="Prepared Calculation exceeds 2KiB"):
        prepare_synthesis(
            standalone_query="Apple outlook",
            intent="market_outlook",
            accepted_evidence_ids=("evidence-1",),
            catalog=catalog,
            tenant_id="tenant-a",
            request_id="request-1",
            accepted_calculations=(
                CalculationExecutor(
                    (
                        TrustedPriceSeries(
                            ref="over-limit-series",
                            instrument_id="AAPL",
                            currency=padded_currency + "xx",
                            unit="price",
                            observations=(
                                PriceObservation(
                                    as_of=date(2026, 1, 2), value=Decimal("100")
                                ),
                                PriceObservation(
                                    as_of=date(2026, 1, 3), value=Decimal("110")
                                ),
                            ),
                            evidence_refs=("evidence-1",),
                            evidence_hashes=(
                                hashlib.sha256(b"Apple revenue grew.").hexdigest(),
                            ),
                        ),
                    )
                ).execute(
                    CalculationRequest(
                        method=CalculationMethod.PERIOD_RETURN,
                        version="v1",
                        series_ref="over-limit-series",
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


def test_evidence_cache_rejects_exact_item_and_total_capacity_overflow() -> None:
    catalog = RequestEvidenceCatalog()
    exact_body = "x" * (16 * 1024)
    catalog.accept_referenced(
        (_evidence(body=exact_body),),
        finding_evidence_ids=("evidence-1",),
        context=_context(),
    )
    with pytest.raises(EvidenceCacheCapacityExceeded):
        catalog.accept_referenced(
            (_evidence(id="too-large", body=f"{exact_body}x"),),
            finding_evidence_ids=("too-large",),
            context=_context(),
        )

    full_catalog = RequestEvidenceCatalog()
    bodies = tuple(
        _evidence(
            id=f"evidence-{index}",
            body=f"{index:04d}" + "x" * (16 * 1024 - 4),
        )
        for index in range(512)
    )
    full_catalog.accept_referenced(
        bodies,
        finding_evidence_ids=tuple(item.id for item in bodies),
        context=_context(),
    )
    with pytest.raises(EvidenceCacheCapacityExceeded):
        full_catalog.accept_referenced(
            (_evidence(id="one-too-many", body="y"),),
            finding_evidence_ids=("one-too-many",),
            context=_context(),
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

    assert exact.return_value == ToolUnavailable(
        reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
        requested_coverage=exact_coverage,
    )
    assert isinstance(exact.return_value, ToolUnavailable)
    assert (
        len(
            json.dumps(
                exact.return_value.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        == 4 * 1024
    )
    assert (
        len(
            json.dumps(
                ToolUnavailable(
                    reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
                    requested_coverage=too_large_coverage,
                ).model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        == 4 * 1024 + 1
    )
    assert too_large.return_value == ToolUnavailable(
        reason=ToolUnavailableReason.RESPONSE_UNUSABLE,
        requested_coverage="Requested coverage could not be safely projected.",
    )


@pytest.mark.asyncio
async def test_unavailable_binding_projects_coverage_and_source_boundaries() -> None:
    class SourceUnreachable(Exception):
        pass

    exact_coverage = "c" * 256
    oversize_coverage = f"{exact_coverage}c"
    exact_source = "s" * 256
    oversize_source = f"{exact_source}s"
    context = _context().model_copy(
        update={
            "allowed_sources": frozenset({exact_source, oversize_source}),
            "allowed_queries": frozenset({exact_coverage, oversize_coverage}),
        }
    )
    capture = SpecialistToolCapture()

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise SourceUnreachable()

    tool = bind_evidence_tool(
        provider,
        context=context,
        capture=capture,
        expected_unavailability=(
            ExpectedToolUnavailability(
                exception_type=SourceUnreachable,
                reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
            ),
        ),
    )

    exact = await tool(_tool_context(), exact_source, exact_coverage)
    oversize_coverage_return = await tool(
        _tool_context(tool_call_id="call-2"), exact_source, oversize_coverage
    )
    oversize_source_return = await tool(
        _tool_context(tool_call_id="call-3"), oversize_source, exact_coverage
    )

    assert exact.return_value == ToolUnavailable(
        reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
        requested_coverage=exact_coverage,
    )
    assert oversize_coverage_return.return_value == ToolUnavailable(
        reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
        requested_coverage=oversize_coverage,
    )
    assert oversize_source_return.return_value == ToolUnavailable(
        reason=ToolUnavailableReason.RESPONSE_UNUSABLE,
        requested_coverage=exact_coverage,
    )
    assert capture.unavailability[0].requested_coverage == exact_coverage
    assert capture.unavailability[0].source == exact_source
    assert capture.unavailability[1].reason is ToolUnavailableReason.RESPONSE_UNUSABLE
    assert (
        capture.unavailability[1].requested_coverage
        == "Requested coverage could not be safely projected."
    )
    assert capture.unavailability[1].source == exact_source
    assert capture.unavailability[2].requested_coverage == exact_coverage
    assert capture.unavailability[2].source is None


@pytest.mark.asyncio
async def test_unavailable_binding_rejects_oversize_tool_id() -> None:
    class SourceUnreachable(Exception):
        pass

    async def provider(source: str, query: str) -> EvidenceEnvelope:
        del source, query
        raise SourceUnreachable()

    exact_tool_id = "t" * 64
    capture = SpecialistToolCapture()
    tool = bind_evidence_tool(
        provider,
        context=_context(),
        capture=capture,
        tool_id=exact_tool_id,
        expected_unavailability=(
            ExpectedToolUnavailability(
                exception_type=SourceUnreachable,
                reason=ToolUnavailableReason.SOURCE_UNREACHABLE,
            ),
        ),
    )

    await tool(_tool_context(tool_name=exact_tool_id), "filing", "Apple revenue")

    assert capture.unavailability[0].tool_id == exact_tool_id
    with pytest.raises(ValueError, match="Tool identifier"):
        bind_evidence_tool(provider, context=_context(), tool_id=f"{exact_tool_id}t")


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
