"""Calculation Executor contract coverage."""

from datetime import date
from decimal import ROUND_DOWN, Context, Decimal, localcontext
from typing import cast

import pytest
from pydantic import ValidationError
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from app.langgraph_v2.calculations import (
    CalculationArtifact,
    CalculationExecutionContext,
    CalculationExecutor,
    CalculationMethod,
    CalculationRequest,
    CalculationToolResult,
    PriceObservation,
    TrustedPriceSeries,
    bind_calculation_tool,
)


def _period_return_artifact(*, ref: str, evidence_ref: str):
    executor = CalculationExecutor(
        (
            TrustedPriceSeries(
                ref=ref,
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(as_of=date(2026, 1, 3), value=Decimal("110")),
                ),
                evidence_refs=(evidence_ref,),
                evidence_hashes=("a" * 64,),
            ),
        )
    )
    return executor.execute(
        CalculationRequest(
            method=CalculationMethod.PERIOD_RETURN,
            version="v1",
            series_ref=ref,
        ),
        context=CalculationExecutionContext(
            tenant_id="tenant-a",
            request_id="request-1",
            task_id=f"task_{'a' * 32}",
            attempt=1,
        ),
    )


def test_period_return_uses_only_a_trusted_series_reference() -> None:
    executor = CalculationExecutor(
        (
            TrustedPriceSeries(
                ref="series-apple-q1",
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(as_of=date(2026, 3, 31), value=Decimal("110")),
                ),
                evidence_refs=("evidence-1",),
                evidence_hashes=("a" * 64,),
            ),
        )
    )
    artifact = executor.execute(
        CalculationRequest(
            method=CalculationMethod.PERIOD_RETURN,
            version="v1",
            series_ref="series-apple-q1",
        ),
        context=CalculationExecutionContext(
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            attempt=1,
        ),
    )

    assert artifact.formatted_value == "10.0000%"
    assert artifact.reproduce_formatted_value() == "10.0000%"
    assert artifact.evidence_refs == ("evidence-1",)

    with pytest.raises(ValidationError):
        CalculationRequest.model_validate(
            {
                "method": "period_return",
                "version": "v1",
                "series_ref": "series-apple-q1",
                "series": [100, 110],
            }
        )


def test_annualized_volatility_uses_sample_simple_returns() -> None:
    executor = CalculationExecutor(
        (
            TrustedPriceSeries(
                ref="series-volatility",
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(as_of=date(2026, 1, 3), value=Decimal("110")),
                    PriceObservation(as_of=date(2026, 1, 4), value=Decimal("99")),
                ),
                evidence_refs=("evidence-1",),
                evidence_hashes=("a" * 64,),
            ),
        )
    )

    artifact = executor.execute(
        CalculationRequest(
            method=CalculationMethod.ANNUALIZED_VOLATILITY,
            version="v1",
            series_ref="series-volatility",
        ),
        context=CalculationExecutionContext(
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            attempt=1,
        ),
    )

    assert artifact.formatted_value == "224.4994%"


def test_executor_uses_fixed_decimal_context_for_content_bound_artifacts() -> None:
    executor = CalculationExecutor(
        (
            TrustedPriceSeries(
                ref="series-context",
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(as_of=date(2026, 1, 3), value=Decimal("117")),
                    PriceObservation(as_of=date(2026, 1, 4), value=Decimal("103")),
                ),
                evidence_refs=("evidence-1",),
                evidence_hashes=("a" * 64,),
            ),
        )
    )
    request = CalculationRequest(
        method=CalculationMethod.ANNUALIZED_VOLATILITY,
        version="v1",
        series_ref="series-context",
    )
    context = CalculationExecutionContext(
        tenant_id="tenant-a",
        request_id="request-1",
        task_id="task-1",
        attempt=1,
    )

    with localcontext(Context(prec=10, rounding=ROUND_DOWN)):
        low_precision = executor.execute(request, context=context)
    with localcontext(Context(prec=28)):
        normal_precision = executor.execute(request, context=context)

    assert low_precision == normal_precision


def test_maximum_drawdown_uses_the_prior_running_peak() -> None:
    executor = CalculationExecutor(
        (
            TrustedPriceSeries(
                ref="series-drawdown",
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(as_of=date(2026, 1, 3), value=Decimal("120")),
                    PriceObservation(as_of=date(2026, 1, 4), value=Decimal("90")),
                    PriceObservation(as_of=date(2026, 1, 5), value=Decimal("110")),
                ),
                evidence_refs=("evidence-1",),
                evidence_hashes=("a" * 64,),
            ),
        )
    )

    artifact = executor.execute(
        CalculationRequest(
            method=CalculationMethod.MAXIMUM_DRAWDOWN,
            version="v1",
            series_ref="series-drawdown",
        ),
        context=CalculationExecutionContext(
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            attempt=1,
        ),
    )

    assert artifact.formatted_value == "-25.0000%"


def test_executor_rejects_an_artifact_that_cannot_be_reproduced() -> None:
    executor = CalculationExecutor(
        (
            TrustedPriceSeries(
                ref="series-integrity",
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(as_of=date(2026, 1, 3), value=Decimal("110")),
                ),
                evidence_refs=("evidence-1",),
                evidence_hashes=("a" * 64,),
            ),
        )
    )
    artifact = executor.execute(
        CalculationRequest(
            method=CalculationMethod.PERIOD_RETURN,
            version="v1",
            series_ref="series-integrity",
        ),
        context=CalculationExecutionContext(
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            attempt=1,
        ),
    )

    executor.require_reproducible(artifact)

    with pytest.raises(ValueError, match="Calculation Artifact is not reproducible"):
        executor.require_reproducible(
            artifact.model_copy(update={"formatted_value": "999.0000%"})
        )


@pytest.mark.asyncio
async def test_calculation_tool_keeps_the_canonical_value_out_of_model_data() -> None:
    executor = CalculationExecutor(
        (
            TrustedPriceSeries(
                ref="series-tool",
                instrument_id="AAPL",
                currency="USD",
                unit="price",
                observations=(
                    PriceObservation(as_of=date(2026, 1, 2), value=Decimal("100")),
                    PriceObservation(as_of=date(2026, 1, 3), value=Decimal("110")),
                ),
                evidence_refs=("evidence-1",),
                evidence_hashes=("a" * 64,),
            ),
        )
    )
    captured: list[CalculationArtifact] = []
    tool = bind_calculation_tool(
        executor,
        context=CalculationExecutionContext(
            tenant_id="tenant-a",
            request_id="request-1",
            task_id="task-1",
            attempt=1,
        ),
        tool_id="price-calculator",
        capture=captured,
    )

    returned = await tool(
        RunContext(
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            tool_call_id="call-1",
            tool_name="price-calculator",
        ),
        CalculationMethod.PERIOD_RETURN,
        "v1",
        "series-tool",
    )

    result = cast(CalculationToolResult, returned.return_value)
    assert result.model_dump(mode="json") == {
        "artifact_id": captured[0].id,
        "method": "period_return",
        "period_start": "2026-01-02",
        "period_end": "2026-01-03",
        "formatted_value": "10.0000%",
    }
    assert "canonical_value" not in result.model_dump()
    assert returned.metadata == captured[0]


def test_calculation_artifact_internal_records_are_deeply_immutable() -> None:
    artifact = _period_return_artifact(ref="series-immutable", evidence_ref="evidence")

    with pytest.raises(ValidationError):
        artifact.normalized_input.series_ref = "forged"
    with pytest.raises(ValidationError):
        artifact.audit_inputs.first_value = "forged"  # type: ignore[union-attr]
    with pytest.raises(ValidationError):
        artifact.execution_record.tool_id = "forged"
