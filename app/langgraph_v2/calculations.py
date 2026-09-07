"""Trusted deterministic Calculation Artifact execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import date
from decimal import ROUND_HALF_UP, Context, Decimal, localcontext
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_ai import RunContext, ToolReturn

MAX_CALCULATIONS_PER_CONTRIBUTION = 8
_V1_DECIMAL_CONTEXT = Context(prec=50, rounding=ROUND_HALF_UP)


class CalculationArtifactInvalid(ValueError):
    """Reject an Artifact that cannot enter accepted calculation state."""


class CalculationMethod(StrEnum):
    """The registered deterministic calculation methods."""

    PERIOD_RETURN = "period_return"
    ANNUALIZED_VOLATILITY = "annualized_volatility"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"


class PriceObservation(BaseModel):
    """One code-resolved positive price observation."""

    model_config = ConfigDict(frozen=True)

    as_of: date
    value: Decimal

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value <= 0:
            raise ValueError("Price observation must be finite and positive")
        return value


class TrustedPriceSeries(BaseModel):
    """A registered series whose values are never model-authored input."""

    model_config = ConfigDict(frozen=True)

    ref: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    observations: tuple[PriceObservation, ...] = Field(min_length=2)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    evidence_hashes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_observations(self) -> TrustedPriceSeries:
        if len(self.evidence_refs) != len(self.evidence_hashes):
            raise ValueError("Trusted series Evidence provenance is invalid")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.evidence_hashes
        ):
            raise ValueError("Trusted series Evidence hash is invalid")
        dates = tuple(observation.as_of for observation in self.observations)
        if dates != tuple(sorted(dates)) or len(set(dates)) != len(dates):
            raise ValueError("Trusted series observations must be strictly ordered")
        return self


class CalculationRequest(BaseModel):
    """The complete model-provided selector, deliberately excluding raw series."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: CalculationMethod
    version: Literal["v1"]
    series_ref: str = Field(min_length=1)


class CalculationExecutionContext(BaseModel):
    """Trusted Run, Task, and outer-attempt provenance for one Artifact."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    tool_id: str = Field(default="calculation", min_length=1)


class CalculationInput(BaseModel):
    """The sole normalized trusted-series selector retained for reproduction."""

    model_config = ConfigDict(frozen=True)

    series_ref: str = Field(min_length=1)


class PeriodReturnAudit(BaseModel):
    """Internal operands retained for the fixed period-return method."""

    model_config = ConfigDict(frozen=True)

    first_value: str
    last_value: str


class AnnualizedVolatilityAudit(BaseModel):
    """Internal simple-return inputs retained for the fixed volatility method."""

    model_config = ConfigDict(frozen=True)

    simple_returns: tuple[str, ...]
    annualization_factor: int


class MaximumDrawdownAudit(BaseModel):
    """Internal peak-relative inputs retained for the fixed drawdown method."""

    model_config = ConfigDict(frozen=True)

    drawdowns: tuple[str, ...]


CalculationAudit = PeriodReturnAudit | AnnualizedVolatilityAudit | MaximumDrawdownAudit


class CalculationExecutionRecord(BaseModel):
    """Frozen executor identity and completion status for one Artifact."""

    model_config = ConfigDict(frozen=True)

    executor: Literal["calculation_executor"]
    status: Literal["ok"]
    tool_id: str = Field(min_length=1)


class CalculationArtifact(BaseModel):
    """Internal reproducible record whose canonical value never enters prompts."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    method: CalculationMethod
    version: Literal["v1"]
    precision: int = Field(ge=0, le=12)
    unit: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    period_start: date
    period_end: date
    as_of_date: date
    assumptions: tuple[str, ...]
    normalized_input: CalculationInput
    evidence_refs: tuple[str, ...]
    evidence_hashes: tuple[str, ...]
    audit_inputs: CalculationAudit
    execution_record: CalculationExecutionRecord
    tenant_id: str
    request_id: str
    task_id: str
    attempt: int = Field(ge=1)
    canonical_value: Decimal
    formatted_value: str

    @field_validator("canonical_value")
    @classmethod
    def _validate_canonical_value(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Calculation value must be finite")
        return value

    def reproduce_formatted_value(self) -> str:
        """Render the stored canonical Decimal with the registered v1 formatter."""
        return _format_percent(self.canonical_value, precision=self.precision)

    def require_integrity(self) -> None:
        """Reject a record whose ID or rendered value no longer matches its value."""
        if self.id != _artifact_id(self.model_dump(mode="json", exclude={"id"})):
            raise ValueError("Calculation Artifact ID is invalid")
        if self.formatted_value != self.reproduce_formatted_value():
            raise ValueError("Calculation Artifact formatted value is invalid")


class CalculationToolResult(BaseModel):
    """Model-visible formatted result, never the canonical Decimal."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    method: CalculationMethod
    period_start: date
    period_end: date
    formatted_value: str


class CalculationExecutor:
    """Resolve registered trusted series and execute the registered calculation."""

    def __init__(self, series: tuple[TrustedPriceSeries, ...]) -> None:
        self._series = {item.ref: item for item in series}
        if len(self._series) != len(series):
            raise ValueError("Trusted series registration conflicts")

    def execute(
        self,
        request: CalculationRequest,
        *,
        context: CalculationExecutionContext,
    ) -> CalculationArtifact:
        """Return one content-bound Artifact from a trusted registered series."""
        series = self._series.get(request.series_ref)
        if series is None:
            raise ValueError("Calculation series is not trusted")
        value: Decimal
        assumptions: tuple[str, ...]
        audit_inputs: CalculationAudit
        with localcontext(_V1_DECIMAL_CONTEXT):
            if request.method is CalculationMethod.PERIOD_RETURN:
                value = series.observations[-1].value / series.observations[
                    0
                ].value - Decimal(1)
                assumptions = ("Positive ordered close prices.",)
                audit_inputs = PeriodReturnAudit(
                    first_value=str(series.observations[0].value),
                    last_value=str(series.observations[-1].value),
                )
            elif request.method is CalculationMethod.ANNUALIZED_VOLATILITY:
                returns = tuple(
                    current.value / previous.value - Decimal(1)
                    for previous, current in zip(
                        series.observations, series.observations[1:], strict=False
                    )
                )
                if len(returns) < 2:
                    raise ValueError(
                        "Annualized volatility requires three observations"
                    )
                mean = sum(returns, Decimal(0)) / Decimal(len(returns))
                sample_variance = sum(
                    ((item - mean) ** 2 for item in returns), Decimal(0)
                ) / Decimal(len(returns) - 1)
                value = sample_variance.sqrt() * Decimal(252).sqrt()
                assumptions = (
                    "Positive ordered close prices.",
                    "Sample standard deviation of simple returns annualized by 252.",
                )
                audit_inputs = AnnualizedVolatilityAudit(
                    simple_returns=tuple(str(item) for item in returns),
                    annualization_factor=252,
                )
            elif request.method is CalculationMethod.MAXIMUM_DRAWDOWN:
                running_peak = series.observations[0].value
                drawdowns: list[Decimal] = []
                for observation in series.observations:
                    running_peak = max(running_peak, observation.value)
                    drawdowns.append(observation.value / running_peak - Decimal(1))
                value = min(drawdowns)
                assumptions = (
                    "Positive ordered close prices.",
                    "Drawdown is measured from each prior running peak.",
                )
                audit_inputs = MaximumDrawdownAudit(
                    drawdowns=tuple(str(item) for item in drawdowns),
                )
            else:
                raise ValueError("Calculation method is not registered")
        values: dict[str, Any] = {
            "method": request.method,
            "version": request.version,
            "precision": 4,
            "unit": "percent",
            "currency": series.currency,
            "period_start": series.observations[0].as_of,
            "period_end": series.observations[-1].as_of,
            "as_of_date": series.observations[-1].as_of,
            "assumptions": assumptions,
            "normalized_input": CalculationInput(series_ref=series.ref).model_dump(
                mode="json"
            ),
            "evidence_refs": series.evidence_refs,
            "evidence_hashes": series.evidence_hashes,
            "audit_inputs": audit_inputs.model_dump(mode="json"),
            "execution_record": CalculationExecutionRecord(
                executor="calculation_executor",
                status="ok",
                tool_id=context.tool_id,
            ).model_dump(mode="json"),
            "tenant_id": context.tenant_id,
            "request_id": context.request_id,
            "task_id": context.task_id,
            "attempt": context.attempt,
            "canonical_value": value,
            "formatted_value": _format_percent(value, precision=4),
        }
        artifact_id = _artifact_id(values)
        return CalculationArtifact(id=artifact_id, **values)

    def require_reproducible(self, artifact: CalculationArtifact) -> None:
        """Fail closed unless the trusted resolver recreates this exact Artifact."""
        try:
            reproduced = self.execute(
                CalculationRequest(
                    method=artifact.method,
                    version=artifact.version,
                    series_ref=artifact.normalized_input.series_ref,
                ),
                context=CalculationExecutionContext(
                    tenant_id=artifact.tenant_id,
                    request_id=artifact.request_id,
                    task_id=artifact.task_id,
                    attempt=artifact.attempt,
                    tool_id=artifact.execution_record.tool_id,
                ),
            )
        except (KeyError, ValueError) as error:
            raise ValueError("Calculation Artifact is not reproducible") from error
        if reproduced != artifact:
            raise ValueError("Calculation Artifact is not reproducible")


def bind_calculation_tool(
    executor: CalculationExecutor,
    *,
    context: CalculationExecutionContext,
    tool_id: str,
    capture: list[CalculationArtifact] | None = None,
) -> Callable[
    [RunContext[None], CalculationMethod, Literal["v1"], str],
    Awaitable[ToolReturn[CalculationToolResult]],
]:
    """Bind one registered executor so the model can select only a trusted ref."""

    async def calculate(
        run_context: RunContext[None],
        method: CalculationMethod,
        version: Literal["v1"],
        series_ref: str,
    ) -> ToolReturn[CalculationToolResult]:
        if run_context.tool_name != tool_id or not run_context.tool_call_id:
            raise ValueError("Calculation Tool call identity is not eligible")
        artifact = executor.execute(
            CalculationRequest(
                method=method,
                version=version,
                series_ref=series_ref,
            ),
            context=context.model_copy(update={"tool_id": tool_id}),
        )
        if capture is not None:
            capture.append(artifact)
        result = CalculationToolResult(
            artifact_id=artifact.id,
            method=artifact.method,
            period_start=artifact.period_start,
            period_end=artifact.period_end,
            formatted_value=artifact.formatted_value,
        )
        return ToolReturn(
            return_value=result,
            metadata=artifact,
        )

    return calculate


def accepted_calculation_artifacts(
    artifacts: tuple[CalculationArtifact, ...],
) -> tuple[CalculationArtifact, ...]:
    """Return unique Artifacts in the supplied code-owned business order."""
    by_id: dict[str, CalculationArtifact] = {}
    for artifact in artifacts:
        try:
            artifact.require_integrity()
        except ValueError as error:
            raise CalculationArtifactInvalid(str(error)) from error
        existing = by_id.get(artifact.id)
        if existing is not None and existing != artifact:
            raise CalculationArtifactInvalid("Calculation Artifact ID conflicts")
        by_id[artifact.id] = artifact
    return tuple(by_id.values())


def require_calculation_evidence_support(
    artifact: CalculationArtifact,
    *,
    evidence_hashes_by_id: Mapping[str, str],
) -> None:
    """Require every Artifact Evidence ref to match trusted resolved body bytes."""
    if any(
        evidence_hashes_by_id.get(evidence_ref) != evidence_hash
        for evidence_ref, evidence_hash in zip(
            artifact.evidence_refs, artifact.evidence_hashes, strict=True
        )
    ):
        raise CalculationArtifactInvalid(
            "Calculation Artifact Evidence support is invalid"
        )


def _artifact_id(values: dict[str, object]) -> str:
    """Return one content-bound ID from the complete internal Artifact record."""
    payload = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"calculation_{hashlib.sha256(payload).hexdigest()[:32]}"


def _format_percent(value: Decimal, *, precision: int) -> str:
    """Render one canonical relative return using the fixed v1 percentage policy."""
    with localcontext(_V1_DECIMAL_CONTEXT):
        quantized = (value * 100).quantize(
            Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP
        )
    return f"{quantized:.{precision}f}%"
