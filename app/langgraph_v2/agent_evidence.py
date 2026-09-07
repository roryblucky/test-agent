"""Request-local Evidence acceptance and deterministic citation gates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import Awaitable, Callable, Collection, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import RunContext, ToolReturn

from app.langgraph_v2.calculations import (
    CalculationArtifact,
    CalculationArtifactInvalid,
    accepted_calculation_artifacts,
    require_calculation_evidence_support,
)
from app.models.workflow import CitationReference

_EVIDENCE_MARKER = re.compile(r"\[\[E:([1-9][0-9]*)\]\]")
_MARKER_LIKE = re.compile(r"\[\[\s*[Ee]\s*:")
_CALCULATION_MARKER = re.compile(r"\[\[C:([1-9][0-9]*)\]\]")
_CALCULATION_MARKER_LIKE = re.compile(r"\[\[\s*[Cc]\s*:")
_TOOL_RETURN_MAX_BYTES = 4 * 1024
_EVIDENCE_BODY_MAX_BYTES = 16 * 1024
_REQUEST_EVIDENCE_BODY_MAX_BYTES = 8 * 1024 * 1024
DATA_GAP_TEXT_MAX_BYTES = 256
DATA_GAP_IDENTIFIER_MAX_ASCII_CHARACTERS = 64
TOOL_TIMEOUT_SECONDS = 20
_UNUSABLE_COVERAGE = "Requested coverage could not be safely projected."
_MAX_PREPARED_EVIDENCE = 64
_MAX_PREPARED_CALCULATIONS = 32
_MAX_PREPARED_CALCULATION_BYTES = 2 * 1024


class PreparedSynthesisLimitExceeded(ValueError):
    """Stop before a bounded Synthesis projection would exceed its limits."""


class SynthesisCandidateRejected(ValueError):
    """A schema or publication-gate rejection eligible for one repair."""

    def __init__(self, validation_errors: tuple[str, ...]) -> None:
        if not validation_errors:
            raise ValueError("Synthesis validation errors are required")
        self.validation_errors = validation_errors
        super().__init__("; ".join(validation_errors))


class SynthesisOutputInvalid(SynthesisCandidateRejected):
    """Keep model output failure separate from fatal runtime failures."""

    def __init__(self) -> None:
        super().__init__(("Synthesis report is invalid",))


def require_data_gap_text(value: str, *, label: str) -> str:
    """Return a Data Gap text field only when it fits its shared byte bound."""
    if len(value.encode("utf-8")) > DATA_GAP_TEXT_MAX_BYTES:
        raise ValueError(f"{label} exceeds {DATA_GAP_TEXT_MAX_BYTES} UTF-8 bytes")
    return value


def require_data_gap_identifier(value: str, *, label: str) -> str:
    """Return an opaque Data Gap identifier only when it fits its shared bound."""
    if (
        not value
        or not value.isascii()
        or len(value) > DATA_GAP_IDENTIFIER_MAX_ASCII_CHARACTERS
    ):
        raise ValueError(
            f"{label} must contain at most "
            f"{DATA_GAP_IDENTIFIER_MAX_ASCII_CHARACTERS} ASCII characters"
        )
    return value


def _is_projectable_data_gap_text(value: str) -> bool:
    try:
        require_data_gap_text(value, label="Data Gap text")
    except ValueError:
        return False
    return bool(value)


class ToolUnavailableReason(StrEnum):
    """Closed model-visible reasons for expected Tool unavailability."""

    SOURCE_UNREACHABLE = "source_unreachable"
    COVERAGE_NOT_SUPPORTED = "coverage_not_supported"
    STALE_ONLY = "stale_only"
    CALL_TIMEOUT = "call_timeout"
    RESPONSE_UNUSABLE = "response_unusable"


class ToolTelemetryStatus(StrEnum):
    """Closed Tool lifecycle statuses emitted to audit and progress streams."""

    REJECTED = "rejected"
    STARTED = "started"
    UNAVAILABLE = "unavailable"
    COMPLETED = "completed"


class ToolUnavailable(BaseModel):
    """Bounded expected inability returned to the active Specialist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool_unavailable"] = "tool_unavailable"
    reason: ToolUnavailableReason
    requested_coverage: str = Field(min_length=1)


class DataGapView(BaseModel):
    """Safe Data Gap projection for actor prompts and publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_coverage: str = Field(min_length=1)
    reason: ToolUnavailableReason
    observed_at: datetime

    @field_validator("requested_coverage")
    @classmethod
    def _validate_coverage(cls, value: str) -> str:
        return require_data_gap_text(value, label="Data Gap coverage")


class ToolUnavailabilityRecord(BaseModel):
    """App-only provenance for one expected unavailable Tool outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    tenant_id: str
    request_id: str
    task_id: str
    attempt: int
    tool_call_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    source: str | None
    observed_at: datetime
    reason: ToolUnavailableReason
    requested_coverage: str = Field(min_length=1)

    @field_validator("id", "tool_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        return require_data_gap_identifier(value, label="Data Gap identifier")

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str | None) -> str | None:
        if value is not None:
            require_data_gap_text(value, label="Data Gap source")
        return value

    @field_validator("requested_coverage")
    @classmethod
    def _validate_requested_coverage(cls, value: str) -> str:
        return require_data_gap_text(value, label="Data Gap coverage")


@dataclass(frozen=True)
class ExpectedToolUnavailability:
    """One explicitly registered provider failure that may become data."""

    exception_type: type[Exception]
    reason: ToolUnavailableReason


class _ProviderTimeoutEscaped(Exception):
    """Keep a provider-raised timeout distinct from binding-owned timeout."""


class EvidenceEnvelope(BaseModel):
    """One successful Tool's app-only source record and request-local body."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    excerpt: str = Field(min_length=1, max_length=4096)
    as_of_date: date
    raw_provider_payload: str | None = None


class EvidenceCacheCapacityExceeded(ValueError):
    """Reject a request-local Evidence write that would exceed a hard cache cap."""


class EvidenceReferenceInvalid(ValueError):
    """Reject a model-authored Evidence ID absent from its terminal Tool output."""


@dataclass
class SpecialistToolCapture:
    """One actor invocation's app-only Evidence and unavailable Tool records."""

    evidence: list[EvidenceEnvelope] = field(default_factory=list[EvidenceEnvelope])
    unavailability: list[ToolUnavailabilityRecord] = field(
        default_factory=list[ToolUnavailabilityRecord]
    )
    calculations: list[CalculationArtifact] = field(
        default_factory=list[CalculationArtifact]
    )


class EvidenceInvocationContext(BaseModel):
    """Frozen request identity and authority for one Specialist invocation."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt: int = Field(default=1, ge=1)
    allowed_tool_ids: frozenset[str] = frozenset()
    allowed_sources: frozenset[str] = frozenset()
    allowed_queries: frozenset[str] = frozenset()


EvidenceProvider = Callable[[str, str], Awaitable[EvidenceEnvelope]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _tool_return_size(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _bounded_unavailable(
    *,
    reason: ToolUnavailableReason,
    requested_coverage: str,
) -> ToolUnavailable:
    unavailable = ToolUnavailable(
        reason=reason,
        requested_coverage=requested_coverage,
    )
    if _tool_return_size(unavailable.model_dump(mode="json")) <= _TOOL_RETURN_MAX_BYTES:
        return unavailable
    return ToolUnavailable(
        reason=ToolUnavailableReason.RESPONSE_UNUSABLE,
        requested_coverage=_UNUSABLE_COVERAGE,
    )


def _unavailability_record(
    *,
    context: EvidenceInvocationContext,
    run_context: RunContext[None],
    tool_id: str,
    source: str | None,
    unavailable: ToolUnavailable,
    now: Callable[[], datetime],
) -> ToolUnavailabilityRecord:
    tool_call_id = run_context.tool_call_id
    if not tool_call_id or run_context.tool_name != tool_id:
        raise ValueError("Tool call identity is not eligible")
    observed_at = now()
    if observed_at.tzinfo is None:
        raise ValueError("Tool observation time must be timezone-aware")
    coverage_is_safe = _is_projectable_data_gap_text(unavailable.requested_coverage)
    source_is_safe = source is None or _is_projectable_data_gap_text(source)
    return ToolUnavailabilityRecord(
        id=f"unavailable_{uuid4().hex}",
        tenant_id=context.tenant_id,
        request_id=context.request_id,
        task_id=context.task_id,
        attempt=context.attempt,
        tool_call_id=tool_call_id,
        tool_id=tool_id,
        source=source if source_is_safe else None,
        observed_at=observed_at.astimezone(UTC),
        reason=(
            unavailable.reason
            if coverage_is_safe and source_is_safe
            else ToolUnavailableReason.RESPONSE_UNUSABLE
        ),
        requested_coverage=(
            unavailable.requested_coverage if coverage_is_safe else _UNUSABLE_COVERAGE
        ),
    )


def _same_canonical_evidence(left: EvidenceEnvelope, right: EvidenceEnvelope) -> bool:
    """Compare evidence content while excluding per-invocation provenance."""
    excluded = {"raw_provider_payload", "task_id"}
    return left.model_dump(exclude=excluded) == right.model_dump(exclude=excluded)


def bind_evidence_tool(
    provider: EvidenceProvider,
    *,
    context: EvidenceInvocationContext,
    capture: SpecialistToolCapture | None = None,
    telemetry: Callable[[ToolTelemetryStatus], None] | None = None,
    tool_id: str = "read_evidence",
    expected_unavailability: tuple[ExpectedToolUnavailability, ...] = (),
    now: Callable[[], datetime] = _utc_now,
) -> Callable[..., Awaitable[ToolReturn[dict[str, str] | ToolUnavailable]]]:
    """Bind one frozen Scope-limited Evidence reader for a Specialist run."""
    require_data_gap_identifier(tool_id, label="Evidence Tool identifier")
    expected_types = tuple(item.exception_type for item in expected_unavailability)
    if len(set(expected_types)) != len(expected_types):
        raise ValueError("Expected Tool unavailability registration conflicts")

    async def provider_result(source: str, query: str) -> EvidenceEnvelope:
        try:
            return await provider(source, query)
        except TimeoutError as error:
            raise _ProviderTimeoutEscaped from error

    def unavailable_result(
        *,
        reason: ToolUnavailableReason,
        source: str,
        query: str,
        run_context: RunContext[None],
    ) -> ToolReturn[dict[str, str] | ToolUnavailable]:
        source_is_safe = _is_projectable_data_gap_text(source)
        unavailable = _bounded_unavailable(
            reason=(
                reason if source_is_safe else ToolUnavailableReason.RESPONSE_UNUSABLE
            ),
            requested_coverage=query,
        )
        record = _unavailability_record(
            context=context,
            run_context=run_context,
            tool_id=tool_id,
            source=source if source_is_safe else None,
            unavailable=unavailable,
            now=now,
        )
        if capture is not None:
            capture.unavailability.append(record)
        if telemetry is not None:
            telemetry(ToolTelemetryStatus.UNAVAILABLE)
        return ToolReturn(return_value=unavailable, metadata=record)

    async def read_evidence(
        run_context: RunContext[None], source: str, query: str
    ) -> ToolReturn[dict[str, str] | ToolUnavailable]:
        """Fetch one source only when its trusted Scope permits it."""
        if source not in context.allowed_sources:
            if telemetry is not None:
                telemetry(ToolTelemetryStatus.REJECTED)
            raise ValueError("Evidence source is not eligible")
        if query not in context.allowed_queries:
            if telemetry is not None:
                telemetry(ToolTelemetryStatus.REJECTED)
            raise ValueError("Evidence query is not eligible")
        if telemetry is not None:
            telemetry(ToolTelemetryStatus.STARTED)
        try:
            evidence = await asyncio.wait_for(
                provider_result(source, query), timeout=TOOL_TIMEOUT_SECONDS
            )
            if (
                evidence.source != source
                or evidence.tenant_id != context.tenant_id
                or evidence.request_id != context.request_id
                or evidence.task_id != context.task_id
            ):
                raise ValueError("Evidence provenance is not eligible")
            return_value = {"evidence_id": evidence.id, "excerpt": evidence.excerpt}
            if _tool_return_size(return_value) > _TOOL_RETURN_MAX_BYTES:
                return unavailable_result(
                    reason=ToolUnavailableReason.RESPONSE_UNUSABLE,
                    source=source,
                    query=query,
                    run_context=run_context,
                )
            if capture is not None:
                capture.evidence.append(evidence)
        except _ProviderTimeoutEscaped as error:
            cause = error.__cause__
            assert isinstance(cause, TimeoutError)
            raise cause
        except TimeoutError:
            return unavailable_result(
                reason=ToolUnavailableReason.CALL_TIMEOUT,
                source=source,
                query=query,
                run_context=run_context,
            )
        except expected_types as error:
            expected = next(
                (
                    item
                    for item in expected_unavailability
                    if type(error) is item.exception_type
                ),
                None,
            )
            if expected is None:
                raise
            return unavailable_result(
                reason=expected.reason,
                source=source,
                query=query,
                run_context=run_context,
            )
        if telemetry is not None:
            telemetry(ToolTelemetryStatus.COMPLETED)
        return ToolReturn(return_value=return_value, metadata=evidence)

    return read_evidence


@dataclass
class RequestEvidenceCatalog:
    """Hold request-local Evidence bodies for accepted-state projections."""

    _evidence: dict[str, EvidenceEnvelope] = field(
        default_factory=dict[str, EvidenceEnvelope]
    )
    _body_sizes: dict[str, int] = field(default_factory=dict[str, int])
    _total_body_bytes: int = 0
    _lock: Any = field(default_factory=threading.RLock, repr=False)

    @property
    def cached_body_bytes(self) -> int:
        """Expose the request-local unique-body footprint for capacity tests."""
        with self._lock:
            return self._total_body_bytes

    def accept_referenced(
        self,
        returned: Iterable[EvidenceEnvelope],
        *,
        finding_evidence_ids: tuple[str, ...],
        context: EvidenceInvocationContext,
    ) -> None:
        """Cache returned bodies after validating one staged Finding reference."""
        returned_items = tuple(returned)
        with self._lock:
            returned_by_id: dict[str, EvidenceEnvelope] = {}
            for evidence in returned_items:
                existing_returned = returned_by_id.get(evidence.id)
                if existing_returned is not None and not _same_canonical_evidence(
                    existing_returned, evidence
                ):
                    raise ValueError("Evidence provenance conflicts")
                returned_by_id[evidence.id] = evidence
            for evidence_id in finding_evidence_ids:
                evidence = returned_by_id.get(evidence_id)
                if evidence is None:
                    raise EvidenceReferenceInvalid("Evidence provenance is missing")
                if (
                    evidence.tenant_id != context.tenant_id
                    or evidence.request_id != context.request_id
                    or evidence.task_id != context.task_id
                ):
                    raise ValueError("Evidence provenance is not eligible")
            for evidence in returned_items:
                if (
                    evidence.tenant_id != context.tenant_id
                    or evidence.request_id != context.request_id
                    or evidence.task_id != context.task_id
                ):
                    raise ValueError("Evidence provenance is not eligible")
                existing = self._evidence.get(evidence.id)
                if existing is not None and not _same_canonical_evidence(
                    existing, evidence
                ):
                    raise ValueError("Evidence body conflicts")
                body_size = len(evidence.body.encode("utf-8"))
                if body_size > _EVIDENCE_BODY_MAX_BYTES:
                    raise EvidenceCacheCapacityExceeded("Evidence body exceeds 16 KiB")
                body_hash = hashlib.sha256(evidence.body.encode("utf-8")).hexdigest()
                if body_hash not in self._body_sizes:
                    if (
                        self._total_body_bytes + body_size
                        > _REQUEST_EVIDENCE_BODY_MAX_BYTES
                    ):
                        raise EvidenceCacheCapacityExceeded(
                            "Request Evidence cache exceeds 8 MiB"
                        )
                    self._body_sizes[body_hash] = body_size
                    self._total_body_bytes += body_size
                stored = evidence.model_copy(update={"raw_provider_payload": None})
                if existing is None or stored.task_id < existing.task_id:
                    self._evidence[evidence.id] = stored

    def resolve(
        self,
        evidence_id: str,
        *,
        tenant_id: str,
        request_id: str,
        accepted_evidence_ids: Collection[str],
        as_of_date: date | None = None,
        max_evidence_age_days: int | None = None,
    ) -> EvidenceEnvelope:
        """Return only Evidence owned by this active Tenant Request."""
        with self._lock:
            evidence = self._evidence.get(evidence_id)
            if (
                evidence is None
                or evidence_id not in accepted_evidence_ids
                or evidence.tenant_id != tenant_id
                or evidence.request_id != request_id
                or (
                    as_of_date is not None
                    and (
                        evidence.as_of_date > as_of_date
                        or (as_of_date - evidence.as_of_date).days
                        > (max_evidence_age_days or 0)
                    )
                )
            ):
                raise ValueError("Evidence is not accepted")
            return evidence


class PreparedEvidence(BaseModel):
    """Bounded source view permitted in a Synthesis prompt."""

    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    source_url: str
    title: str
    excerpt: str


class PreparedCalculation(BaseModel):
    """Value-free Calculation projection permitted in a Synthesis prompt."""

    model_config = ConfigDict(frozen=True)

    alias: str
    method: str
    unit: str
    currency: str
    period_start: date
    period_end: date
    as_of_date: date
    assumptions: tuple[str, ...]

    def canonical_json_size(self) -> int:
        """Measure the exact bounded prompt projection."""
        return len(
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )


class PreparedSynthesis(BaseModel):
    """Frozen bounded business input for one Synthesis invocation."""

    model_config = ConfigDict(frozen=True)

    standalone_query: str
    intent: str
    evidence: tuple[PreparedEvidence, ...] = Field(max_length=_MAX_PREPARED_EVIDENCE)
    data_gaps: tuple[DataGapView, ...] = ()
    calculations: tuple[PreparedCalculation, ...] = Field(
        max_length=_MAX_PREPARED_CALCULATIONS,
        default=(),
    )
    calculation_artifacts: tuple[CalculationArtifact, ...] = Field(
        default=(), exclude=True, repr=False
    )


class FinancialResearchReport(BaseModel):
    """Minimal model-authored Markdown candidate with Evidence markers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    markdown_report: str = Field(min_length=1)


class PublishedReport(BaseModel):
    """Code-gated Markdown and citations for the public response."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: tuple[CitationReference, ...]


class SynthesisActor(Protocol):
    """Produce one typed report candidate from a frozen projection."""

    async def synthesize(self, prepared: PreparedSynthesis) -> FinancialResearchReport:
        """Produce the first candidate without repair feedback."""
        ...


class SynthesisRepairActor(SynthesisActor, Protocol):
    """Produce the one allowed repair for an unchanged frozen projection."""

    async def repair(
        self,
        prepared: PreparedSynthesis,
        *,
        validation_errors: tuple[str, ...],
    ) -> FinancialResearchReport:
        """Produce a repaired candidate from deterministic gate errors only."""
        ...


def prepare_synthesis(
    *,
    standalone_query: str,
    intent: str,
    accepted_evidence_ids: tuple[str, ...],
    catalog: RequestEvidenceCatalog,
    tenant_id: str,
    request_id: str,
    as_of_date: date | None = None,
    max_evidence_age_days: int = 7,
    data_gaps: tuple[DataGapView, ...] = (),
    accepted_calculations: tuple[CalculationArtifact, ...] = (),
) -> PreparedSynthesis:
    """Build the sole bounded Evidence projection Synthesis may receive."""
    if len(accepted_evidence_ids) > _MAX_PREPARED_EVIDENCE:
        raise ValueError("Prepared Evidence count exceeds 64")
    if len(set(accepted_evidence_ids)) != len(accepted_evidence_ids):
        raise ValueError("Prepared Evidence identifiers must be unique")
    resolved_evidence = tuple(
        catalog.resolve(
            evidence_id,
            tenant_id=tenant_id,
            request_id=request_id,
            accepted_evidence_ids=accepted_evidence_ids,
            as_of_date=as_of_date,
            max_evidence_age_days=max_evidence_age_days,
        )
        for evidence_id in accepted_evidence_ids
    )
    if any(not item.body for item in resolved_evidence):
        raise ValueError("Evidence body is unavailable for Synthesis")
    evidence = tuple(
        PreparedEvidence(
            id=item.id,
            source=item.source,
            source_url=item.source_url,
            title=item.title,
            excerpt=item.excerpt,
        )
        for item in resolved_evidence
    )
    artifacts = accepted_calculation_artifacts(accepted_calculations)
    if any(
        artifact.tenant_id != tenant_id or artifact.request_id != request_id
        for artifact in artifacts
    ):
        raise CalculationArtifactInvalid("Calculation Artifact provenance is invalid")
    evidence_hashes_by_id = {
        item.id: hashlib.sha256(item.body.encode("utf-8")).hexdigest()
        for item in resolved_evidence
    }
    for artifact in artifacts:
        require_calculation_evidence_support(
            artifact, evidence_hashes_by_id=evidence_hashes_by_id
        )
    if len(artifacts) > _MAX_PREPARED_CALCULATIONS:
        raise PreparedSynthesisLimitExceeded("Prepared Calculation count exceeds 32")
    calculations = tuple(
        PreparedCalculation(
            alias=f"C:{index}",
            method=artifact.method,
            unit=artifact.unit,
            currency=artifact.currency,
            period_start=artifact.period_start,
            period_end=artifact.period_end,
            as_of_date=artifact.as_of_date,
            assumptions=artifact.assumptions,
        )
        for index, artifact in enumerate(artifacts, start=1)
    )
    if any(
        calculation.canonical_json_size() > _MAX_PREPARED_CALCULATION_BYTES
        for calculation in calculations
    ):
        raise PreparedSynthesisLimitExceeded("Prepared Calculation exceeds 2KiB")
    return PreparedSynthesis(
        standalone_query=standalone_query,
        intent=intent,
        evidence=evidence,
        data_gaps=data_gaps,
        calculations=calculations,
        calculation_artifacts=artifacts,
    )


def publish_report(
    candidate: FinancialResearchReport,
    prepared: PreparedSynthesis,
) -> PublishedReport:
    """Validate every Evidence marker and derive public citations in code."""
    _validate_calculation_projection(prepared)
    calculation_markers = _CALCULATION_MARKER.findall(candidate.markdown_report)
    evidence_markers = _EVIDENCE_MARKER.findall(candidate.markdown_report)
    validation_errors = (
        *_marker_errors(
            marker_name="Evidence",
            markers=evidence_markers,
            marker_remainder=_EVIDENCE_MARKER.sub("", candidate.markdown_report),
            marker_like=_MARKER_LIKE,
            catalog_size=len(prepared.evidence),
            require_all=True,
        ),
        *_marker_errors(
            marker_name="Calculation",
            markers=calculation_markers,
            marker_remainder=_CALCULATION_MARKER.sub("", candidate.markdown_report),
            marker_like=_CALCULATION_MARKER_LIKE,
            catalog_size=len(prepared.calculation_artifacts),
            require_all=False,
        ),
    )
    if validation_errors:
        raise SynthesisCandidateRejected(validation_errors)
    answer = _CALCULATION_MARKER.sub(
        lambda match: _render_calculation(
            prepared.calculation_artifacts[int(match.group(1)) - 1]
        ),
        candidate.markdown_report,
    )
    citations: list[CitationReference] = []
    for marker in evidence_markers:
        index = int(marker)
        evidence = prepared.evidence[index - 1]
        citations.append(
            CitationReference(
                index=index,
                evidence_id=evidence.id,
                source=evidence.source,
                title=evidence.title,
                url=evidence.source_url,
                snippet=evidence.excerpt,
            )
        )
    return PublishedReport(answer=answer, citations=tuple(citations))


async def synthesize_report(
    actor: SynthesisActor,
    prepared: PreparedSynthesis,
) -> PublishedReport:
    """Own the one allowed repair while preserving one frozen Synthesis value."""
    try:
        return publish_report(await actor.synthesize(prepared), prepared)
    except SynthesisCandidateRejected as rejection:
        repair_actor = cast(SynthesisRepairActor, actor)
        if not callable(getattr(repair_actor, "repair", None)):
            raise RuntimeError("Synthesis actor does not support repair") from rejection
        return publish_report(
            await repair_actor.repair(
                prepared, validation_errors=rejection.validation_errors
            ),
            prepared,
        )


def _marker_errors(
    *,
    marker_name: str,
    markers: tuple[str, ...] | list[str],
    marker_remainder: str,
    marker_like: re.Pattern[str],
    catalog_size: int,
    require_all: bool,
) -> tuple[str, ...]:
    """Return deterministic strict-marker failures without semantic inference."""
    errors: list[str] = []
    if marker_like.search(marker_remainder):
        errors.append(f"{marker_name} marker is malformed")
    indexes = tuple(int(marker) for marker in markers)
    if any(index > catalog_size for index in indexes):
        errors.append(f"{marker_name} marker is not eligible")
    if len(set(indexes)) != len(indexes):
        errors.append(f"{marker_name} marker is duplicated")
    if require_all and set(indexes) != set(range(1, catalog_size + 1)):
        errors.append(f"{marker_name} marker is missing")
    return tuple(errors)


def _validate_calculation_projection(prepared: PreparedSynthesis) -> None:
    """Reject a reconstructed or misaligned value-free Calculation alias map."""
    if len(prepared.calculations) != len(prepared.calculation_artifacts):
        raise ValueError("Prepared Calculation projection is invalid")
    for index, (projection, artifact) in enumerate(
        zip(prepared.calculations, prepared.calculation_artifacts, strict=True),
        start=1,
    ):
        if projection != PreparedCalculation(
            alias=f"C:{index}",
            method=artifact.method,
            unit=artifact.unit,
            currency=artifact.currency,
            period_start=artifact.period_start,
            period_end=artifact.period_end,
            as_of_date=artifact.as_of_date,
            assumptions=artifact.assumptions,
        ):
            raise ValueError("Prepared Calculation projection is invalid")


def _render_calculation(artifact: CalculationArtifact) -> str:
    """Render one trusted value with the concise interpretation disclosure."""
    method = artifact.method.value.replace("_", " ")
    period = f"{artifact.period_start.isoformat()} to {artifact.period_end.isoformat()}"
    assumptions = "; ".join(artifact.assumptions)
    return (
        f"{artifact.formatted_value} ({method}; {artifact.unit}; {artifact.currency}; "
        f"{period}; assumptions: {assumptions})"
    )
