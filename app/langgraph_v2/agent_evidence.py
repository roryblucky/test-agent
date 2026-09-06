"""Request-local Evidence acceptance and deterministic citation gates."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext, ToolReturn

from app.models.workflow import CitationReference

_EVIDENCE_MARKER = re.compile(r"\[\[E:([1-9][0-9]*)\]\]")
_MARKER_LIKE = re.compile(r"\[\[\s*E\s*:")
_TOOL_RETURN_MAX_BYTES = 4 * 1024
TOOL_TIMEOUT_SECONDS = 20
_UNUSABLE_COVERAGE = "Requested coverage could not be safely projected."


class ToolUnavailableReason(StrEnum):
    """Closed model-visible reasons for expected Tool unavailability."""

    SOURCE_UNREACHABLE = "source_unreachable"
    COVERAGE_NOT_SUPPORTED = "coverage_not_supported"
    STALE_ONLY = "stale_only"
    CALL_TIMEOUT = "call_timeout"
    RESPONSE_UNUSABLE = "response_unusable"


class ToolUnavailable(BaseModel):
    """Bounded expected inability returned to the active Specialist."""

    model_config = ConfigDict(frozen=True)

    kind: str = "tool_unavailable"
    reason: ToolUnavailableReason
    requested_coverage: str = Field(min_length=1)


class DataGapView(BaseModel):
    """Safe Data Gap projection for actor prompts and publication."""

    model_config = ConfigDict(frozen=True)

    requested_coverage: str
    reason: ToolUnavailableReason
    observed_at: datetime


class ToolUnavailabilityRecord(BaseModel):
    """App-only provenance for one expected unavailable Tool outcome."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: str
    request_id: str
    task_id: str
    attempt: int
    tool_call_id: str
    tool_id: str
    source: str | None
    observed_at: datetime
    reason: ToolUnavailableReason
    requested_coverage: str


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
    source: str,
    unavailable: ToolUnavailable,
    now: Callable[[], datetime],
) -> ToolUnavailabilityRecord:
    tool_call_id = run_context.tool_call_id
    if not tool_call_id or run_context.tool_name != tool_id:
        raise ValueError("Tool call identity is not eligible")
    observed_at = now()
    if observed_at.tzinfo is None:
        raise ValueError("Tool observation time must be timezone-aware")
    return ToolUnavailabilityRecord(
        id=f"unavailable_{uuid4().hex}",
        tenant_id=context.tenant_id,
        request_id=context.request_id,
        task_id=context.task_id,
        attempt=context.attempt,
        tool_call_id=tool_call_id,
        tool_id=tool_id,
        source=source,
        observed_at=observed_at.astimezone(UTC),
        reason=unavailable.reason,
        requested_coverage=unavailable.requested_coverage,
    )


def _same_canonical_evidence(left: EvidenceEnvelope, right: EvidenceEnvelope) -> bool:
    """Compare accepted Evidence while excluding noncanonical provider payload."""
    excluded = {"raw_provider_payload"}
    return left.model_dump(exclude=excluded) == right.model_dump(exclude=excluded)


def bind_evidence_tool(
    provider: EvidenceProvider,
    *,
    context: EvidenceInvocationContext,
    returned_evidence: list[EvidenceEnvelope] | None = None,
    returned_unavailability: list[ToolUnavailabilityRecord] | None = None,
    telemetry: Callable[[str], None] | None = None,
    tool_id: str = "read_evidence",
    expected_unavailability: tuple[ExpectedToolUnavailability, ...] = (),
    now: Callable[[], datetime] = _utc_now,
) -> Callable[..., Awaitable[ToolReturn[dict[str, str] | ToolUnavailable]]]:
    """Bind one frozen Scope-limited Evidence reader for a Specialist run."""
    expected_types = tuple(item.exception_type for item in expected_unavailability)
    if len(set(expected_types)) != len(expected_types):
        raise ValueError("Expected Tool unavailability registration conflicts")

    async def provider_result(source: str, query: str) -> EvidenceEnvelope:
        try:
            return await provider(source, query)
        except TimeoutError as error:
            raise _ProviderTimeoutEscaped from error

    async def read_evidence(
        run_context: RunContext[None], source: str, query: str
    ) -> ToolReturn[dict[str, str] | ToolUnavailable]:
        """Fetch one source only when its trusted Scope permits it."""
        if source not in context.allowed_sources:
            if telemetry is not None:
                telemetry("rejected")
            raise ValueError("Evidence source is not eligible")
        if query not in context.allowed_queries:
            if telemetry is not None:
                telemetry("rejected")
            raise ValueError("Evidence query is not eligible")
        if telemetry is not None:
            telemetry("started")
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
                unavailable = _bounded_unavailable(
                    reason=ToolUnavailableReason.RESPONSE_UNUSABLE,
                    requested_coverage=query,
                )
                record = _unavailability_record(
                    context=context,
                    run_context=run_context,
                    tool_id=tool_id,
                    source=source,
                    unavailable=unavailable,
                    now=now,
                )
                if returned_unavailability is not None:
                    returned_unavailability.append(record)
                if telemetry is not None:
                    telemetry("unavailable")
                return ToolReturn(return_value=unavailable, metadata=record)
            if returned_evidence is not None:
                returned_evidence.append(evidence)
        except _ProviderTimeoutEscaped as error:
            cause = error.__cause__
            assert isinstance(cause, TimeoutError)
            raise cause
        except TimeoutError:
            unavailable = _bounded_unavailable(
                reason=ToolUnavailableReason.CALL_TIMEOUT,
                requested_coverage=query,
            )
            record = _unavailability_record(
                context=context,
                run_context=run_context,
                tool_id=tool_id,
                source=source,
                unavailable=unavailable,
                now=now,
            )
            if returned_unavailability is not None:
                returned_unavailability.append(record)
            if telemetry is not None:
                telemetry("unavailable")
            return ToolReturn(return_value=unavailable, metadata=record)
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
            unavailable = _bounded_unavailable(
                reason=expected.reason,
                requested_coverage=query,
            )
            record = _unavailability_record(
                context=context,
                run_context=run_context,
                tool_id=tool_id,
                source=source,
                unavailable=unavailable,
                now=now,
            )
            if returned_unavailability is not None:
                returned_unavailability.append(record)
            if telemetry is not None:
                telemetry("unavailable")
            return ToolReturn(return_value=unavailable, metadata=record)
        if telemetry is not None:
            telemetry("completed")
        return ToolReturn(return_value=return_value, metadata=evidence)

    return read_evidence


@dataclass
class RequestEvidenceCatalog:
    """Hold only Finding-referenced Evidence bodies for one active request."""

    _evidence: dict[str, EvidenceEnvelope] = field(
        default_factory=dict[str, EvidenceEnvelope]
    )

    def accept_referenced(
        self,
        returned: Iterable[EvidenceEnvelope],
        *,
        finding_evidence_ids: tuple[str, ...],
        context: EvidenceInvocationContext,
    ) -> None:
        """Accept exact successful provenance referenced by the terminal Finding."""
        returned_items = tuple(returned)
        returned_by_id: dict[str, EvidenceEnvelope] = {}
        for evidence in returned_items:
            existing_returned = returned_by_id.get(evidence.id)
            if existing_returned is not None and not _same_canonical_evidence(
                existing_returned, evidence
            ):
                raise ValueError("Evidence provenance conflicts")
            returned_by_id[evidence.id] = evidence
        accepted: list[EvidenceEnvelope] = []
        for evidence_id in finding_evidence_ids:
            evidence = returned_by_id.get(evidence_id)
            if evidence is None:
                raise ValueError("Evidence provenance is missing")
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
            accepted.append(evidence)
        for evidence in accepted:
            self._evidence[evidence.id] = evidence

    def resolve(
        self,
        evidence_id: str,
        *,
        tenant_id: str,
        request_id: str,
        as_of_date: date | None = None,
        max_evidence_age_days: int | None = None,
    ) -> EvidenceEnvelope:
        """Return only Evidence owned by this active Tenant Request."""
        evidence = self._evidence.get(evidence_id)
        if (
            evidence is None
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


class PreparedSynthesis(BaseModel):
    """Frozen bounded business input for one Synthesis invocation."""

    model_config = ConfigDict(frozen=True)

    standalone_query: str
    intent: str
    evidence: tuple[PreparedEvidence, ...]
    data_gaps: tuple[DataGapView, ...] = ()


class FinancialResearchReport(BaseModel):
    """Minimal model-authored Markdown candidate with Evidence markers."""

    model_config = ConfigDict(frozen=True)

    markdown_report: str = Field(min_length=1)


class PublishedReport(BaseModel):
    """Code-gated Markdown and citations for the public response."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: tuple[CitationReference, ...]


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
) -> PreparedSynthesis:
    """Build the sole bounded Evidence projection Synthesis may receive."""
    evidence = tuple(
        PreparedEvidence(
            id=item.id,
            source=item.source,
            source_url=item.source_url,
            title=item.title,
            excerpt=item.excerpt,
        )
        for item in (
            catalog.resolve(
                evidence_id,
                tenant_id=tenant_id,
                request_id=request_id,
                as_of_date=as_of_date,
                max_evidence_age_days=max_evidence_age_days,
            )
            for evidence_id in accepted_evidence_ids
        )
    )
    return PreparedSynthesis(
        standalone_query=standalone_query,
        intent=intent,
        evidence=evidence,
        data_gaps=data_gaps,
    )


def publish_report(
    candidate: FinancialResearchReport,
    prepared: PreparedSynthesis,
) -> PublishedReport:
    """Validate every Evidence marker and derive public citations in code."""
    markers = _EVIDENCE_MARKER.findall(candidate.markdown_report)
    marker_text = _EVIDENCE_MARKER.sub("", candidate.markdown_report)
    if _MARKER_LIKE.search(marker_text) or not markers:
        raise ValueError("Evidence marker is invalid")
    citations: list[CitationReference] = []
    for marker in markers:
        index = int(marker)
        if index > len(prepared.evidence):
            raise ValueError("Evidence marker is not eligible")
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
    return PublishedReport(answer=candidate.markdown_report, citations=tuple(citations))
