"""Request-local Evidence acceptance and deterministic citation gates."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from app.models.workflow import CitationReference

_EVIDENCE_MARKER = re.compile(r"\[\[E:([1-9][0-9]*)\]\]")


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
        tenant_id: str,
        request_id: str,
        task_id: str,
    ) -> None:
        """Accept exact successful provenance referenced by the terminal Finding."""
        returned_items = tuple(returned)
        returned_by_id = {evidence.id: evidence for evidence in returned_items}
        if len(returned_by_id) != len(returned_items):
            raise ValueError("Evidence provenance conflicts")
        for evidence_id in finding_evidence_ids:
            evidence = returned_by_id.get(evidence_id)
            if evidence is None:
                raise ValueError("Evidence provenance is missing")
            if (
                evidence.tenant_id != tenant_id
                or evidence.request_id != request_id
                or evidence.task_id != task_id
            ):
                raise ValueError("Evidence provenance is not eligible")
            existing = self._evidence.get(evidence.id)
            if existing is not None and existing != evidence:
                raise ValueError("Evidence body conflicts")
            self._evidence[evidence.id] = evidence

    def resolve(
        self,
        evidence_id: str,
        *,
        tenant_id: str,
        request_id: str,
    ) -> EvidenceEnvelope:
        """Return only Evidence owned by this active Tenant Request."""
        evidence = self._evidence.get(evidence_id)
        if (
            evidence is None
            or evidence.tenant_id != tenant_id
            or evidence.request_id != request_id
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


class FinancialResearchReport(BaseModel):
    """Minimal model-authored Markdown candidate with Evidence markers."""

    model_config = ConfigDict(frozen=True)

    markdown: str = Field(min_length=1)


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
            )
            for evidence_id in accepted_evidence_ids
        )
    )
    return PreparedSynthesis(
        standalone_query=standalone_query,
        intent=intent,
        evidence=evidence,
    )


def publish_report(
    candidate: FinancialResearchReport,
    prepared: PreparedSynthesis,
) -> PublishedReport:
    """Validate every Evidence marker and derive public citations in code."""
    markers = _EVIDENCE_MARKER.findall(candidate.markdown)
    marker_text = _EVIDENCE_MARKER.sub("", candidate.markdown)
    if "[[E:" in marker_text or not markers:
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
    return PublishedReport(answer=candidate.markdown, citations=tuple(citations))
