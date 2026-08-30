"""Service for RAG citation extraction, validation, and fallback positioning."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai.usage import RunUsage

from app.core.model_registry import ModelRegistry
from app.models.domain import Document
from app.models.workflow import AggregatedEvidence, CitationReference

logger = logging.getLogger(__name__)

# Sentence splitting pattern as specified in design V4.2
_SENTENCE_PATTERN = re.compile(r"[^。！？\n]+?(?:[。！？\n]+|\.(?=\s|$)|$)")

# Units and key financial words for fallback scoring
KEY_TERMS = {
    "%",
    "亿",
    "万",
    "千",
    "百万",
    "十亿",
    "美元",
    "人民币",
    "元",
    "revenue",
    "income",
    "profit",
    "sales",
    "growth",
    "增长",
    "减少",
    "同比",
    "环比",
    "turnover",
    "expense",
    "cost",
    "margin",
    "ebitda",
    "ebit",
}

QUOTE_EXTRACTION_INSTRUCTIONS = (
    "You are a citation extraction agent. Your job is to extract exact verbatim passages "
    "(supporting sentences or data rows) from the provided evidence text that support each claim "
    "made in the assistant's answer.\n"
    "Instructions:\n"
    "- Return only the QuoteExtractionResult schema.\n"
    "- For each citation index [n] present in the assistant's answer, extract the matching list of quoted_passages from that evidence content.\n"
    "- The quoted_passages MUST be exact verbatim substrings of the evidence content. Do not paraphrase or edit the passages.\n"
    "- For paraphrased claims, extract the original verbatim sentences that were paraphrased.\n"
    "- For claims involving mathematical calculation or reasoning over data, extract the exact sentences/data rows containing the raw figures/data used.\n"
    "- If the evidence does not support the claim, or if you cannot find any supporting passages, return an empty list of quoted_passages.\n"
    "- Do not return the entire evidence content; return only the specific sentences or lines that are directly relevant to the claim."
)


@dataclass(frozen=True)
class ClaimExtraction:
    """A claim extracted from the answer with its citation index and position."""

    citation_index: int
    claim_text: str
    position_in_answer: int


@dataclass(frozen=True)
class LocatedSpan:
    """A localized slice of the evidence content."""

    start: int
    end: int
    text: str


class QuoteExtractionItem(BaseModel):
    """Verbatim passages extracted by LLM for a specific citation index."""

    citation_index: int
    quoted_passages: list[str] = Field(default_factory=list)


class QuoteExtractionResult(BaseModel):
    """The structured list of all extractions."""

    extractions: list[QuoteExtractionItem] = Field(
        default_factory=list[QuoteExtractionItem]
    )


def extract_claims(answer: str) -> list[ClaimExtraction]:
    """Split answer into sentences and locate citation indices [n]."""
    sentences: list[LocatedSpan] = [
        LocatedSpan(
            text=match.group(0),
            start=match.start(),
            end=match.end(),
        )
        for match in _SENTENCE_PATTERN.finditer(answer)
    ]

    if not sentences:
        return []

    claims: list[ClaimExtraction] = []
    seen_indices: set[int] = set()

    for match in re.finditer(r"\[([1-9]\d*)\]", answer):
        citation_index = int(match.group(1))
        if citation_index in seen_indices:
            continue

        match_start = match.start()
        sentence_idx = -1
        for i, s in enumerate(sentences):
            if s.start <= match_start < s.end:
                sentence_idx = i
                break

        if sentence_idx == -1:
            continue

        # If citation is near the start of the sentence (e.g. index 0-3), attribute to previous sentence
        attributed_sentence_idx = sentence_idx
        if match_start - sentences[sentence_idx].start <= 3 and sentence_idx > 0:
            attributed_sentence_idx = sentence_idx - 1

        target_sentence = sentences[attributed_sentence_idx]
        # Clean claim text: remove all citation markers, clean up space before punctuation
        clean_text = re.sub(r"\s*\[\d+\]\s*", " ", target_sentence.text).strip()
        clean_text = re.sub(r"\s+([。！？\.\?!])", r"\1", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        claims.append(
            ClaimExtraction(
                citation_index=citation_index,
                claim_text=clean_text,
                position_in_answer=target_sentence.start,
            )
        )
        seen_indices.add(citation_index)

    return claims


async def extract_quotes_with_llm(
    answer: str,
    evidence_map: dict[int, str],
    registry: ModelRegistry,
) -> tuple[dict[int, list[str]], RunUsage]:
    """Use fast LLM to extract verbatim quotes supporting each citation index."""
    evidence_block = "\n\n".join(
        f"Evidence [{idx}]:\n{content}" for idx, content in evidence_map.items()
    )
    prompt = (
        f"Assistant Answer:\n{answer}\n\n"
        f"Retrieved Evidence:\n{evidence_block}\n\n"
        "Extract the supporting verbatim passages for each citation index."
    )
    usage = RunUsage()
    try:
        agent = registry.create_agent(
            "fast",
            output_type=QuoteExtractionResult,
            instructions=QUOTE_EXTRACTION_INSTRUCTIONS,
        )
        async with agent.run_stream(prompt) as stream:
            result = await stream.get_output()
            # Accumulate usage
            run_usage = stream.usage()
            usage.requests += run_usage.requests
            usage.input_tokens += run_usage.input_tokens
            usage.output_tokens += run_usage.output_tokens
            usage.tool_calls += run_usage.tool_calls

        extractions: dict[int, list[str]] = {}
        for item in result.extractions:
            extractions[item.citation_index] = item.quoted_passages
        return extractions, usage
    except Exception as e:
        logger.exception("Error extracting quotes with LLM: %s", e)
        return {}, usage


def locate_single_passage(
    ev_content: str, psg: str, allow_split: bool = True
) -> list[LocatedSpan]:
    """Attempt to locate a single passage string inside evidence content using locator chain."""
    # 1. Exact substring
    start = ev_content.find(psg)
    if start != -1:
        return [LocatedSpan(start=start, end=start + len(psg), text=psg)]

    # 2. Relaxed whitespace
    escaped_passage = re.escape(psg)
    pattern_str = re.sub(
        r"(?:\\\s)+",
        lambda _: r"\s+",
        escaped_passage,
    )
    try:
        match = re.search(pattern_str, ev_content)
        if match:
            return [
                LocatedSpan(start=match.start(), end=match.end(), text=match.group(0))
            ]
    except re.error:
        pass

    # 3. Punctuation-normalized projection
    norm_ev: list[str] = []
    index_map: list[int] = []
    for idx, char in enumerate(ev_content):
        if char.isalnum():
            norm_ev.append(char.lower())
            index_map.append(idx)
    norm_ev_str = "".join(norm_ev)

    norm_passage = "".join(char.lower() for char in psg if char.isalnum())
    if norm_passage:
        norm_start = norm_ev_str.find(norm_passage)
        if norm_start != -1:
            norm_end = norm_start + len(norm_passage)
            raw_start = index_map[norm_start]
            raw_end = index_map[norm_end - 1] + 1
            return [
                LocatedSpan(
                    start=raw_start, end=raw_end, text=ev_content[raw_start:raw_end]
                )
            ]

    # 4. Multi-sentence split locator
    if allow_split:
        sub_sentences = [
            m.group(0) for m in _SENTENCE_PATTERN.finditer(psg) if m.group(0).strip()
        ]
        if len(sub_sentences) > 1:
            sub_spans: list[LocatedSpan] = []
            for s in sub_sentences:
                located = locate_single_passage(
                    ev_content, s.strip(), allow_split=False
                )
                if located:
                    sub_spans.extend(located)
                else:
                    return []
            return sub_spans

    return []


def validate_and_locate_quotes(
    evidence_content: str, passages: list[str]
) -> list[LocatedSpan]:
    """Validate list of quotes, sort, and merge overlapping/adjacent spans with reslicing."""
    located_spans: list[LocatedSpan] = []
    for passage in passages:
        passage = passage.strip()
        if not passage:
            continue
        spans = locate_single_passage(evidence_content, passage)
        if spans:
            located_spans.extend(spans)

    if not located_spans:
        return []

    # Sort spans by start offset
    located_spans.sort(key=lambda x: x.start)

    # Merge overlapping/adjacent intervals
    merged: list[LocatedSpan] = []
    for span in located_spans:
        if not merged:
            merged.append(span)
        else:
            last = merged[-1]
            if span.start <= last.end:
                # Merge interval
                merged_start = last.start
                merged_end = max(last.end, span.end)
                merged_text = evidence_content[merged_start:merged_end]
                merged[-1] = LocatedSpan(
                    start=merged_start, end=merged_end, text=merged_text
                )
            else:
                merged.append(span)

    return merged


def extract_numbers(text: str) -> set[str]:
    """Extract decimal, integer, and percent numbers from text."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def extract_entities(text: str) -> set[str]:
    """Extract quarters and years."""
    patterns = [r"\bQ[1-4]\b", r"\b20\d{2}\b", r"\b19\d{2}\b"]
    entities: set[str] = set()
    for p in patterns:
        entities.update(re.findall(p, text, re.IGNORECASE))
    return {e.lower() for e in entities}


def extract_key_terms(text: str) -> set[str]:
    """Extract key terms and financial units."""
    text_lower = text.lower()
    return {term for term in KEY_TERMS if term in text_lower}


def extract_keywords(text: str) -> set[str]:
    """Extract alphanumeric words and Chinese characters."""
    words = re.findall(r"\b\w{2,}\b", text.lower())
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return set(words).union(set(chinese_chars))


def fallback_locate_supporting_windows(
    claim_text: str, evidence_content: str
) -> list[LocatedSpan]:
    """Deterministically find top 2-3 non-overlapping windows of evidence supporting a claim."""
    # 1. Split into sentence matches
    sentence_matches = list(_SENTENCE_PATTERN.finditer(evidence_content))
    if not sentence_matches:
        return []

    # Also treat lines as additional sentence units to support table/JSON structures
    # We construct windows based on either sentences or lines
    lines: list[LocatedSpan] = []
    line_offset = 0
    for line in evidence_content.splitlines(keepends=True):
        if line.strip():
            lines.append(
                LocatedSpan(start=line_offset, end=line_offset + len(line), text=line)
            )
        line_offset += len(line)

    blocks = [
        LocatedSpan(start=sm.start(), end=sm.end(), text=sm.group(0))
        for sm in sentence_matches
    ]
    for ln in lines:
        # Avoid duplicate sentences matching lines exactly
        if not any(b.start == ln.start and b.end == ln.end for b in blocks):
            blocks.append(ln)

    blocks.sort(key=lambda x: x.start)

    num_claim = extract_numbers(claim_text)
    ent_claim = extract_entities(claim_text)
    term_claim = extract_key_terms(claim_text)
    kw_claim = extract_keywords(claim_text)

    candidates: list[tuple[float, int, int, str]] = []
    n = len(blocks)

    # 2. Generate candidate windows (length 1, 2, 3 blocks)
    for length in [1, 2, 3]:
        for i in range(n - length + 1):
            j = i + length
            start = blocks[i].start
            end = blocks[j - 1].end
            window_text = evidence_content[start:end]

            # Compute overlap scores
            num_window = extract_numbers(window_text)
            ent_window = extract_entities(window_text)
            term_window = extract_key_terms(window_text)
            kw_window = extract_keywords(window_text)

            num_score = (
                len(num_claim.intersection(num_window)) / len(num_claim)
                if num_claim
                else 0.0
            )
            ent_score = (
                len(ent_claim.intersection(ent_window)) / len(ent_claim)
                if ent_claim
                else 0.0
            )
            term_score = (
                len(term_claim.intersection(term_window)) / len(term_claim)
                if term_claim
                else 0.0
            )
            kw_score = (
                len(kw_claim.intersection(kw_window)) / len(kw_claim)
                if kw_claim
                else 0.0
            )

            fuzzy_sim = SequenceMatcher(None, claim_text, window_text).ratio()

            if num_claim:
                score = (
                    0.4 * num_score
                    + 0.2 * ent_score
                    + 0.15 * term_score
                    + 0.15 * kw_score
                    + 0.1 * fuzzy_sim
                )
            else:
                score = (
                    0.3 * ent_score
                    + 0.2 * term_score
                    + 0.3 * kw_score
                    + 0.2 * fuzzy_sim
                )

            # Length penalty to favor shorter windows if scores are identical
            score -= 0.01 * (length - 1)

            if score > 0.15:
                candidates.append((score, start, end, window_text))

    # Sort candidates by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Select non-overlapping windows
    selected: list[tuple[float, int, int, str]] = []
    for score, start, end, text in candidates:
        overlap = False
        for _, s_start, s_end, _ in selected:
            if start < s_end and end > s_start:
                overlap = True
                break
        if not overlap:
            selected.append((score, start, end, text))
            if len(selected) >= 3:
                break

    # Sort selected windows by start index (original order)
    selected.sort(key=lambda x: x[1])

    return [
        LocatedSpan(start=start, end=end, text=text) for _, start, end, text in selected
    ]


def py_index_to_utf16_offset(text: str, index: int) -> int:
    """Translate python string code point offset to UTF-16 code unit offset."""
    return len(text[:index].encode("utf-16-le")) // 2


def span_to_utf16(text: str, start: int, end: int) -> dict[str, int]:
    """Convert codepoint span to UTF-16 span."""
    return {
        "start": py_index_to_utf16_offset(text, start),
        "end": py_index_to_utf16_offset(text, end),
    }


def safe_parse_page_number(val: Any) -> int | None:
    """Tolerantly parse page number to int, handling stringified floats like '3.0'."""
    if val is None:
        return None
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return None


def build_evidence_index(
    evidence_items: list[AggregatedEvidence],
    documents: list[Document] | None = None,
) -> dict[int, AggregatedEvidence]:
    """Safely map citation index to AggregatedEvidence (or classic Document wrapper)."""
    evidence_by_index: dict[int, AggregatedEvidence] = {}
    if evidence_items:
        for ev in evidence_items:
            if ev.citation_index is not None:
                evidence_by_index[ev.citation_index] = ev
    elif documents:
        for idx, doc in enumerate(documents, start=1):
            source_type = (
                getattr(doc, "source_type", None)
                or doc.metadata.get("source_type")
                or "document"
            )
            page_number = (
                getattr(doc, "page_number", None)
                or doc.metadata.get("page_number")
                or doc.metadata.get("pageNumber")
            )
            section = (
                getattr(doc, "section_title", None)
                or doc.metadata.get("section_title")
                or doc.metadata.get("section")
            )

            url_val = (
                getattr(doc, "source_url", None)
                or doc.metadata.get("source_url")
                or doc.metadata.get("url")
            )
            evidence_by_index[idx] = AggregatedEvidence(
                evidence_id=doc.id,
                source=str(doc.metadata.get("source") or "classic_rag"),
                content=doc.content,
                tool_call_id="classic_rag",
                title=str(
                    doc.metadata.get("title") or getattr(doc, "title", None) or doc.id
                ),
                url=str(url_val) if url_val is not None else None,
                score=doc.score,
                metadata=doc.metadata,
                citation_index=idx,
                source_type=str(source_type) if source_type is not None else None,
                page_number=safe_parse_page_number(page_number),
                section=str(section) if section is not None else None,
            )
    return evidence_by_index


def _citation_text(evidence: AggregatedEvidence) -> str:
    """Return citeable text for document or structured evidence."""
    if evidence.content:
        return evidence.content
    if evidence.structured_facts:
        return json.dumps(
            evidence.structured_facts,
            ensure_ascii=False,
            sort_keys=True,
        )
    return ""


async def build_citations(
    answer: str,
    evidence_items: list[AggregatedEvidence],
    documents: list[Document] | None = None,
    registry: ModelRegistry | None = None,
) -> tuple[list[CitationReference], RunUsage]:
    """Orchestrate claim extraction, LLM quote verification, and fallback positioning."""
    # 1. Early Return if no claims
    claims = extract_claims(answer)
    if not claims:
        return [], RunUsage()

    evidence_by_index = build_evidence_index(evidence_items, documents)

    llm_quotes: dict[int, list[str]] = {}
    usage = RunUsage()
    if registry:
        evidence_map = {
            idx: evidence_text
            for idx, ev in evidence_by_index.items()
            if (evidence_text := _citation_text(ev))
        }
        if evidence_map:
            llm_quotes, usage = await extract_quotes_with_llm(
                answer, evidence_map, registry
            )

    citations: list[CitationReference] = []
    for claim in claims:
        evidence = evidence_by_index.get(claim.citation_index)
        # Missing evidence fault tolerance: skip safely without crashing
        if evidence is None:
            continue

        evidence_text = _citation_text(evidence)
        if not evidence_text:
            continue

        located = validate_and_locate_quotes(
            evidence_text,
            llm_quotes.get(claim.citation_index, []),
        )

        status: Literal["located", "fallback_located", "unlocated"] = "located"
        if not located:
            located = fallback_locate_supporting_windows(
                claim.claim_text,
                evidence_text,
            )
            status = "fallback_located" if located else "unlocated"

        spans = [span_to_utf16(evidence_text, span.start, span.end) for span in located]
        passages = [span.text for span in located]

        citations.append(
            CitationReference(
                index=claim.citation_index,
                evidence_id=evidence.evidence_id,
                source=evidence.source,
                source_type=evidence.source_type,
                title=evidence.title,
                url=evidence.url,
                snippet=evidence_text[:300],
                quoted_text=" ... ".join(passages) if passages else None,
                quoted_passages=passages,
                page_number=evidence.page_number,
                section=evidence.section,
                published_at=evidence.published_at,
                highlight_content=evidence_text,
                highlight_spans=spans,
                offset_encoding="utf-16",
                attribution_status=status,
                metadata=evidence.metadata,
            )
        )

    return sorted(citations, key=lambda citation: citation.index), usage
