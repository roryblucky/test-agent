"""Unit tests for RAG Citation Extraction and Location Service."""

import re
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.usage import RunUsage

from app.models.workflow import AggregatedEvidence
from app.services.citation_extractor import (
    QUOTE_EXTRACTION_INSTRUCTIONS,
    QuoteExtractionItem,
    QuoteExtractionResult,
    build_citations,
    extract_claims,
    extract_quotes_with_llm,
    fallback_locate_supporting_windows,
    py_index_to_utf16_offset,
    safe_parse_page_number,
    span_to_utf16,
    validate_and_locate_quotes,
)


def test_extract_claims_decimal_sentence_splitting() -> None:
    """Decimal numbers like 15.2 are not split, and sentence is correctly attributed."""
    answer = "We saw 15.2 million users. This was a record [1]. The total count is 10. Next sentence [2]."
    claims = extract_claims(answer)

    assert len(claims) == 2
    assert claims[0].citation_index == 1
    assert claims[0].claim_text == "This was a record."

    assert claims[1].citation_index == 2
    assert claims[1].claim_text == "Next sentence."


def test_extract_claims_attribution_shift() -> None:
    """If [n] is near the start of a sentence, it attributes to the previous sentence."""
    answer = "First sentence. [1] Second sentence."
    claims = extract_claims(answer)

    assert len(claims) == 1
    assert claims[0].citation_index == 1
    assert claims[0].claim_text == "First sentence."


def test_relaxed_whitespace_regex() -> None:
    """The relaxed whitespace pattern resolves correctly."""
    escaped = re.escape("hello  world")
    pattern_str = re.sub(r'(?:\\\s)+', lambda _: r'\s+', escaped)
    assert pattern_str == r"hello\s+world"


def test_py_index_to_utf16_offset() -> None:
    """UTF-16 surrogate pairs are correctly accounted for."""
    text = "A😊B"
    idx = text.index("B")
    assert idx == 2
    assert py_index_to_utf16_offset(text, idx) == 3


def test_span_to_utf16() -> None:
    """span_to_utf16 converts correctly."""
    text = "A😊B"
    span = span_to_utf16(text, 1, 3)
    assert span == {"start": 1, "end": 4}


def test_located_spans_merging_reslicing() -> None:
    """Overlapping and adjacent spans merge correctly and text is resliced from original evidence."""
    evidence_content = "This is a very long sentence with many words."
    passages = ["a very", "very long", "many words"]
    located = validate_and_locate_quotes(evidence_content, passages)

    assert len(located) == 2
    assert located[0].start == 8
    assert located[0].end == 19
    assert located[0].text == "a very long"  # resliced from original, not string concatenated

    assert located[1].start == 34
    assert located[1].end == 44
    assert located[1].text == "many words"


def test_exact_locator() -> None:
    """Exact match locates correctly."""
    evidence = "Verbatim exact passage here."
    located = validate_and_locate_quotes(evidence, ["exact passage"])
    assert len(located) == 1
    assert located[0].text == "exact passage"
    assert located[0].start == 9


def test_relaxed_whitespace_locator() -> None:
    """Match ignores spacing differences."""
    evidence = "Verbatim exact   passage  here."
    located = validate_and_locate_quotes(evidence, ["exact passage"])
    assert len(located) == 1
    assert located[0].text == "exact   passage"


def test_punctuation_normalized_locator() -> None:
    """Match ignores punctuation differences."""
    evidence = "Verbatim exact-passage, here."
    located = validate_and_locate_quotes(evidence, ["exact passage"])
    assert len(located) == 1
    assert located[0].text == "exact-passage"


def test_split_multi_sentence_locator() -> None:
    """Multi-sentence passages are split and located separately."""
    evidence = "Sentence number one. Intermediary text. Sentence number two."
    located = validate_and_locate_quotes(evidence, ["Sentence number one. Sentence number two."])
    assert len(located) == 2
    assert located[0].text == "Sentence number one."
    assert located[1].text == "Sentence number two."


def test_fallback_windows_scoring_numeric_and_units() -> None:
    """Fallback scoring favors numeric, entities, and key terms overlap."""
    evidence = (
        "Q2 revenue was 15 million dollars.\n"
        "We achieved positive margin.\n"
        "Q3 income grew by 20% to 18 million dollars."
    )
    claim = "Revenue grew by 20% in Q3."
    located = fallback_locate_supporting_windows(claim, evidence)

    assert len(located) > 0
    assert "Q3" in located[0].text
    assert "20%" in located[0].text


@pytest.mark.asyncio
async def test_build_citations_missing_evidence_async() -> None:
    """Async: build_citations does not crash and returns empty citations when index is missing."""
    answer = "Here is claim [1]."
    citations, usage = await build_citations(answer, [])
    assert citations == []
    assert isinstance(usage, RunUsage)


@pytest.mark.asyncio
async def test_all_failures_unlocated() -> None:
    """If no exact matches and no fallback matches exceed the threshold, returns unlocated citation."""
    evidence_items = [
        AggregatedEvidence(
            evidence_id="ev-1",
            source="test",
            content="Completely irrelevant context with no numbers.",
            tool_call_id="search",
            citation_index=1,
        )
    ]
    citations, usage = await build_citations("Some claim with numbers 100% [1]", evidence_items)
    assert len(citations) == 1
    assert citations[0].attribution_status == "unlocated"
    assert citations[0].highlight_spans == []


@pytest.mark.asyncio
async def test_extract_quotes_with_llm_mock() -> None:
    """Verify extract_quotes_with_llm invokes registry agent and maps citations correctly."""
    mock_agent = MagicMock()
    mock_stream = AsyncMock()
    mock_stream.get_output = AsyncMock(return_value=QuoteExtractionResult(
        extractions=[
            QuoteExtractionItem(citation_index=1, quoted_passages=["exact quote"])
        ]
    ))
    mock_stream.usage = MagicMock(return_value=RunUsage(input_tokens=10, output_tokens=5))
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_registry = MagicMock()
    mock_registry.create_agent.return_value = mock_agent

    result, usage = await extract_quotes_with_llm(
        answer="Claim [1].",
        evidence_map={1: "exact quote content"},
        registry=mock_registry,
    )

    assert result == {1: ["exact quote"]}
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    mock_registry.create_agent.assert_called_once_with(
        "fast",
        output_type=QuoteExtractionResult,
        instructions=QUOTE_EXTRACTION_INSTRUCTIONS,
    )


def test_safe_parse_page_number() -> None:
    """Verify safe_parse_page_number tolerantly handles various float/string formats."""
    assert safe_parse_page_number("3.0") == 3
    assert safe_parse_page_number("3") == 3
    assert safe_parse_page_number(3) == 3
    assert safe_parse_page_number(3.7) == 3
    assert safe_parse_page_number(None) is None
    assert safe_parse_page_number("invalid") is None
    assert safe_parse_page_number("page 5") is None
