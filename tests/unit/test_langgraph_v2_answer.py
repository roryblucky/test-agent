from app.langgraph_v2.answer import split_answer_chunks


def test_split_answer_chunks_preserves_text_and_prefers_boundaries() -> None:
    answer = "First sentence. Second sentence\nThird; final"

    chunks = split_answer_chunks(answer)

    assert chunks == ["First sentence.", " Second sentence\n", "Third;", " final"]
    assert "".join(chunks) == answer


def test_split_answer_chunks_hard_splits_at_unicode_codepoint_limit() -> None:
    answer = "界" * 241

    chunks = split_answer_chunks(answer)

    assert [len(chunk) for chunk in chunks] == [240, 1]
    assert "".join(chunks) == answer


def test_split_answer_chunks_normalizes_only_crlf() -> None:
    answer = "A\r\nB\rC"

    chunks = split_answer_chunks(answer)

    assert chunks == ["A\n", "B\rC"]
    assert "".join(chunks) == "A\nB\rC"
