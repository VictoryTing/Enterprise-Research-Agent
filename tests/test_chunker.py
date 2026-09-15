"""Tests for RAG text chunking."""

import pytest

from app.rag.chunker import TextChunker


def test_chunker_uses_string_evidence_id_and_overlap() -> None:
    chunker = TextChunker(
        chunk_size=10,
        overlap=3,
    )

    chunks = chunker.split(
        evidence_id="ev_test123",
        text="abcdefghijklmnop",
    )

    assert len(chunks) == 2

    assert chunks[0].chunk_id == "ev_test123-0"
    assert chunks[0].evidence_id == "ev_test123"
    assert chunks[0].text == "abcdefghij"
    assert chunks[0].start == 0
    assert chunks[0].end == 10

    assert chunks[1].chunk_id == "ev_test123-1"
    assert chunks[1].evidence_id == "ev_test123"
    assert chunks[1].text == "hijklmnop"
    assert chunks[1].start == 7
    assert chunks[1].end == 16


def test_chunker_rejects_invalid_overlap() -> None:
    with pytest.raises(
        ValueError,
        match="overlap must be smaller than chunk_size",
    ):
        TextChunker(
            chunk_size=100,
            overlap=100,
        )


def test_chunker_returns_empty_list_for_blank_text() -> None:
    chunker = TextChunker()

    chunks = chunker.split(
        evidence_id="ev_empty",
        text="   ",
    )

    assert chunks == []