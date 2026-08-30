"""Text chunking utilities for RAG."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: str
    evidence_id: int
    text: str
    start: int
    end: int


class TextChunker:
    """Split evidence content into overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        evidence_id: int,
        text: str,
    ) -> list[TextChunk]:

        text = text.strip()

        if not text:
            return []

        chunks: list[TextChunk] = []

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{evidence_id}-{chunk_index}",
                        evidence_id=evidence_id,
                        text=chunk_text,
                        start=start,
                        end=end,
                    )
                )

            if end >= len(text):
                break

            start = end - self.overlap
            chunk_index += 1

        return chunks