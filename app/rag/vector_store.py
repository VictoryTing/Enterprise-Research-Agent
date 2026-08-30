"""In-memory vector store for semantic retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from app.core.config import LLM_API_KEY


@dataclass
class VectorDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


class VectorStore:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:

        self.client = OpenAI(
            api_key=LLM_API_KEY
        )

        self.embedding_model = embedding_model
        self.documents: list[VectorDocument] = []

    def _embed(self, text: str) -> list[float]:

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )

        return response.data[0].embedding

    @staticmethod
    def _cosine_similarity(
        a: list[float],
        b: list[float],
    ) -> float:

        if len(a) != len(b):
            raise ValueError(
                "Embedding dimensions do not match."
            )

        dot_product = sum(
            x * y
            for x, y in zip(a, b)
        )

        norm_a = math.sqrt(
            sum(x * x for x in a)
        )

        norm_b = math.sqrt(
            sum(x * x for x in b)
        )

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        embedding = self._embed(text)

        document = VectorDocument(
            doc_id=doc_id,
            text=text,
            metadata=metadata or {},
            embedding=embedding,
        )

        self.documents.append(document)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[VectorDocument, float]]:

        if not self.documents:
            return []

        query_embedding = self._embed(query)

        scored_documents = []

        for document in self.documents:

            score = self._cosine_similarity(
                query_embedding,
                document.embedding,
            )

            scored_documents.append(
                (document, score)
            )

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_documents[:top_k]