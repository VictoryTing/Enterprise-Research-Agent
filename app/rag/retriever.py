"""Hybrid evidence retrieval for RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.memory.evidence import Evidence, EvidenceStore
from app.rag.vector_store import VectorStore


@dataclass
class RetrievedEvidence:
    """Evidence selected for the current research query."""

    evidence: Evidence
    score: float


class EvidenceRetriever:
    """Retrieve relevant evidence using hybrid keyword + vector retrieval."""

    def __init__(
        self,
        evidence_store: EvidenceStore,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.evidence_store = evidence_store
        self.vector_store = vector_store or VectorStore()

        # Evidence already indexed in the vector store.
        self._indexed_ids: set[str] = set()

    def retrieve(
        self,
        query: str,
        task_id: int | None = None,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        """Retrieve relevant evidence using hybrid retrieval."""

        if not query.strip():
            return []

        # ---------------------------------------------------------
        # 1. Get evidence
        # ---------------------------------------------------------
        evidence_items = (
            self.evidence_store.get_by_task(task_id)
            if task_id is not None
            else self.evidence_store.get_all()
        )

        if not evidence_items:
            return []

        # ---------------------------------------------------------
        # 2. Index evidence into VectorStore
        # ---------------------------------------------------------
        for evidence in evidence_items:
            doc_id = evidence.evidence_id

            if doc_id not in self._indexed_ids:
                self.vector_store.add(
                    doc_id=doc_id,
                    text=f"{evidence.title}\n{evidence.content}",
                    metadata={
                        "evidence_id": evidence.evidence_id,
                        "task_id": evidence.task_id,
                        "title": evidence.title,
                        "source": evidence.source,
                        "url": evidence.url,
                    },
                )

                self._indexed_ids.add(doc_id)

        # ---------------------------------------------------------
        # 3. Keyword retrieval
        # ---------------------------------------------------------
        query_terms = self._tokenize(query)

        keyword_scores: dict[str, float] = {}

        for evidence in evidence_items:
            text = (
                f"{evidence.title} "
                f"{evidence.content}"
            ).lower()

            keyword_scores[evidence.evidence_id] = (
                self._keyword_score(
                    query_terms,
                    text,
                )
            )

        # ---------------------------------------------------------
        # 4. Semantic vector retrieval
        # ---------------------------------------------------------
        vector_results = self.vector_store.search(
            query=query,
            top_k=len(evidence_items),
        )

        vector_scores: dict[str, float] = {
            document.doc_id: max(0.0, score)
            for document, score in vector_results
        }

        # ---------------------------------------------------------
        # 5. Hybrid scoring
        # ---------------------------------------------------------
        scored: list[RetrievedEvidence] = []

        for evidence in evidence_items:
            evidence_id = evidence.evidence_id

            keyword_score = keyword_scores.get(
                evidence_id,
                0.0,
            )

            semantic_score = vector_scores.get(
                evidence_id,
                0.0,
            )

            # Semantic retrieval is more important,
            # while keyword matching provides lexical precision.
            hybrid_score = (
                0.7 * semantic_score
                + 0.3 * keyword_score
            )

            if hybrid_score > 0:
                scored.append(
                    RetrievedEvidence(
                        evidence=evidence,
                        score=hybrid_score,
                    )
                )

        # ---------------------------------------------------------
        # 6. Rank
        # ---------------------------------------------------------
        scored.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return scored[:top_k]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Convert text into normalized keyword tokens."""

        return {
            token
            for token in re.findall(
                r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
                text.lower(),
            )
            if len(token) > 1
        }

    @staticmethod
    def _keyword_score(
        query_terms: set[str],
        document: str,
    ) -> float:
        """Calculate a simple keyword-overlap score."""

        if not query_terms:
            return 0.0

        document_terms = EvidenceRetriever._tokenize(
            document
        )

        overlap = query_terms & document_terms

        return len(overlap) / len(query_terms)