"""Hybrid evidence retrieval for RAG."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.memory.evidence import Evidence, EvidenceStore
from app.rag.reranker import EvidenceReranker
from app.rag.vector_store import VectorStore

from app.observability.logger import get_logger

logger = get_logger("retriever")

@dataclass
class RetrievedEvidence:
    """Evidence selected and scored for the current query."""

    evidence: Evidence
    score: float #当前最终排序依据。未开启 Reranker 时等于 RRF；开启后等于 Reranker 分数
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    keyword_rank: int | None = None
    semantic_rank: int | None = None
    rrf_score: float = 0.0  # BM25 和 Vector 排名融合后的分数
    reranker_score: float | None = None  #BGE 对 Query–Evidence 的精排分数


class EvidenceRetriever:
    """Retrieve evidence using BM25, vector search and RRF fusion."""

    def __init__(
        self,
        evidence_store: EvidenceStore,
        vector_store: VectorStore | None = None,
        reranker: EvidenceReranker | None = None,
        rrf_k: int = 60,
        rerank_candidate_k: int = 10,
    ) -> None:
        self.evidence_store = evidence_store
        self.vector_store = vector_store or VectorStore()
        self.reranker = reranker
        self.rerank_candidate_k = rerank_candidate_k
        self.rrf_k = rrf_k

        # Evidence already indexed in the vector store.
        self._indexed_ids: set[str] = set()

    def retrieve(
        self,
        query: str,
        task_id: int | None = None,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        """Retrieve relevant evidence with BM25 + vector + RRF."""

        query = query.strip()

        if not query or top_k <= 0:
            return []

        evidence_items = (
            self.evidence_store.get_by_task(task_id)
            if task_id is not None
            else self.evidence_store.get_all()
        )

        if not evidence_items:
            return []

        self._index_evidence(evidence_items)

        keyword_scores = self._bm25_scores(
            query=query,
            evidence_items=evidence_items,
        )

        keyword_ranking = sorted(
            (
                (evidence_id, score)
                for evidence_id, score in keyword_scores.items()
                if score > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        keyword_ranks = {
            evidence_id: rank
            for rank, (evidence_id, _) in enumerate(
                keyword_ranking,
                start=1,
            )
        }

        vector_results = self.vector_store.search(
            query=query,
            top_k=len(evidence_items),
        )

        current_ids = {
            evidence.evidence_id
            for evidence in evidence_items
        }

        semantic_scores: dict[str, float] = {}
        semantic_ranks: dict[str, int] = {}

        semantic_rank = 1

        for document, score in vector_results:
            if document.doc_id not in current_ids:
                continue

            semantic_scores[document.doc_id] = max(0.0, score)
            semantic_ranks[document.doc_id] = semantic_rank
            semantic_rank += 1

        retrieved: list[RetrievedEvidence] = []

        for evidence in evidence_items:
            evidence_id = evidence.evidence_id
            keyword_rank = keyword_ranks.get(evidence_id)
            semantic_rank_value = semantic_ranks.get(evidence_id)

            rrf_score = 0.0

            if keyword_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + keyword_rank)

            if semantic_rank_value is not None:
                rrf_score += 1.0 / (
                    self.rrf_k + semantic_rank_value
                )

            if rrf_score == 0:
                continue

            retrieved.append(
                RetrievedEvidence(
                    evidence=evidence,
                    score=rrf_score,
                    rrf_score=rrf_score,
                    keyword_score=keyword_scores.get(
                        evidence_id,
                        0.0,
                    ),
                    semantic_score=semantic_scores.get(
                        evidence_id,
                        0.0,
                    ),
                    keyword_rank=keyword_rank,
                    semantic_rank=semantic_rank_value,
                )
            )

# ---------------------------------------------------------
# 6. Rank candidates by RRF
# ---------------------------------------------------------
        retrieved.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        candidate_count = min(
            max(top_k, self.rerank_candidate_k),
            len(retrieved),
        )

        candidates = retrieved[:candidate_count]

# ---------------------------------------------------------
# 7. Optional reranking
# ---------------------------------------------------------
        if self.reranker is not None and candidates:
            try:
                reranker_scores = self.reranker.score(
                    query=query,
                    evidence_items=[
                        item.evidence
                        for item in candidates
                    ],
                )

                for item in candidates:
                    reranker_score = reranker_scores.get(
                        item.evidence.evidence_id
                    )

                    item.reranker_score = reranker_score

                    if reranker_score is not None:
                        item.score = reranker_score

                candidates.sort(
                    key=lambda item: (
                        item.reranker_score is not None,
                        item.reranker_score
                        if item.reranker_score is not None
                        else float("-inf"),
                        item.rrf_score,
                    ),
                    reverse=True,
                )

            except Exception as exc:
                logger.warning(
                    (
                        "Reranker failed; falling back "
                        "to RRF ranking: %s"
                    ),
                    exc,
                )

                # Reset any partially updated scores.
                for item in candidates:
                    item.reranker_score = None
                    item.score = item.rrf_score

                candidates.sort(
                    key=lambda item: item.rrf_score,
                    reverse=True,
                )
        return candidates[:top_k]

    def _index_evidence(
        self,
        evidence_items: list[Evidence],
    ) -> None:
        """Index evidence that has not been embedded yet."""

        for evidence in evidence_items:
            evidence_id = evidence.evidence_id

            if evidence_id in self._indexed_ids:
                continue

            self.vector_store.add(
                doc_id=evidence_id,
                text=f"{evidence.title}\n{evidence.content}",
                metadata={
                    "evidence_id": evidence.evidence_id,
                    "task_id": evidence.task_id,
                    "title": evidence.title,
                    "source": evidence.source,
                    "url": evidence.url,
                },
            )

            self._indexed_ids.add(evidence_id)

    @classmethod
    def _bm25_scores(
        cls,
        query: str,
        evidence_items: list[Evidence],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> dict[str, float]:
        """Calculate BM25 scores for the current evidence collection."""

        query_terms = cls._tokenize(query)

        documents = {
            evidence.evidence_id: cls._tokenize(
                f"{evidence.title} {evidence.content}"
            )
            for evidence in evidence_items
        }

        if not query_terms:
            return {
                evidence_id: 0.0
                for evidence_id in documents
            }

        document_count = len(documents)

        average_length = (
            sum(len(tokens) for tokens in documents.values())
            / document_count
        )

        document_frequency: Counter[str] = Counter()

        for tokens in documents.values():
            document_frequency.update(set(tokens))

        scores: dict[str, float] = {}

        for evidence_id, tokens in documents.items():
            term_frequency = Counter(tokens)
            document_length = len(tokens)
            score = 0.0

            for term in query_terms:
                frequency = term_frequency.get(term, 0)

                if frequency == 0:
                    continue

                containing_documents = document_frequency[term]

                inverse_document_frequency = math.log(
                    1
                    + (
                        document_count
                        - containing_documents
                        + 0.5
                    )
                    / (containing_documents + 0.5)
                )

                length_normalization = (
                    1
                    - b
                    + b
                    * document_length
                    / max(average_length, 1.0)
                )

                score += inverse_document_frequency * (
                    frequency * (k1 + 1)
                ) / (
                    frequency
                    + k1 * length_normalization
                )

            scores[evidence_id] = score

        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize English words, numbers and Chinese characters."""

        return re.findall(
            r"[a-zA-Z0-9][a-zA-Z0-9_-]*|[\u4e00-\u9fff]",
            text.lower(),
        )