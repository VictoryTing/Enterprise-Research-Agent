"""Tests for BM25 and RRF hybrid evidence retrieval."""

from types import SimpleNamespace

from app.memory.evidence import Evidence, EvidenceStore
from app.rag.retriever import EvidenceRetriever


class FakeVectorStore:
    """A deterministic vector store that never calls an external API."""

    def __init__(self, ranked_ids: list[str]) -> None:
        self.ranked_ids = ranked_ids
        self.documents: dict[str, object] = {}

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        self.documents[doc_id] = SimpleNamespace(
            doc_id=doc_id,
            text=text,
            metadata=metadata or {},
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[object, float]]:
        del query

        results = []

        for rank, doc_id in enumerate(
            self.ranked_ids[:top_k],
            start=1,
        ):
            document = self.documents[doc_id]
            score = 1.0 - rank * 0.1
            results.append((document, score))

        return results

class FakeReranker:
    """A deterministic reranker that never loads a real model."""

    def __init__(
        self,
        scores: dict[str, float],
    ) -> None:
        self.scores = scores

    def score(
        self,
        query: str,
        evidence_items: list[Evidence],
    ) -> dict[str, float]:
        del query

        return {
            item.evidence_id: self.scores[item.evidence_id]
            for item in evidence_items
        }

class FailingReranker:
    """A reranker that always fails for fallback testing."""

    def score(
        self,
        query: str,
        evidence_items: list[Evidence],
    ) -> dict[str, float]:
        del query
        del evidence_items

        raise RuntimeError("Simulated reranker failure")

def make_evidence(
    title: str,
    content: str,
    url: str,
) -> Evidence:
    return Evidence(
        task_id=1,
        title=title,
        content=content,
        url=url,
    )


def test_bm25_ranks_relevant_document_higher() -> None:
    relevant = make_evidence(
        title="NVIDIA Blackwell",
        content="NVIDIA Blackwell accelerates AI infrastructure.",
        url="https://example.com/nvidia",
    )

    irrelevant = make_evidence(
        title="Apple iPhone",
        content="Apple released a new smartphone.",
        url="https://example.com/apple",
    )

    scores = EvidenceRetriever._bm25_scores(
        query="NVIDIA AI infrastructure",
        evidence_items=[relevant, irrelevant],
    )

    assert scores[relevant.evidence_id] > 0
    assert scores[irrelevant.evidence_id] == 0
    assert (
        scores[relevant.evidence_id]
        > scores[irrelevant.evidence_id]
    )


def test_rrf_combines_keyword_and_vector_rankings() -> None:
    evidence_store = EvidenceStore()

    keyword_first = make_evidence(
        title="NVIDIA Blackwell Infrastructure",
        content="Blackwell powers NVIDIA AI infrastructure.",
        url="https://example.com/blackwell",
    )

    shared_result = make_evidence(
        title="NVIDIA AI Platform",
        content="NVIDIA provides an AI computing platform.",
        url="https://example.com/platform",
    )

    semantic_only = make_evidence(
        title="Accelerated Computing",
        content="GPU systems support modern data centers.",
        url="https://example.com/computing",
    )

    evidence_store.add(keyword_first)
    evidence_store.add(shared_result)
    evidence_store.add(semantic_only)

    fake_vector_store = FakeVectorStore(
        ranked_ids=[
            shared_result.evidence_id,
            semantic_only.evidence_id,
            keyword_first.evidence_id,
        ]
    )

    retriever = EvidenceRetriever(
        evidence_store=evidence_store,
        vector_store=fake_vector_store,
    )

    results = retriever.retrieve(
        query="NVIDIA AI infrastructure",
        task_id=1,
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].evidence.evidence_id == (
        shared_result.evidence_id
    )
    assert results[0].keyword_rank is not None
    assert results[0].semantic_rank == 1
    assert results[0].score > results[1].score

def test_reranker_can_change_rrf_order() -> None:
    evidence_store = EvidenceStore()

    nvidia = make_evidence(
        title="NVIDIA AI Infrastructure",
        content="NVIDIA provides GPU infrastructure for AI agents.",
        url="https://example.com/nvidia",
    )

    apple = make_evidence(
        title="Apple AI Infrastructure",
        content="Apple develops infrastructure and devices.",
        url="https://example.com/apple",
    )

    evidence_store.add(nvidia)
    evidence_store.add(apple)

    fake_vector_store = FakeVectorStore(
        ranked_ids=[
            nvidia.evidence_id,
            apple.evidence_id,
        ]
    )

    fake_reranker = FakeReranker(
        scores={
            nvidia.evidence_id: 0.1,
            apple.evidence_id: 0.9,
        }
    )

    retriever = EvidenceRetriever(
        evidence_store=evidence_store,
        vector_store=fake_vector_store,
        reranker=fake_reranker,
        rerank_candidate_k=2,
    )

    results = retriever.retrieve(
        query="NVIDIA AI infrastructure",
        task_id=1,
        top_k=2,
    )

    assert len(results) == 2

    # Reranker promotes Apple above the original RRF winner.
    assert results[0].evidence.evidence_id == apple.evidence_id
    assert results[0].reranker_score == 0.9
    assert results[0].score == 0.9

    # The original first-stage score remains available.
    assert results[0].rrf_score > 0

def test_reranker_failure_falls_back_to_rrf() -> None:
    evidence_store = EvidenceStore()

    first = make_evidence(
        title="NVIDIA AI Infrastructure",
        content="NVIDIA provides infrastructure for AI agents.",
        url="https://example.com/nvidia",
    )

    second = make_evidence(
        title="Enterprise Computing Platform",
        content="GPU systems support enterprise workloads.",
        url="https://example.com/platform",
    )

    evidence_store.add(first)
    evidence_store.add(second)

    ranked_ids = [
        first.evidence_id,
        second.evidence_id,
    ]

    # Obtain the expected result using RRF only.
    baseline_retriever = EvidenceRetriever(
        evidence_store=evidence_store,
        vector_store=FakeVectorStore(
            ranked_ids=ranked_ids
        ),
    )

    baseline_results = baseline_retriever.retrieve(
        query="NVIDIA AI infrastructure",
        task_id=1,
        top_k=2,
    )

    # Run the same retrieval with a failing reranker.
    fallback_retriever = EvidenceRetriever(
        evidence_store=evidence_store,
        vector_store=FakeVectorStore(
            ranked_ids=ranked_ids
        ),
        reranker=FailingReranker(),
        rerank_candidate_k=2,
    )

    fallback_results = fallback_retriever.retrieve(
        query="NVIDIA AI infrastructure",
        task_id=1,
        top_k=2,
    )

    assert len(fallback_results) == 2

    # The fallback order must equal the original RRF order.
    assert [
        item.evidence.evidence_id
        for item in fallback_results
    ] == [
        item.evidence.evidence_id
        for item in baseline_results
    ]

    # Reranker scores must be cleared after failure.
    assert all(
        item.reranker_score is None
        for item in fallback_results
    )

    # Final scores must return to the original RRF scores.
    assert all(
        item.score == item.rrf_score
        for item in fallback_results
    )