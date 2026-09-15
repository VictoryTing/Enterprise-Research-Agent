"""Optional BGE reranker for evidence retrieval."""

from __future__ import annotations

from importlib import import_module

from typing import Protocol

from app.memory.evidence import Evidence


class EvidenceReranker(Protocol):
    """Interface implemented by evidence rerankers."""

    def score(
        self,
        query: str,
        evidence_items: list[Evidence],
    ) -> dict[str, float]:
        """Return one relevance score for each evidence item."""
        ...


class BGEReranker:
    """Lazy-loading wrapper around BAAI BGE reranker."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = False,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None

    def _load_model(self) -> None:
        """Load the model only when reranking is first requested."""

        if self._model is not None:
            return

        try:
            flag_embedding = import_module("FlagEmbedding")
            flag_reranker = getattr(
                flag_embedding,
                "FlagReranker",
            )
        except (ModuleNotFoundError, AttributeError) as exc:
            raise RuntimeError(
                "BGE reranking requires FlagEmbedding. "
                "Install the optional reranker dependencies first."
            ) from exc

        self._model = flag_reranker(
            self.model_name,
            use_fp16=self.use_fp16,
        )

    def score(
        self,
        query: str,
        evidence_items: list[Evidence],
    ) -> dict[str, float]:
        """Score query-evidence pairs with the BGE cross-encoder."""

        if not query.strip() or not evidence_items:
            return {}

        self._load_model()

        pairs = [
            [
                query,
                f"{evidence.title}\n{evidence.content}",
            ]
            for evidence in evidence_items
        ]

        raw_scores = self._model.compute_score(
            pairs,
            normalize=True,
        )

        if isinstance(raw_scores, (int, float)):
            raw_scores = [float(raw_scores)]

        if len(raw_scores) != len(evidence_items):
            raise RuntimeError(
                "Reranker returned an unexpected number of scores."
            )

        return {
            evidence.evidence_id: float(score)
            for evidence, score in zip(
                evidence_items,
                raw_scores,
            )
        }