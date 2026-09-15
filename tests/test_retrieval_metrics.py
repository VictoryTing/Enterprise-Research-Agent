"""Tests for retrieval evaluation metrics."""

import pytest

from eval.retrieval_metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_finds_relevant_document() -> None:
    retrieved = ["doc_a", "doc_b", "doc_c"]
    relevant = {"doc_b"}

    assert recall_at_k(retrieved, relevant, k=1) == 0.0
    assert recall_at_k(retrieved, relevant, k=2) == 1.0


def test_recall_at_k_rejects_invalid_k() -> None:
    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        recall_at_k(
            ["doc_a"],
            {"doc_a"},
            k=0,
        )


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    retrieved = ["doc_a", "doc_b", "doc_c"]
    relevant = {"doc_b", "doc_c"}

    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_mean_reciprocal_rank() -> None:
    result = mean_reciprocal_rank(
        [1.0, 0.5, 0.0]
    )

    assert result == 0.5