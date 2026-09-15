"""Retrieval evaluation metrics."""

from __future__ import annotations


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate the proportion of relevant items found in top-k."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        return 0.0

    retrieved_top_k = set(retrieved_ids[:k])
    matched = retrieved_top_k & relevant_ids

    return len(matched) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""

    for rank, evidence_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if evidence_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    reciprocal_ranks: list[float],
) -> float:
    """Calculate the mean reciprocal rank across queries."""

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(
        reciprocal_ranks
    )