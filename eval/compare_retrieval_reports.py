"""Compare two saved retrieval evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def index_cases(report: dict) -> dict:
    cases = {}

    for case in report["cases"]:
        query_id = case["query_id"]

        if query_id in cases:
            raise ValueError(
                f"Duplicate query_id: {query_id}"
            )

        cases[query_id] = case

    return cases


def first_relevant_rank(case: dict) -> int | None:
    relevant_ids = set(case["relevant_ids"])

    for rank, doc_id in enumerate(
        case["retrieved_ids"],
        start=1,
    ):
        if doc_id in relevant_ids:
            return rank

    return None


def compare_reports(
    baseline: dict,
    reranked: dict,
) -> None:
    baseline_cases = index_cases(baseline)
    reranked_cases = index_cases(reranked)

    if baseline_cases.keys() != reranked_cases.keys():
        raise ValueError("The query IDs do not match.")

    for key in ("top_k", "task_id", "rerank_candidate_k"):
        if baseline["config"][key] != reranked["config"][key]:
            raise ValueError(f"Configuration mismatch: {key}")

    for query_id, case in baseline_cases.items():
        other = reranked_cases[query_id]

        if (
            case["query"] != other["query"]
            or set(case["relevant_ids"])
            != set(other["relevant_ids"])
        ):
            raise ValueError(
                f"Query or labels differ: {query_id}"
            )

    print("=== Metric Comparison ===")
    print(
        f"{'Metric':<20}"
        f"{'RRF':>12}"
        f"{'BGE':>12}"
        f"{'Delta':>12}"
    )

    metrics = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
    ]

    for metric in metrics:
        before = baseline["summary"][metric]
        after = reranked["summary"][metric]

        print(
            f"{metric:<20}"
            f"{before:>12.3f}"
            f"{after:>12.3f}"
            f"{after - before:>+12.3f}"
        )

    improved = 0
    regressed = 0
    unchanged = 0

    print()
    print("=== Relevant Document Rank Changes ===")

    for query_id, case in baseline_cases.items():
        other = reranked_cases[query_id]

        before_rank = first_relevant_rank(case)
        after_rank = first_relevant_rank(other)

        before_rr = (
            1.0 / before_rank
            if before_rank is not None
            else 0.0
        )
        after_rr = (
            1.0 / after_rank
            if after_rank is not None
            else 0.0
        )

        if after_rr > before_rr:
            improved += 1
            status = "IMPROVED"
        elif after_rr < before_rr:
            regressed += 1
            status = "REGRESSED"
        else:
            unchanged += 1
            continue

        print(
            f"{query_id}: {status}, "
            f"rank {before_rank} -> {after_rank}"
        )
        print(f"Query: {case['query']}")
        print()

    print(
        f"Improved: {improved}, "
        f"Regressed: {regressed}, "
        f"Unchanged: {unchanged}"
    )

    print(
        "\nNote: latency comes from separate runs; "
        "environment and network conditions may differ."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare saved RRF and BGE reports."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("reranked", type=Path)
    args = parser.parse_args()

    baseline = load_report(args.baseline)
    reranked = load_report(args.reranked)

    if baseline["config"]["reranker_enabled"]:
        raise ValueError("Baseline must have reranker disabled.")

    if not reranked["config"]["reranker_enabled"]:
        raise ValueError("BGE report must have reranker enabled.")

    compare_reports(baseline, reranked)


if __name__ == "__main__":
    main()