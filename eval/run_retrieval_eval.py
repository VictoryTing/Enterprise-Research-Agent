"""Run retrieval evaluation on the labelled dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.core.config import (
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANK_CANDIDATE_K,
)
from app.memory.evidence import Evidence, EvidenceStore
from app.rag.reranker import BGEReranker
from app.rag.retriever import EvidenceRetriever
from eval.retrieval_metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def load_dataset() -> dict:
    """Load the retrieval evaluation dataset."""

    dataset_path = Path(__file__).with_name(
        "retrieval_cases.json"
    )

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def percentile(
    values: list[float],
    percent: float,
) -> float:
    """Calculate a percentile using linear interpolation."""

    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percent
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    fraction = position - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return lower_value + (
        upper_value - lower_value
    ) * fraction

def save_report(report: dict) -> Path:
    """Save each evaluation run without overwriting previous results."""

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    mode = (
        "bge"
        if report["config"]["reranker_enabled"]
        else "rrf"
    )

    output_path = output_dir / (
        f"retrieval_{mode}_{timestamp}.json"
    )

    with output_path.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    return output_path

def main() -> None:
    dataset = load_dataset()

    evidence_store = EvidenceStore()

    for document in dataset["documents"]:
        evidence_store.add(
            Evidence(
                task_id=document["task_id"],
                title=document["title"],
                url=document["url"],
                content=document["content"],
                evidence_id=document["evidence_id"],
            )
        )

    reranker = (
        BGEReranker(
            model_name=RERANKER_MODEL,
            use_fp16=False,
        )
        if RERANKER_ENABLED
        else None
    )

    retriever = EvidenceRetriever(
        evidence_store=evidence_store,
        reranker=reranker,
        rerank_candidate_k=RERANK_CANDIDATE_K,
    )

    recall_at_1_scores: list[float] = []
    recall_at_3_scores: list[float] = []
    recall_at_5_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    case_results: list[dict] = []

    print(
        "Reranker:",
        "enabled" if reranker is not None else "disabled",
    )
    print()

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------
    # The first retrieval may include document indexing,
    # embedding initialization and reranker model loading.
    warmup_start = perf_counter()

    retriever.retrieve(
        query="NVIDIA enterprise AI platform",
        task_id=1,
        top_k=5,
    )

    warmup_latency_ms = (
        perf_counter() - warmup_start
    ) * 1000

    print(
        f"Warm-up latency: "
        f"{warmup_latency_ms:.2f} ms"
    )
    print()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    for case in dataset["queries"]:
        start_time = perf_counter()

        results = retriever.retrieve(
            query=case["query"],
            task_id=1,
            top_k=5,
        )

        latency_ms = (
            perf_counter() - start_time
        ) * 1000

        retrieved_ids = [
            item.evidence.evidence_id
            for item in results
        ]

        relevant_ids = set(case["relevant_ids"])

        recall_1 = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=1,
        )

        recall_3 = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=3,
        )

        recall_5 = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=5,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        recall_at_1_scores.append(recall_1)
        recall_at_3_scores.append(recall_3)
        recall_at_5_scores.append(recall_5)
        reciprocal_ranks.append(rr)
        latencies_ms.append(latency_ms)
        case_results.append(
            {
                "query_id": case["query_id"],
                "query": case["query"],
                "relevant_ids": sorted(relevant_ids),
                "retrieved_ids": retrieved_ids,
                "recall_at_1": recall_1,
                "recall_at_3": recall_3,
                "recall_at_5": recall_5,
                "reciprocal_rank": rr,
                "latency_ms": latency_ms,
            }
        )

        print(f"[{case['query_id']}] {case['query']}")
        print(f"Retrieved: {retrieved_ids}")
        print(
            f"Recall@1={recall_1:.3f}, "
            f"Recall@3={recall_3:.3f}, "
            f"RR={rr:.3f}, "
            f"Latency={latency_ms:.2f} ms"
        )
        print()

    query_count = len(dataset["queries"])

    if query_count == 0:
        print("No evaluation queries found.")
        return

    average_recall_1 = (
        sum(recall_at_1_scores) / query_count
    )

    average_recall_3 = (
        sum(recall_at_3_scores) / query_count
    )

    average_recall_5 = (
        sum(recall_at_5_scores) / query_count
    )

    average_latency = (
        sum(latencies_ms) / query_count
    )

    p50_latency = percentile(
        latencies_ms,
        0.50,
    )

    p95_latency = percentile(
        latencies_ms,
        0.95,
    )

    mrr = mean_reciprocal_rank(
        reciprocal_ranks
    )

    print("=== Overall Results ===")
    print(f"Queries:       {query_count}")
    print(f"Recall@1:      {average_recall_1:.3f}")
    print(f"Recall@3:      {average_recall_3:.3f}")
    print(f"Recall@5:      {average_recall_5:.3f}")
    print(f"MRR:           {mrr:.3f}")
    print(
        f"Warm-up:       "
        f"{warmup_latency_ms:.2f} ms"
    )
    print(
        f"Avg latency:   "
        f"{average_latency:.2f} ms"
    )
    print(
        f"P50 latency:   "
        f"{p50_latency:.2f} ms"
    )
    print(
        f"P95 latency:   "
        f"{p95_latency:.2f} ms"
    )
    report = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": {
            "name": "retrieval_cases.json",
            "document_count": len(dataset["documents"]),
            "query_count": query_count,
        },
        "config": {
            "reranker_enabled": reranker is not None,
            "reranker_model": (
                RERANKER_MODEL
                if reranker is not None
                else None
            ),
            "rerank_candidate_k": RERANK_CANDIDATE_K,
            "top_k": 5,
            "task_id": 1,
        },
        "summary": {
            "recall_at_1": average_recall_1,
            "recall_at_3": average_recall_3,
            "recall_at_5": average_recall_5,
            "mrr": mrr,
            "warmup_latency_ms": warmup_latency_ms,
            "avg_latency_ms": average_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
        },
        "cases": case_results,
    }

    output_path = save_report(report)
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()