"""Tests for saving retrieval evaluation reports."""

import json
from pathlib import Path

from eval import run_retrieval_eval


def test_save_report_preserves_content(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Redirect output to pytest's temporary directory.
    monkeypatch.setattr(
        run_retrieval_eval,
        "__file__",
        str(tmp_path / "run_retrieval_eval.py"),
    )

    report = {
        "config": {
            "reranker_enabled": True,
        },
        "summary": {
            "recall_at_1": 1.0,
        },
        "cases": [
            {
                "query": "哪个产品提供推理微服务？",
                "retrieved_ids": ["eval_nim"],
            }
        ],
    }

    output_path = run_retrieval_eval.save_report(report)

    assert output_path.exists()
    assert output_path.parent == tmp_path / "results"
    assert output_path.name.startswith("retrieval_bge_")

    with output_path.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report


def test_save_report_keeps_previous_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_retrieval_eval,
        "__file__",
        str(tmp_path / "run_retrieval_eval.py"),
    )

    report = {
        "config": {
            "reranker_enabled": False,
        },
        "cases": [],
    }

    first_path = run_retrieval_eval.save_report(report)
    first_content = first_path.read_text(encoding="utf-8")

    second_path = run_retrieval_eval.save_report(report)

    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()
    assert second_path.name.startswith("retrieval_rrf_")
    assert first_path.read_text(encoding="utf-8") == first_content