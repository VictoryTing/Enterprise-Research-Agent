"""Evidence models and an in-memory store for a single research run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable
import uuid


@dataclass
class Evidence:
    task_id: int
    title: str
    url: str
    content: str
    source: str = "web_search"
    relevance_score: float = 0.0
    source_quality: float = 0.0
    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    collected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)



class EvidenceStore:
    """Deduplicates and exposes evidence collected during one run."""

    def __init__(self) -> None:
        self._evidence: list[Evidence] = []
        self._seen_urls: set[str] = set()

    def add(self, evidence: Evidence) -> bool:
        normalized_url = evidence.url.strip().lower()

        if normalized_url and normalized_url in self._seen_urls:
            return False

        if normalized_url:
            self._seen_urls.add(normalized_url)

        self._evidence.append(evidence)
        return True

    def add_search_results(
        self,
        task_id: int,
        result: dict,
    ) -> list[Evidence]:
        added: list[Evidence] = []

        for item in result.get("results", []):
            evidence = Evidence(
                task_id=task_id,
                title=item.get("title", "Untitled source"),
                url=item.get("url", ""),
                content=item.get("snippet", ""),
            )

            if self.add(evidence):
                added.append(evidence)

        return added

    def get_all(self) -> list[Evidence]:
        return list(self._evidence)

    def get_by_task(self, task_id: int) -> list[Evidence]:
        return [
            item
            for item in self._evidence
            if item.task_id == task_id
        ]

    def get_by_ids(self, evidence_ids: Iterable[str]) -> list[Evidence]:
        selected = set(evidence_ids)

        return [
            item
            for item in self._evidence
            if item.evidence_id in selected
        ]

    def count(self) -> int:
        return len(self._evidence)

    def get_ranked(
        self,
        task_id: int | None = None,
        top_k: int | None = None,
    ) -> list[Evidence]:
        """Return evidence ranked by relevance and source quality."""

        evidence = self._evidence

        if task_id is not None:
            evidence = [
                item
                for item in evidence
                if item.task_id == task_id
            ]

        ranked = sorted(
            evidence,
            key=lambda item: (
                item.relevance_score * 0.7
                + item.source_quality * 0.3
            ),
            reverse=True,
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked